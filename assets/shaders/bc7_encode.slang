module bc7_encode;

// Prototype BC7 encoder: one thread per 4x4 block, mode 6 only (one subset, 7-bit RGBA endpoints with a p-bit each, 16
// four-bit indices). Mode 6 is the all-rounder: it handles gradients and alpha, and loses to the multi-subset modes
// only where a block holds two unrelated colours.
//
// Per block: a principal axis from the 4x4 colour covariance picks the first endpoints, two rounds of least squares
// refine them against the indices they imply, the endpoints are quantised with the better p-bit, and every texel takes
// the closest of the 16 interpolated colours. Layout and tables are from the Khronos Data Format Specification, 20.1.

// A batch is many images (and mip levels) in one dispatch. Each is a job: its first texel in `src`, its size, and the
// number of the first block it owns. The blocks of all jobs are numbered one after another, so a thread's block number
// says both which job it works on and where its output goes.
struct Bc7PushConstants
{
    uint* src; // RGBA8, one uint per texel, every job's rows one after another
    uint4* dst; // one uint4 per block, jobs in order
    uint4* jobs; // two uint4 per job: (first texel, width, height, blocks_x), (first block, 0, 0, 0)
    uint job_count;
    uint total_blocks;
};

[vk::push_constant] Bc7PushConstants pc;

static const uint bc7_weights[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

float4 unpack_texel(uint packed)
{
    return float4(float(packed & 0xFF), float((packed >> 8) & 0xFF), float((packed >> 16) & 0xFF),
                  float((packed >> 24) & 0xFF));
}

float4 interpolate(float4 e0, float4 e1, uint index)
{
    const float w = float(bc7_weights[index]);

    return floor(((64.0 - w) * e0 + w * e1 + 32.0) / 64.0);
}

// The index whose interpolated colour is closest to `texel`, searched around the projection onto the endpoint segment.
uint best_index(float4 texel, float4 e0, float4 e1, out float error)
{
    const float4 axis = e1 - e0;
    const float length_squared = dot(axis, axis);
    float u = length_squared > 1e-4 ? dot(texel - e0, axis) / length_squared : 0.0;
    const int guess = int(clamp(round(u * 15.0), 0.0, 15.0));

    uint best = 0;
    float best_error = 1e30;

    for (int candidate = max(guess - 1, 0); candidate <= min(guess + 1, 15); ++candidate) {
        const float4 delta = texel - interpolate(e0, e1, uint(candidate));
        const float candidate_error = dot(delta, delta);

        if (candidate_error < best_error) {
            best_error = candidate_error;
            best = uint(candidate);
        }
    }

    error = best_error;

    return best;
}

// 7-bit value plus p-bit, as the 8-bit value the decoder reconstructs.
float4 quantise_endpoint(float4 endpoint, uint pbit, out uint4 stored)
{
    const float4 q = clamp(round((endpoint - float(pbit)) * 0.5), 0.0, 127.0);

    stored = uint4(q);

    return q * 2.0 + float(pbit);
}

float4 quantise_best(float4 endpoint, out uint4 stored, out uint pbit)
{
    uint4 stored0;
    uint4 stored1;
    const float4 v0 = quantise_endpoint(endpoint, 0, stored0);
    const float4 v1 = quantise_endpoint(endpoint, 1, stored1);
    const float4 d0 = v0 - endpoint;
    const float4 d1 = v1 - endpoint;

    if (dot(d1, d1) < dot(d0, d0)) {
        stored = stored1;
        pbit = 1;

        return v1;
    }

    stored = stored0;
    pbit = 0;

    return v0;
}

// Writes `bits` low bits of `value` at `position` of the 128-bit block, least significant bit first.
void put_bits(inout uint4 block, uint position, uint value, uint bits)
{
    const uint word = position >> 5;
    const uint shift = position & 31;
    const uint mask = bits == 32 ? 0xFFFFFFFF : ((1u << bits) - 1u);
    const uint masked = value & mask;

    block[word] |= masked << shift;

    if (shift + bits > 32) {
        block[word + 1] |= masked >> (32 - shift);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main_cs(uint3 dispatch_id: SV_DispatchThreadID)
{
    const uint block_id = dispatch_id.x;

    if (block_id >= pc.total_blocks) {
        return;
    }

    // The last job that starts at or before this block.
    uint low = 0;
    uint high = pc.job_count - 1;

    while (low < high) {
        const uint middle = (low + high + 1) >> 1;

        if (pc.jobs[middle * 2 + 1].x <= block_id) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }

    const uint4 job = pc.jobs[low * 2];
    const uint first_texel = job.x;
    const uint width = job.y;
    const uint height = job.z;
    const uint local_block = block_id - pc.jobs[low * 2 + 1].x;
    const uint block_x = local_block % job.w;
    const uint block_y = local_block / job.w;

    float4 texels[16];
    float4 mean = float4(0.0);

    for (uint i = 0; i < 16; ++i) {
        const uint x = min(block_x * 4 + (i & 3), width - 1);
        const uint y = min(block_y * 4 + (i >> 2), height - 1);

        texels[i] = unpack_texel(pc.src[first_texel + y * width + x]);
        mean += texels[i];
    }

    mean *= 1.0 / 16.0;

    // Covariance of the 16 texels, then a few power iterations for its principal axis.
    float4x4 covariance = float4x4(0.0);

    for (uint i = 0; i < 16; ++i) {
        const float4 d = texels[i] - mean;

        covariance += float4x4(d.x * d, d.y * d, d.z * d, d.w * d);
    }

    float4 axis = float4(1.0, 1.0, 1.0, 1.0);

    for (uint i = 0; i < 4; ++i) {
        axis = mul(covariance, axis);

        const float length_axis = length(axis);

        axis = length_axis > 1e-6 ? axis / length_axis : float4(0.5, 0.5, 0.5, 0.5);
    }

    float t_min = 1e30;
    float t_max = -1e30;
    float4 e0 = mean;
    float4 e1 = mean;

    for (uint i = 0; i < 16; ++i) {
        const float t = dot(texels[i] - mean, axis);

        if (t < t_min) {
            t_min = t;
            e0 = texels[i];
        }

        if (t > t_max) {
            t_max = t;
            e1 = texels[i];
        }
    }

    // Refine: indices for the current endpoints, then the endpoints that fit those indices best (least squares).
    for (uint round_index = 0; round_index < 2; ++round_index) {
        float a = 0.0;
        float b = 0.0;
        float c = 0.0;
        float4 alpha = float4(0.0);
        float4 beta = float4(0.0);

        for (uint i = 0; i < 16; ++i) {
            float unused_error;
            const uint index = best_index(texels[i], e0, e1, unused_error);
            const float w = float(bc7_weights[index]) / 64.0;

            a += (1.0 - w) * (1.0 - w);
            b += w * (1.0 - w);
            c += w * w;
            alpha += (1.0 - w) * texels[i];
            beta += w * texels[i];
        }

        const float determinant = a * c - b * b;

        if (abs(determinant) > 1e-4) {
            e0 = clamp((c * alpha - b * beta) / determinant, 0.0, 255.0);
            e1 = clamp((a * beta - b * alpha) / determinant, 0.0, 255.0);
        }
    }

    uint4 stored0;
    uint4 stored1;
    uint pbit0;
    uint pbit1;
    const float4 q0 = quantise_best(e0, stored0, pbit0);
    const float4 q1 = quantise_best(e1, stored1, pbit1);

    uint indices[16];

    for (uint i = 0; i < 16; ++i) {
        float unused_error;

        indices[i] = best_index(texels[i], q0, q1, unused_error);
    }

    // The first texel's index is stored in three bits, so it must be below 8: swap the endpoints if it is not.
    uint4 first = stored0;
    uint4 second = stored1;
    uint first_pbit = pbit0;
    uint second_pbit = pbit1;

    if (indices[0] >= 8) {
        first = stored1;
        second = stored0;
        first_pbit = pbit1;
        second_pbit = pbit0;

        for (uint i = 0; i < 16; ++i) {
            indices[i] = 15 - indices[i];
        }
    }

    uint4 block = uint4(0);

    put_bits(block, 0, 64, 7); // mode 6: six zero bits, then a one
    put_bits(block, 7, first.x, 7);
    put_bits(block, 14, second.x, 7);
    put_bits(block, 21, first.y, 7);
    put_bits(block, 28, second.y, 7);
    put_bits(block, 35, first.z, 7);
    put_bits(block, 42, second.z, 7);
    put_bits(block, 49, first.w, 7);
    put_bits(block, 56, second.w, 7);
    put_bits(block, 63, first_pbit, 1);
    put_bits(block, 64, second_pbit, 1);

    put_bits(block, 65, indices[0], 3);

    for (uint i = 1; i < 16; ++i) {
        put_bits(block, 68 + (i - 1) * 4, indices[i], 4);
    }

    pc.dst[block_id] = block;
}
