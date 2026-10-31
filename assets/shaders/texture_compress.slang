module texture_compress;

// GPU texture block compressor: one thread per 4x4 block.
//
// BC7 tries two modes per block and keeps the one with less error:
//   mode 6  one subset, 7-bit RGBA endpoints with a p-bit each, 16 four-bit indices: the all-rounder, good for gradients;
//   mode 5  7-bit RGB endpoints with two-bit indices and, separately, 8-bit alpha endpoints with their own two-bit
//           indices: good where alpha changes independently of colour, such as the hard edge of a cut-out.
// Each fits its endpoints the same way: a principal axis from the covariance of the texels picks the first pair, two
// rounds of least squares refine them against the indices they imply, they are quantised, and every texel takes the
// closest interpolated value. BC5 is two BC4 blocks. Layout and tables are from the Khronos Data Format Specification,
// 20.1.

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
    uint format; // 0: BC7, 1: BC5
};

static const uint format_bc5 = 1;

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

// One BC4 block (8 bytes) of `values`, written at bit `base` of `block`: the extremes as endpoints, so they are exact, and
// the six values between them at even steps (the eight-value mode, endpoint 0 above endpoint 1) as the other palette
// entries. When every value is equal the endpoints are equal and every index is 0.
void put_bc4(inout uint4 block, uint base, float values[16])
{
    float low = 255.0;
    float high = 0.0;

    for (uint i = 0; i < 16; ++i) {
        low = min(low, values[i]);
        high = max(high, values[i]);
    }

    const float e0 = high;
    const float e1 = low;

    float palette[8];

    palette[0] = e0;
    palette[1] = e1;

    for (uint i = 2; i < 8; ++i) {
        palette[i] = (float(8 - i) * e0 + float(i - 1) * e1) / 7.0;
    }

    put_bits(block, base, uint(e0), 8);
    put_bits(block, base + 8, uint(e1), 8);

    for (uint texel = 0; texel < 16; ++texel) {
        uint best = 0;
        float best_error = 1e30;

        for (uint candidate = 0; candidate < 8; ++candidate) {
            const float error = abs(values[texel] - palette[candidate]);

            if (error < best_error) {
                best_error = error;
                best = candidate;
            }
        }

        put_bits(block, base + 16 + texel * 3, best, 3);
    }
}

static const uint bc7_weights2[4] = {0, 21, 43, 64};

// One mode's encoding of a block, and the squared error it leaves (over all four channels).
struct Candidate
{
    uint4 block;
    float error;
};

// The 8-bit value a decoder reconstructs from a 7-bit endpoint with no p-bit: the top bit is replicated into the low one.
float expand7(uint q)
{
    return float((q << 1) | (q >> 6));
}

uint quantise7(float value)
{
    return uint(clamp(round(value * 127.0 / 255.0), 0.0, 127.0));
}

float interpolate_scalar(float e0, float e1, uint weight)
{
    return floor(((64.0 - float(weight)) * e0 + float(weight) * e1 + 32.0) / 64.0);
}

float3 interpolate_rgb(float3 e0, float3 e1, uint weight)
{
    return floor(((64.0 - float(weight)) * e0 + float(weight) * e1 + 32.0) / 64.0);
}

// Mode 6: one subset of 7-bit RGBA endpoints with a p-bit each and sixteen four-bit indices.
Candidate encode_mode6(float4 texels[16])
{
    float4 mean = float4(0.0);

    for (uint i = 0; i < 16; ++i) {
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
    float error = 0.0;

    for (uint i = 0; i < 16; ++i) {
        float texel_error;

        indices[i] = best_index(texels[i], q0, q1, texel_error);
        error += texel_error;
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

    Candidate result;

    result.block = block;
    result.error = error;

    return result;
}

// Mode 5: 7-bit RGB endpoints with two-bit indices, and 8-bit alpha endpoints with their own two-bit indices (no
// rotation). Alpha is exact at its extremes, so a hard edge costs nothing in alpha however the colours vary.
Candidate encode_mode5(float4 texels[16])
{
    // Colour: the principal axis of the RGB values, extremes along it, two rounds of least squares.
    float3 mean = float3(0.0);

    for (uint i = 0; i < 16; ++i) {
        mean += texels[i].xyz;
    }

    mean *= 1.0 / 16.0;

    float3x3 covariance = float3x3(0.0);

    for (uint i = 0; i < 16; ++i) {
        const float3 d = texels[i].xyz - mean;

        covariance += float3x3(d.x * d, d.y * d, d.z * d);
    }

    float3 axis = float3(1.0, 1.0, 1.0);

    for (uint i = 0; i < 4; ++i) {
        axis = mul(covariance, axis);

        const float length_axis = length(axis);

        axis = length_axis > 1e-6 ? axis / length_axis : float3(0.577, 0.577, 0.577);
    }

    float t_min = 1e30;
    float t_max = -1e30;
    float3 c0 = mean;
    float3 c1 = mean;

    for (uint i = 0; i < 16; ++i) {
        const float t = dot(texels[i].xyz - mean, axis);

        if (t < t_min) {
            t_min = t;
            c0 = texels[i].xyz;
        }

        if (t > t_max) {
            t_max = t;
            c1 = texels[i].xyz;
        }
    }

    for (uint round_index = 0; round_index < 2; ++round_index) {
        const float3 line = c1 - c0;
        const float length_squared = dot(line, line);
        float a = 0.0;
        float b = 0.0;
        float c = 0.0;
        float3 alpha_sum = float3(0.0);
        float3 beta_sum = float3(0.0);

        for (uint i = 0; i < 16; ++i) {
            const float u = length_squared > 1e-4 ? dot(texels[i].xyz - c0, line) / length_squared : 0.0;
            const float w = float(bc7_weights2[uint(clamp(round(u * 3.0), 0.0, 3.0))]) / 64.0;

            a += (1.0 - w) * (1.0 - w);
            b += w * (1.0 - w);
            c += w * w;
            alpha_sum += (1.0 - w) * texels[i].xyz;
            beta_sum += w * texels[i].xyz;
        }

        const float determinant = a * c - b * b;

        if (abs(determinant) > 1e-4) {
            c0 = clamp((c * alpha_sum - b * beta_sum) / determinant, 0.0, 255.0);
            c1 = clamp((a * beta_sum - b * alpha_sum) / determinant, 0.0, 255.0);
        }
    }

    uint3 q0 = uint3(quantise7(c0.x), quantise7(c0.y), quantise7(c0.z));
    uint3 q1 = uint3(quantise7(c1.x), quantise7(c1.y), quantise7(c1.z));
    float3 v0 = float3(expand7(q0.x), expand7(q0.y), expand7(q0.z));
    float3 v1 = float3(expand7(q1.x), expand7(q1.y), expand7(q1.z));

    uint colour_index[16];
    float error = 0.0;

    for (uint i = 0; i < 16; ++i) {
        uint best = 0;
        float best_error = 1e30;

        for (uint candidate = 0; candidate < 4; ++candidate) {
            const float3 delta = texels[i].xyz - interpolate_rgb(v0, v1, bc7_weights2[candidate]);
            const float candidate_error = dot(delta, delta);

            if (candidate_error < best_error) {
                best_error = candidate_error;
                best = candidate;
            }
        }

        colour_index[i] = best;
        error += best_error;
    }

    // Alpha: the extremes, then one round of least squares.
    float alpha_low = 255.0;
    float alpha_high = 0.0;

    for (uint i = 0; i < 16; ++i) {
        alpha_low = min(alpha_low, texels[i].w);
        alpha_high = max(alpha_high, texels[i].w);
    }

    float a0 = alpha_low;
    float a1 = alpha_high;

    if (alpha_high > alpha_low) {
        float a = 0.0;
        float b = 0.0;
        float c = 0.0;
        float alpha_sum = 0.0;
        float beta_sum = 0.0;

        for (uint i = 0; i < 16; ++i) {
            const float u = (texels[i].w - alpha_low) / (alpha_high - alpha_low);
            const float w = float(bc7_weights2[uint(clamp(round(u * 3.0), 0.0, 3.0))]) / 64.0;

            a += (1.0 - w) * (1.0 - w);
            b += w * (1.0 - w);
            c += w * w;
            alpha_sum += (1.0 - w) * texels[i].w;
            beta_sum += w * texels[i].w;
        }

        const float determinant = a * c - b * b;

        if (abs(determinant) > 1e-4) {
            a0 = clamp(round((c * alpha_sum - b * beta_sum) / determinant), 0.0, 255.0);
            a1 = clamp(round((a * beta_sum - b * alpha_sum) / determinant), 0.0, 255.0);
        }
    }

    uint alpha_index[16];

    for (uint i = 0; i < 16; ++i) {
        uint best = 0;
        float best_error = 1e30;

        for (uint candidate = 0; candidate < 4; ++candidate) {
            const float delta = texels[i].w - interpolate_scalar(a0, a1, bc7_weights2[candidate]);
            const float candidate_error = delta * delta;

            if (candidate_error < best_error) {
                best_error = candidate_error;
                best = candidate;
            }
        }

        alpha_index[i] = best;
        error += best_error;
    }

    // The first texel's index of each set is stored in one bit, so it must be below 2: swap that set's endpoints if not.
    if (colour_index[0] >= 2) {
        const uint3 swapped_q = q0;
        q0 = q1;
        q1 = swapped_q;

        for (uint i = 0; i < 16; ++i) {
            colour_index[i] = 3 - colour_index[i];
        }
    }

    if (alpha_index[0] >= 2) {
        const float swapped_a = a0;
        a0 = a1;
        a1 = swapped_a;

        for (uint i = 0; i < 16; ++i) {
            alpha_index[i] = 3 - alpha_index[i];
        }
    }

    uint4 block = uint4(0);

    put_bits(block, 0, 32, 6); // mode 5: five zero bits, then a one
    put_bits(block, 6, 0, 2); // rotation: none
    put_bits(block, 8, q0.x, 7);
    put_bits(block, 15, q1.x, 7);
    put_bits(block, 22, q0.y, 7);
    put_bits(block, 29, q1.y, 7);
    put_bits(block, 36, q0.z, 7);
    put_bits(block, 43, q1.z, 7);
    put_bits(block, 50, uint(a0), 8);
    put_bits(block, 58, uint(a1), 8);

    put_bits(block, 66, colour_index[0], 1);

    for (uint i = 1; i < 16; ++i) {
        put_bits(block, 67 + (i - 1) * 2, colour_index[i], 2);
    }

    put_bits(block, 97, alpha_index[0], 1);

    for (uint i = 1; i < 16; ++i) {
        put_bits(block, 98 + (i - 1) * 2, alpha_index[i], 2);
    }

    Candidate result;

    result.block = block;
    result.error = error;

    return result;
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

    for (uint i = 0; i < 16; ++i) {
        const uint x = min(block_x * 4 + (i & 3), width - 1);
        const uint y = min(block_y * 4 + (i >> 2), height - 1);

        texels[i] = unpack_texel(pc.src[first_texel + y * width + x]);
    }

    if (pc.format == format_bc5) {
        float red[16];
        float green[16];

        for (uint i = 0; i < 16; ++i) {
            red[i] = texels[i].x;
            green[i] = texels[i].y;
        }

        uint4 bc5 = uint4(0);

        put_bc4(bc5, 0, red);
        put_bc4(bc5, 64, green);

        pc.dst[block_id] = bc5;

        return;
    }

    const Candidate mode6 = encode_mode6(texels);
    const Candidate mode5 = encode_mode5(texels);

    pc.dst[block_id] = mode5.error < mode6.error ? mode5.block : mode6.block;
}
