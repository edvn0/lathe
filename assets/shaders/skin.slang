import scene_types;

// Pre-skin pass: deforms rest-pose vertices by a joint palette into a scratch CompressedVertex array that the draw
// shaders then read like any other geometry (GpuDraw::vertex_address), so none of them know about skinning. The CPU
// reference is skin_vertex() in include/gpu/skinning.hxx.
//
// Dispatch: one thread per vertex, flattened over a job table. Job j owns ceil(vertex_count / skin_chunk_size)
// consecutive workgroups (a chunk each); job_first_chunk[j] is the exclusive prefix sum of those counts
// (job_count + 1 entries, the last being the total). A group finds its job by binary search, so any number of jobs
// shares one dispatch. Dispatch (min(total, 65535), ceil(total / 65535), 1); groups past the total do nothing.

static const uint skin_chunk_size = 64;
static const uint skin_dispatch_width = 65535;

// Mirrors GpuSkinVertex in skinning.hxx: bytes of `joints` are joint indices, bytes of `weights` are unorm8.
struct GpuSkinVertex
{
    uint joints;
    uint weights;
};

// Mirrors GpuSkinJob in skinning.hxx. palette[palette_offset + joint] = joint_world * inverse_bind.
struct GpuSkinJob
{
    Ptr<CompressedVertex> rest_vertex_addr;
    Ptr<GpuSkinVertex> skin_addr;
    Ptr<CompressedVertex> out_addr;
    uint vertex_count;
    uint palette_offset;
};

struct SkinPushConstants
{
    Ptr<GpuSkinJob> jobs;
    Ptr<uint> job_first_chunk;
    Ptr<float4x4> palette;
    uint job_count;
    uint chunk_count;
};

[vk::push_constant] SkinPushConstants pc;

float2 encode_octahedral(float3 direction)
{
    float l1_norm = abs(direction.x) + abs(direction.y) + abs(direction.z);
    float2 encoded = direction.xy / l1_norm;

    if (direction.z < 0.0) {
        float2 wrapped = 1.0 - abs(encoded.yx);
        encoded = float2(encoded.x >= 0.0 ? wrapped.x : -wrapped.x, encoded.y >= 0.0 ? wrapped.y : -wrapped.y);
    }

    return encoded;
}

// Matches glm::packSnorm2x16.
uint pack_snorm2x16(float2 value)
{
    int2 quantised = int2(round(clamp(value, -1.0, 1.0) * 32767.0));
    return (uint(quantised.x) & 0xFFFFu) | (uint(quantised.y) << 16);
}

float3 matrix_column(float3x3 m, uint column)
{
    return float3(m[0][column], m[1][column], m[2][column]);
}

// The last job whose first chunk is at or before `chunk`; empty jobs share their successor's first chunk and lose.
uint chunk_job(uint chunk)
{
    uint low = 0;
    uint high = pc.job_count - 1;

    while (low < high) {
        uint middle = (low + high + 1) / 2;
        if (pc.job_first_chunk[middle] <= chunk) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }

    return low;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main_cs(uint3 group_id: SV_GroupID, uint3 group_thread_id: SV_GroupThreadID)
{
    uint chunk = group_id.y * skin_dispatch_width + group_id.x;
    if (chunk >= pc.chunk_count || pc.job_count == 0) {
        return;
    }

    uint job_index = chunk_job(chunk);
    GpuSkinJob job = pc.jobs[job_index];

    uint index = (chunk - pc.job_first_chunk[job_index]) * skin_chunk_size + group_thread_id.x;
    if (index >= job.vertex_count) {
        return;
    }

    CompressedVertex rest = job.rest_vertex_addr[index];
    GpuSkinVertex skin = job.skin_addr[index];
    DecodedVertex vertex = decode_vertex(rest);

    float weights[4];
    float total = 0.0;
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        weights[i] = float((skin.weights >> (8 * i)) & 0xFFu) / 255.0;
        total += weights[i];
    }

    // No influence at all: stay at rest rather than collapsing to the origin.
    float4x4 blended = float4x4(1.0);
    if (total > 0.0) {
        blended = float4x4(0.0);
        [unroll]
        for (uint i = 0; i < 4; ++i) {
            blended += (weights[i] / total) * pc.palette[job.palette_offset + ((skin.joints >> (8 * i)) & 0xFFu)];
        }
    }

    float3x3 linear_part = float3x3(blended);
    float3 c0 = matrix_column(linear_part, 0);
    float3 c1 = matrix_column(linear_part, 1);
    float3 c2 = matrix_column(linear_part, 2);

    // The cofactor matrix is det * inverse-transpose; only its direction (and det's sign) matters for a normal.
    float3 cof0 = cross(c1, c2);
    float3 cof1 = cross(c2, c0);
    float3 cof2 = cross(c0, c1);
    float determinant = dot(c0, cof0);

    float3 normal = vertex.normal;
    float3 skinned_normal = abs(determinant) > 1e-12
            ? (cof0 * normal.x + cof1 * normal.y + cof2 * normal.z) * (determinant < 0.0 ? -1.0 : 1.0)
            : mul(linear_part, normal);
    float3 skinned_tangent = mul(linear_part, vertex.tangent.xyz);
    float3 position = mul(blended, float4(vertex.position, 1.0)).xyz;

    CompressedVertex skinned = rest;
    skinned.position_x = uint16_t(f32tof16(position.x));
    skinned.position_y = uint16_t(f32tof16(position.y));
    skinned.position_z = uint16_t(f32tof16(position.z));
    skinned.normal_oct = pack_snorm2x16(encode_octahedral(normalize(skinned_normal)));
    skinned.tangent_oct = (pack_snorm2x16(encode_octahedral(normalize(skinned_tangent))) & 0xFFFFFFFEu) |
                         (vertex.tangent.w < 0.0 ? 1u : 0u);

    job.out_addr[index] = skinned;
}
