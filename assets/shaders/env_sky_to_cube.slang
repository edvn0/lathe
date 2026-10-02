module env_sky_to_cube;

import bindless;
import scene_types;
import environment;

// Renders the procedural sky into one face of the radiance cube (mip 0), from the Preetham state in the frame's UBO.
// The sun disc is left out on purpose: the directional light already lights the scene with it.
struct EnvSkyToCubePushConstants
{
    Ptr<UBO> ubo;
    uint dst_storage_index;
    uint face;
    uint size;

    // Keeps the generated C++ struct's size aligned, as in gtao_denoise.slang.
    uint _padding;
};

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

[vk::push_constant] EnvSkyToCubePushConstants pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= pc.size || id.y >= pc.size) {
        return;
    }

    const float2 uv = (float2(id.xy) + 0.5) / float(pc.size);
    const float3 direction = cube_texel_direction(pc.face, uv);

    storage_2d[pc.dst_storage_index][id.xy] = float4(environment_sky(pc.ubo, direction), 1.0);
}
