// Probe GI, phase 2: shows what the probe volume holds. Every pixel looks up the nearest probe of its world position
// (reconstructed from the depth buffer) and shows that probe's state.
//   1 radiance   the mean radiance of the probe's rays
//   2 state      green active, red switched off (inside geometry), dark outside the volume
//   3 back faces the fraction of rays that hit a back face, as a ramp
import bindless;

public [[vk::binding(4, 0)]] RWTexture2D<float4> storage_2d[];

struct Push
{
    float4x4 inverse_view_projection;
    Ptr<float4> probe_state;
    uint hdr;
    uint depth;
    uint result;
    int base_x;
    int base_y;
    int base_z;
    uint count_x;
    uint count_y;
    uint count_z;
    float spacing;
    uint mode;
};

[vk::push_constant] Push pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main_cs(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    sampled_2d[pc.hdr].GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
    {
        return;
    }

    float4 scene = sampled_2d[pc.hdr].Load(int3(id.xy, 0));
    float depth = sampled_2d_depth[pc.depth].Load(int3(id.xy, 0));
    float3 colour = scene.rgb;

    if (depth > 0.0)
    {
        float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
        float4 world = mul(pc.inverse_view_projection, float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0 - depth, 1.0));
        float3 position = world.xyz / world.w;

        int3 base = int3(pc.base_x, pc.base_y, pc.base_z);
        int3 counts = int3(pc.count_x, pc.count_y, pc.count_z);
        int3 index = int3(floor(position / pc.spacing + 0.5));
        int3 relative = index - base;

        if (all(relative >= 0) && all(relative < counts))
        {
            int3 slot3 = ((index % counts) + counts) % counts;
            uint slot = uint(slot3.x + counts.x * (slot3.y + counts.y * slot3.z));
            float4 a = pc.probe_state[slot * 2];
            float4 b = pc.probe_state[slot * 2 + 1];
            if (pc.mode == 1)
            {
                colour = b.rgb;
            }
            else if (pc.mode == 2)
            {
                colour = a.w > 0.5 ? float3(0.1, 0.8, 0.2) : float3(0.9, 0.1, 0.1);
            }
            else
            {
                colour = lerp(float3(0.0, 0.0, 0.3), float3(1.0, 0.9, 0.1), b.w);
            }
        }
        else if (pc.mode != 1)
        {
            colour *= 0.3;
        }
    }

    storage_2d[pc.result][id.xy] = float4(colour, scene.a);
}
