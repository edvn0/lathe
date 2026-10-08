import scene_types;
import light_clusters;

static const uint group_size = 256;

struct LightCullPushConstants
{
    Ptr<UBO> ubo;
    Ptr<GpuLight> lights;

    Ptr<float4> frustum_planes;

    Ptr<float4> visible_spheres;
    Ptr<uint> visible_light_indices;
    Ptr<uint> visible_light_count;

    uint light_count;
    uint _padding;
};

[vk::push_constant] LightCullPushConstants pc;

groupshared uint s_wave_totals[group_size];
groupshared uint s_wave_offsets[group_size];
groupshared uint s_block_total;

bool sphere_in_frustum(float4 sphere)
{
    [unroll]
    for (uint plane_index = 0; plane_index < 6; ++plane_index) {
        const float4 plane = pc.frustum_planes[plane_index];

        if (dot(plane.xyz, sphere.xyz) + plane.w < -sphere.w) {
            return false;
        }
    }

    return true;
}

[shader("compute")]
[numthreads(group_size, 1, 1)]
void main_cs(uint lane: SV_GroupIndex)
{
    const float4x4 view = pc.ubo.view;

    const uint lane_in_wave = WaveGetLaneIndex();
    const uint wave_id = lane / WaveGetLaneCount();
    const uint wave_count = (group_size + WaveGetLaneCount() - 1) / WaveGetLaneCount();

    uint written = 0;

    for (uint base = 0; base < pc.light_count; base += group_size) {
        const uint light_index = base + lane;

        bool visible = false;
        float4 sphere = float4(0.0);

        if (light_index < pc.light_count) {
            const GpuLight light = pc.lights[light_index];
            sphere = light_bounding_sphere(light);
            visible = light_screen_fade(pc.ubo, light) > 0.0 && sphere_in_frustum(sphere);
        }

        const uint wave_exclusive = WavePrefixCountBits(visible);
        const uint wave_total = WaveActiveCountBits(visible);

        if (lane_in_wave == 0) {
            s_wave_totals[wave_id] = wave_total;
        }

        GroupMemoryBarrierWithGroupSync();

        if (lane == 0) {
            uint running = 0;
            for (uint wave = 0; wave < wave_count; ++wave) {
                s_wave_offsets[wave] = running;
                running += s_wave_totals[wave];
            }
            s_block_total = running;
        }

        GroupMemoryBarrierWithGroupSync();

        if (visible) {
            const uint slot = written + s_wave_offsets[wave_id] + wave_exclusive;
            pc.visible_spheres[slot] = float4(mul(view, float4(sphere.xyz, 1.0)).xyz, sphere.w);
            pc.visible_light_indices[slot] = light_index;
        }

        written += s_block_total;

        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0) {
        pc.visible_light_count[0] = written;
    }
}
