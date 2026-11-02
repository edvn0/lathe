// Probe GI: one thread per probe reads that probe's rays (gi_probe_trace) and updates its state (see gi_probes.slang).
// A slot given to another probe by a scroll of the volume is reset. A probe seeing mostly back faces is inside
// geometry: it is switched off for now and moved through the nearest back face it saw, to be tried again next frame. A
// probe very close to a front face is nudged away from it. The offset never leaves 45% of a cell.
import gi_probes;

struct Push
{
    Ptr<float4> probe_state;
    Ptr<uint2> ray_data;
    int base_x;
    int base_y;
    int base_z;
    uint count_x;
    uint count_y;
    uint count_z;
    float spacing;
    uint rays_per_probe;
    float back_face_limit;
    float4 rotation;
    uint frame;
    uint sleep_params;
};

[vk::push_constant] Push pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID)
{
    uint slot = id.x;
    int3 counts = int3(pc.count_x, pc.count_y, pc.count_z);
    if (slot >= uint(counts.x * counts.y * counts.z))
    {
        return;
    }

    int3 world_index = gi_world_index(slot, int3(pc.base_x, pc.base_y, pc.base_z), counts);
    if (gi_probe_skipped(pc.probe_state, slot, world_index, pc.frame, pc.sleep_params))
    {
        // Asleep: no new rays, nothing to classify or blend this frame.
        pc.probe_state[slot * gi_state_stride + 4].y = 0.0;
        return;
    }

    float3 sum = float3(0.0);
    uint back = 0;
    float nearest_back = 1.0e30;
    float3 nearest_back_direction = float3(0.0);
    float nearest_front = 1.0e30;
    float3 nearest_front_direction = float3(0.0);
    for (uint i = 0; i < pc.rays_per_probe; ++i)
    {
        uint2 packed = pc.ray_data[slot * pc.rays_per_probe + i];
        float distance = f16tof32(packed.y >> 16);
        float3 radiance = float3(f16tof32(packed.x & 0xFFFF), f16tof32(packed.x >> 16), f16tof32(packed.y & 0xFFFF));
        sum += min(radiance, float3(64.0));
        float3 direction = gi_rotate(pc.rotation, gi_fibonacci_direction(i, pc.rays_per_probe));
        if (distance < 0.0)
        {
            back += 1;
            if (-distance < nearest_back)
            {
                nearest_back = -distance;
                nearest_back_direction = direction;
            }
        }
        else if (distance < nearest_front)
        {
            nearest_front = distance;
            nearest_front_direction = direction;
        }
    }

    float fraction = float(back) / float(pc.rays_per_probe);
    float4 offset_state = pc.probe_state[slot * gi_state_stride];
    float4 index_state = pc.probe_state[slot * gi_state_stride + 2];
    float3 offset = offset_state.xyz;

    float age = min(index_state.w + 1.0, 1000.0);
    if (any(int3(index_state.xyz) != world_index))
    {
        offset = float3(0.0);
        age = 0.0;
    }

    float limit = 0.45 * pc.spacing;
    bool inside = fraction > pc.back_face_limit;
    if (inside && nearest_back < 1.0e29)
    {
        offset += nearest_back_direction * (nearest_back + 0.1 * pc.spacing);
    }
    else if (nearest_front < 0.1 * pc.spacing)
    {
        offset -= nearest_front_direction * (0.1 * pc.spacing - nearest_front);
    }
    float length_sq = dot(offset, offset);
    if (length_sq > limit * limit)
    {
        offset *= limit * rsqrt(length_sq);
    }

    float3 mean = sum / float(pc.rays_per_probe);

    // Lighting changed a lot (a light or the sun moved, a door opened) when this frame's mean strays from the slow average:
    // the blend then takes the new rays in faster instead of fading over seconds.
    float4 history = pc.probe_state[slot * gi_state_stride + 3];
    if (age < 0.5)
    {
        history = float4(mean, 0.0);
    }
    else
    {
        float now = dot(mean, float3(0.2126, 0.7152, 0.0722));
        float before = dot(history.rgb, float3(0.2126, 0.7152, 0.0722));
        bool changed = abs(now - before) / (max(now, before) + 0.02) > 0.35;
        history = float4(lerp(history.rgb, mean, changed ? 0.5 : 0.05), changed ? 1.0 : 0.0);
    }

    // Quiet traced frames count toward sleep; a change, a new probe or an offset that is still moving wakes it.
    float quiet = pc.probe_state[slot * gi_state_stride + 4].x;
    bool moving = any(abs(offset - offset_state.xyz) > 1.0e-4);
    quiet = (age < 8.0 || history.w > 0.5 || moving) ? 0.0 : min(quiet + 1.0, 65535.0);
    pc.probe_state[slot * gi_state_stride + 4] = float4(quiet, 1.0, 0.0, 0.0);

    pc.probe_state[slot * gi_state_stride] = float4(offset, inside ? 0.0 : 1.0);
    pc.probe_state[slot * gi_state_stride + 1] = float4(mean, fraction);
    pc.probe_state[slot * gi_state_stride + 3] = history;
    pc.probe_state[slot * gi_state_stride + 2] = float4(float3(world_index), age);
}
