// Probe GI, phase 2: one thread per probe reads that probe's rays (gi_probe_trace) and writes its state:
//   probe_state[2 * slot + 0] = (relocation offset xyz, classification: 1 active, 0 off)
//   probe_state[2 * slot + 1] = (mean ray radiance rgb, fraction of rays that hit a back face)
// A probe whose rays mostly see back faces is inside geometry and is switched off. Relocation is phase 3.
struct Push
{
    Ptr<float4> probe_state;
    Ptr<uint2> ray_data;
    uint probe_count;
    uint rays_per_probe;
    float back_face_limit;
};

[vk::push_constant] Push pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= pc.probe_count)
    {
        return;
    }

    float3 sum = float3(0.0);
    uint back = 0;
    for (uint i = 0; i < pc.rays_per_probe; ++i)
    {
        uint2 packed = pc.ray_data[slot * pc.rays_per_probe + i];
        float distance = f16tof32(packed.y >> 16);
        float3 radiance = float3(f16tof32(packed.x & 0xFFFF), f16tof32(packed.x >> 16), f16tof32(packed.y & 0xFFFF));
        sum += min(radiance, float3(64.0));
        back += distance < 0.0 ? 1 : 0;
    }

    float fraction = float(back) / float(pc.rays_per_probe);
    float4 offset = pc.probe_state[slot * 2];
    pc.probe_state[slot * 2] = float4(offset.xyz, fraction > pc.back_face_limit ? 0.0 : 1.0);
    pc.probe_state[slot * 2 + 1] = float4(sum / float(pc.rays_per_probe), fraction);
}
