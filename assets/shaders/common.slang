// What most shaders share besides the bindless sampled arrays (import bindless; for those): the storage image array and
// the small maths every pass ends up needing. Declare none of it locally. Ray query shaders also import ray_tracing.
module common;

public static const uint storage_2d_binding = 4;

// Storage images: what compute passes write their results to. (hiz_build.slang views binding 4 as r32f itself, so it
// imports bindless alone.)
public [[vk::binding(storage_2d_binding, 0)]]
RWTexture2D<float4> storage_2d[];

public static const float pi = 3.14159265358979323846;

// PCG hash (O'Neill): a well mixed 32 bit hash for noise and per pixel randomness.
public uint pcg_hash(uint value)
{
    const uint state = value * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

public float3 srgb_to_linear(float3 c)
{
    return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4));
}
