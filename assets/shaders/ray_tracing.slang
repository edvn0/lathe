// What ray query shaders share: the scene TLAS (SceneAccelerationStructure, at binding 6 of the bindless set), the
// per instance table that maps a hit's instance id to its geometry and material, and the vertex fetches a hit needs.
module ray_tracing;

import scene_types;

public [[vk::binding(6, 0)]] RaytracingAccelerationStructure scene_tlas;

// Mirrors SceneAccelerationStructure::InstanceData.
public struct RtInstanceData
{
    public Ptr<CompressedVertex> vertices;
    public Ptr<uint> indices;
    public uint material_index;
    public uint index_count;
    public uint flags;
    public uint reserved;
};

public float3 vertex_position(Ptr<CompressedVertex> vertices, uint index)
{
    CompressedVertex v = vertices[index];
    return float3(f16tof32(v.position_x), f16tof32(v.position_y), f16tof32(v.position_z));
}

public float2 vertex_uv(Ptr<CompressedVertex> vertices, uint index)
{
    CompressedVertex v = vertices[index];
    return float2(f16tof32(v.texcoord_u), f16tof32(v.texcoord_v));
}
