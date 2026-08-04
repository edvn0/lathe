module bindless;


public static const uint sampled_2d_binding = 0;
public static const uint sampler_binding = 1;
public static const uint comparison_sampler_binding = 2;
public static const uint sampled_2d_depth_binding = 3;

public [[vk::binding(sampled_2d_binding, 0)]]
Texture2D<float4> sampled_2d[];

public [[vk::binding(sampler_binding, 0)]]
SamplerState samplers[];

public SamplerState get_sampler(uint index) {
    var sampler = samplers[index];
    return sampler;
}

public [[vk::binding(comparison_sampler_binding, 0)]]
SamplerComparisonState comparison_samplers[];

// A scalar-element-type view of the same sampled_2d images, needed for
// SampleCmpLevelZero (Slang's compare-sample overloads require a
// scalar-element texture). Aliasing this onto binding 0 crashes the Slang
// SPIR-V backend (assert failure in slang-ir.cpp, structType), so this is
// a distinct binding written by the same descriptor sync path as
// sampled_2d -- see gpu_resource_table.cxx.
public [[vk::binding(sampled_2d_depth_binding, 0)]]
Texture2D<float> sampled_2d_depth[];