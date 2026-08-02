module bindless;


public static const uint sampled_2d_binding = 0;
public static const uint sampler_binding = 1;
public static const uint comparison_sampler_binding = 2;

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