module pbr;

static const float pi = 3.14159265358979323846;

// Trowbridge-Reitz / GGX normal distribution function.
float distribution_ggx(float ndoth, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float ndoth2 = ndoth * ndoth;
    float denom = (ndoth2 * (a2 - 1.0) + 1.0);
    denom = pi * denom * denom;
    return a2 / max(denom, 1e-6);
}

// Height-correlated Smith visibility term (already divided by the BRDF's
// 4 * ndotv * ndotl denominator, so callers multiply it straight into the
// specular term with no extra division).
float visibility_smith_ggx_correlated(float ndotv, float ndotl, float roughness)
{
    float a2 = roughness * roughness * roughness * roughness;
    float lambda_v = ndotl * sqrt(ndotv * ndotv * (1.0 - a2) + a2);
    float lambda_l = ndotv * sqrt(ndotl * ndotl * (1.0 - a2) + a2);
    return 0.5 / max(lambda_v + lambda_l, 1e-6);
}

float3 fresnel_schlick(float vdoth, float3 f0)
{
    float t = clamp(1.0 - vdoth, 0.0, 1.0);
    float t5 = t * t * t * t * t;
    return f0 + (float3(1.0) - f0) * t5;
}

// Direct lighting contribution from one light with incoming radiance
// `radiance` (already attenuated/cone-masked by the caller) and direction
// `light_dir` (points from the surface toward the light), for a surface with
// the given shading normal, view direction, albedo, metallic and roughness.
// Returns the outgoing radiance contribution (not yet multiplied by any
// shadow term -- callers apply that separately, since only the directional
// light in this renderer casts shadows).
public float3 shade_punctual(float3 normal, float3 view_dir, float3 light_dir, float3 radiance, float3 albedo,
                      float metallic, float roughness)
{
    float ndotl = dot(normal, light_dir);

    if (ndotl <= 0.0) {
        return float3(0.0);
    }

    float3 half_vector = normalize(view_dir + light_dir);
    float ndotv = max(dot(normal, view_dir), 1e-4);
    float ndoth = max(dot(normal, half_vector), 0.0);
    float vdoth = max(dot(view_dir, half_vector), 0.0);

    // Perceptually-linear roughness -> alpha remap, clamped away from 0 so
    // the NDF/visibility terms above never divide by zero for mirror-smooth
    // materials.
    float alpha = max(roughness * roughness, 1e-3);

    float3 f0 = lerp(float3(0.04), albedo, metallic);

    float d = distribution_ggx(ndoth, alpha);
    float vis = visibility_smith_ggx_correlated(ndotv, ndotl, alpha);
    float3 f = fresnel_schlick(vdoth, f0);

    float3 specular = d * vis * f;

    // Metals have no diffuse response; kd removes the fraction of light
    // already accounted for by the specular Fresnel reflectance.
    float3 kd = (float3(1.0) - f) * (1.0 - metallic);
    float3 diffuse = kd * albedo / pi;

    return (diffuse + specular) * radiance * ndotl;
}
