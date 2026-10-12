#include "assets/model_skin_import.hxx"

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>
#include <ranges>
#include <unordered_map>

namespace {
    auto fail(ModelLoadErrorType type, std::string message) -> std::unexpected<ModelLoadError> {
        return std::unexpected(ModelLoadError{
                .type = type,
                .cause =
                        ErrorCause{
                                ErrorContext{
                                        .message =
                                                FlyString{
                                                        std::move(message),
                                                },
                                },
                        },
        });
    }

    constexpr glm::mat4 mirror_matrix{
            1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, -1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
    };

    auto mirror_quat(glm::quat const &q) -> glm::quat { return glm::quat{q.w, -q.x, -q.y, q.z}; }

    auto decompose(glm::mat4 const &m) -> Animation::JointTransform {
        Animation::JointTransform out;
        out.translation = glm::vec3{m[3]};
        glm::vec3 const axis_x{m[0]};
        glm::vec3 const axis_y{m[1]};
        glm::vec3 const axis_z{m[2]};
        out.scale = glm::vec3{glm::length(axis_x), glm::length(axis_y), glm::length(axis_z)};
        auto const safe = [](float v) { return v > 1e-12F ? v : 1.0F; };
        glm::mat3 const rotation{axis_x / safe(out.scale.x), axis_y / safe(out.scale.y), axis_z / safe(out.scale.z)};
        out.rotation = glm::normalize(glm::quat_cast(rotation));
        return out;
    }

    auto node_matrix(fastgltf::Node const &node) -> glm::mat4 {
        if (auto const *trs = std::get_if<fastgltf::TRS>(&node.transform)) {
            return Animation::local_matrix(glm::vec3{trs->translation[0], trs->translation[1], trs->translation[2]},
                                           glm::quat{trs->rotation[3], trs->rotation[0], trs->rotation[1],
                                                     trs->rotation[2]},
                                           glm::vec3{trs->scale[0], trs->scale[1], trs->scale[2]});
        }
        return glm::make_mat4(std::get<fastgltf::math::fmat4x4>(node.transform).data());
    }

    template<typename T>
    auto read_all(fastgltf::Asset const &asset, std::size_t accessor_index) -> std::vector<T> {
        auto const &accessor = asset.accessors[accessor_index];
        std::vector<T> values(accessor.count);
        fastgltf::copyFromAccessor<T>(asset, accessor, values.data());
        return values;
    }

    template<typename V>
    void push_keys(std::vector<float> &times, std::vector<V> &values, std::span<float const> in_times,
                   std::span<V const> in_values, bool step, bool mirror, auto &&fix) {
        (void) mirror;
        for (std::size_t i = 0; i < in_times.size() && i < in_values.size(); ++i) {
            auto const t = in_times[i];
            if (!times.empty() && !(t > times.back())) {
                continue;
            }
            if (step && !times.empty()) {
                auto const gap = t - times.back();
                auto const eps = std::min(1e-4F, gap * 0.01F);
                if (t - eps > times.back()) {
                    times.push_back(t - eps);
                    values.push_back(values.back());
                }
            }
            times.push_back(t);
            values.push_back(fix(in_values[i]));
        }
    }
}

auto ImportedClip::make_clip() const -> std::unique_ptr<Animation::KeyframeClip> {
    auto clip = std::make_unique<Animation::KeyframeClip>(base, duration);
    for (auto const &track: translations) {
        clip->add_translation_track(track);
    }
    for (auto const &track: rotations) {
        clip->add_rotation_track(track);
    }
    for (auto const &track: scales) {
        clip->add_scale_track(track);
    }
    return clip;
}

auto ModelAnimationData::find_clip(std::string_view const name) const -> ImportedClip const * {
    for (auto const &clip: clips) {
        if (clip.name == name) {
            return &clip;
        }
    }
    for (auto const &clip: clips) {
        if (clip.name.size() > name.size() && clip.name.ends_with(name) &&
            clip.name[clip.name.size() - name.size() - 1] == '|') {
            return &clip;
        }
    }
    return nullptr;
}

auto quantise_skin_weights(std::array<float, 4> weights) -> std::array<std::uint16_t, 4> {
    float sum = 0.0F;
    for (auto &w: weights) {
        w = std::isfinite(w) && w > 0.0F ? w : 0.0F;
        sum += w;
    }
    std::array<std::uint16_t, 4> out{};
    if (!(sum > 1e-12F)) {
        out[0] = 65535;
        return out;
    }
    std::uint32_t total = 0;
    std::size_t largest = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        out[i] = static_cast<std::uint16_t>(std::lround(std::min(weights[i] / sum, 1.0F) * 65535.0F));
        total += out[i];
        if (weights[i] > weights[largest]) {
            largest = i;
        }
    }
    auto const fixed = static_cast<std::int64_t>(out[largest]) + 65535 - static_cast<std::int64_t>(total);
    out[largest] = static_cast<std::uint16_t>(std::clamp<std::int64_t>(fixed, 0, 65535));
    return out;
}

auto read_skin_vertices(fastgltf::Asset const &asset, fastgltf::Primitive const &primitive,
                        std::span<std::uint32_t const> const joint_remap, std::size_t const vertex_count)
        -> std::expected<std::vector<SkinVertex>, ModelLoadError> {
    auto const joints_it = primitive.findAttribute("JOINTS_0");
    auto const weights_it = primitive.findAttribute("WEIGHTS_0");
    if (joints_it == primitive.attributes.end() || weights_it == primitive.attributes.end()) {
        return std::vector<SkinVertex>{};
    }

    auto const &joint_accessor = asset.accessors[joints_it->accessorIndex];
    if (joint_accessor.componentType != fastgltf::ComponentType::UnsignedByte &&
        joint_accessor.componentType != fastgltf::ComponentType::UnsignedShort) {
        return fail(ModelLoadErrorType::invalid_accessor, "JOINTS_0 must be uint8 or uint16");
    }
    auto const &weight_accessor = asset.accessors[weights_it->accessorIndex];
    if (weight_accessor.componentType != fastgltf::ComponentType::Float && !weight_accessor.normalized) {
        return fail(ModelLoadErrorType::invalid_accessor, "WEIGHTS_0 must be float or normalized ubyte/ushort");
    }
    if (joint_accessor.count != vertex_count || weight_accessor.count != vertex_count) {
        return fail(ModelLoadErrorType::invalid_accessor, "JOINTS_0/WEIGHTS_0 count differs from POSITION");
    }

    auto const joints = read_all<glm::u16vec4>(asset, joints_it->accessorIndex);
    auto const weights = read_all<glm::vec4>(asset, weights_it->accessorIndex);

    std::vector<SkinVertex> out(vertex_count);
    for (std::size_t v = 0; v < vertex_count; ++v) {
        auto const quantised = quantise_skin_weights({weights[v].x, weights[v].y, weights[v].z, weights[v].w});
        for (std::size_t k = 0; k < 4; ++k) {
            out[v].weights[k] = quantised[k];
            if (quantised[k] == 0) {
                out[v].joints[k] = 0;
                continue;
            }
            if (joints[v][static_cast<glm::length_t>(k)] >= joint_remap.size()) {
                return fail(ModelLoadErrorType::invalid_accessor, "JOINTS_0 index outside the skin");
            }
            out[v].joints[k] = static_cast<std::uint16_t>(joint_remap[joints[v][static_cast<glm::length_t>(k)]]);
        }
    }
    return out;
}

auto import_gltf_skin(fastgltf::Asset const &asset, bool const mirror_z)
        -> std::expected<std::optional<GltfSkinImport>, ModelLoadError> {
    if (asset.skins.empty()) {
        return std::optional<GltfSkinImport>{};
    }

    auto const &skin = asset.skins.front();
    auto const joint_count = skin.joints.size();
    if (joint_count == 0 || joint_count > 65535) {
        return fail(ModelLoadErrorType::invalid_argument, "skin has no joints or more than 65535");
    }

    std::vector<std::int64_t> node_parent(asset.nodes.size(), -1);
    for (std::size_t n = 0; n < asset.nodes.size(); ++n) {
        for (auto const child: asset.nodes[n].children) {
            if (child < node_parent.size()) {
                node_parent[child] = static_cast<std::int64_t>(n);
            }
        }
    }

    std::unordered_map<std::size_t, std::size_t> node_to_joint;
    for (std::size_t k = 0; k < joint_count; ++k) {
        if (skin.joints[k] >= asset.nodes.size() || !node_to_joint.emplace(skin.joints[k], k).second) {
            return fail(ModelLoadErrorType::invalid_argument, "skin joint is invalid or duplicated");
        }
    }

    std::vector<std::int64_t> parent_k(joint_count, -1);
    std::vector<std::vector<std::size_t>> children(joint_count);
    std::vector<std::size_t> roots;
    for (std::size_t k = 0; k < joint_count; ++k) {
        auto const parent_node = node_parent[skin.joints[k]];
        auto const found = parent_node >= 0 ? node_to_joint.find(static_cast<std::size_t>(parent_node))
                                            : node_to_joint.end();
        if (found != node_to_joint.end()) {
            parent_k[k] = static_cast<std::int64_t>(found->second);
            children[found->second].push_back(k);
        } else {
            roots.push_back(k);
        }
    }

    std::vector<std::size_t> order;
    order.reserve(joint_count);
    std::vector<std::size_t> stack;
    for (auto const root: roots) {
        stack.push_back(root);
        while (!stack.empty()) {
            auto const k = stack.back();
            stack.pop_back();
            order.push_back(k);
            for (unsigned long &it: std::views::reverse(children[k])) {
                stack.push_back(it);
            }
        }
    }
    if (order.size() != joint_count) {
        return fail(ModelLoadErrorType::invalid_argument, "skin joint hierarchy has a cycle");
    }

    std::vector<std::uint32_t> remap(joint_count);
    for (std::size_t i = 0; i < joint_count; ++i) {
        remap[order[i]] = static_cast<std::uint32_t>(i);
    }

    auto const mirror_t = [&](glm::vec3 v) {
        if (mirror_z) {
            v.z = -v.z;
        }
        return v;
    };
    auto const mirror_q = [&](glm::quat q) { return mirror_z ? mirror_quat(q) : q; };

    std::vector<std::string> names(joint_count);
    std::vector<std::int32_t> parents(joint_count, Animation::no_parent);
    Animation::Pose bind{joint_count};
    // Static transform of the non-joint ancestors folded into each root joint's bind pose; the root's animation
    // tracks need the same prefix or they would replace it (e.g. a Z-up to Y-up node above the skeleton).
    std::vector<glm::mat4> ancestor_prefix(joint_count, glm::mat4{1.0F});
    for (std::size_t i = 0; i < joint_count; ++i) {
        auto const k = order[i];
        auto const &node = asset.nodes[skin.joints[k]];
        names[i] = node.name.empty() ? std::format("joint_{}", k) : std::string{node.name};
        parents[i] = parent_k[k] >= 0 ? static_cast<std::int32_t>(remap[static_cast<std::size_t>(parent_k[k])])
                                      : Animation::no_parent;

        auto local = node_matrix(node);
        if (parent_k[k] < 0) {
            for (auto ancestor = node_parent[skin.joints[k]]; ancestor >= 0;
                 ancestor = node_parent[static_cast<std::size_t>(ancestor)]) {
                local = node_matrix(asset.nodes[static_cast<std::size_t>(ancestor)]) * local;
                ancestor_prefix[i] = node_matrix(asset.nodes[static_cast<std::size_t>(ancestor)]) * ancestor_prefix[i];
            }
        }
        auto joint = decompose(local);
        joint.translation = mirror_t(joint.translation);
        joint.rotation = mirror_q(joint.rotation);
        bind.set_joint(i, joint);
    }

    std::vector<glm::mat4> inverse_bind;
    if (skin.inverseBindMatrices.has_value()) {
        auto const raw = read_all<glm::mat4>(asset, *skin.inverseBindMatrices);
        if (raw.size() < joint_count) {
            return fail(ModelLoadErrorType::invalid_accessor, "fewer inverse bind matrices than joints");
        }
        inverse_bind.resize(joint_count);
        for (std::size_t i = 0; i < joint_count; ++i) {
            inverse_bind[i] = mirror_z ? mirror_matrix * raw[order[i]] * mirror_matrix : raw[order[i]];
        }
    }

    auto data = std::make_shared<ModelAnimationData>(
            ModelAnimationData{Animation::Skeleton{std::move(names), std::move(parents), bind, std::move(inverse_bind)},
                               {}});

    for (std::size_t a = 0; a < asset.animations.size(); ++a) {
        auto const &animation = asset.animations[a];
        ImportedClip clip;
        clip.name = animation.name.empty() ? std::format("animation_{}", a) : std::string{animation.name};
        clip.base = bind;

        for (auto const &channel: animation.channels) {
            if (!channel.nodeIndex.has_value()) {
                continue;
            }
            auto const found = node_to_joint.find(*channel.nodeIndex);
            if (found == node_to_joint.end() || channel.samplerIndex >= animation.samplers.size()) {
                continue;
            }
            auto const &sampler = animation.samplers[channel.samplerIndex];
            auto const joint = remap[found->second];
            auto const times = read_all<float>(asset, sampler.inputAccessor);
            bool const step = sampler.interpolation == fastgltf::AnimationInterpolation::Step;
            bool const cubic = sampler.interpolation == fastgltf::AnimationInterpolation::CubicSpline;
            auto const strip = [&]<typename V>(std::vector<V> values) {
                if (!cubic) {
                    return values;
                }
                std::vector<V> kept;
                for (std::size_t i = 1; i < values.size(); i += 3) {
                    kept.push_back(values[i]);
                }
                return kept;
            };

            switch (channel.path) {
                case fastgltf::AnimationPath::Translation:
                case fastgltf::AnimationPath::Scale: {
                    auto values = strip(read_all<glm::vec3>(asset, sampler.outputAccessor));
                    Animation::Vec3Track track{.joint = joint};
                    bool const is_t = channel.path == fastgltf::AnimationPath::Translation;
                    auto const &prefix = ancestor_prefix[joint];
                    push_keys<glm::vec3>(track.times, track.values, times, values, step, mirror_z, [&](glm::vec3 v) {
                        return is_t ? mirror_t(glm::vec3{prefix * glm::vec4{v, 1.0F}}) : v;
                    });
                    if (!track.times.empty()) {
                        clip.duration = std::max(clip.duration, track.times.back());
                        (is_t ? clip.translations : clip.scales).push_back(std::move(track));
                    }
                    break;
                }
                case fastgltf::AnimationPath::Rotation: {
                    auto values = strip(read_all<glm::vec4>(asset, sampler.outputAccessor));
                    std::vector<glm::quat> quats;
                    quats.reserve(values.size());
                    for (auto const &v: values) {
                        quats.emplace_back(v.w, v.x, v.y, v.z);
                    }
                    Animation::QuatTrack track{.joint = joint};
                    auto const prefix_rotation = decompose(ancestor_prefix[joint]).rotation;
                    push_keys<glm::quat>(track.times, track.values, times, quats, step, mirror_z, [&](glm::quat q) {
                        auto const norm = glm::length(q);
                        return mirror_q(prefix_rotation *
                                        (norm > 1e-12F ? q / norm : glm::quat{1.0F, 0.0F, 0.0F, 0.0F}));
                    });
                    if (!track.times.empty()) {
                        clip.duration = std::max(clip.duration, track.times.back());
                        clip.rotations.push_back(std::move(track));
                    }
                    break;
                }
                default:
                    break;
            }
        }

        if (clip.translations.empty() && clip.rotations.empty() && clip.scales.empty()) {
            continue;
        }
        clip.duration = std::max(clip.duration, 1e-4F);
        data->clips.push_back(std::move(clip));
    }

    std::vector<bool> mesh_is_skinned(asset.meshes.size(), false);
    for (auto const &node: asset.nodes) {
        if (node.skinIndex.has_value() && *node.skinIndex == 0 && node.meshIndex.has_value() &&
            *node.meshIndex < mesh_is_skinned.size()) {
            mesh_is_skinned[*node.meshIndex] = true;
        }
    }

    return GltfSkinImport{
            .data = std::move(data),
            .skin_index = 0,
            .joint_remap = std::move(remap),
            .mesh_is_skinned = std::move(mesh_is_skinned),
    };
}
