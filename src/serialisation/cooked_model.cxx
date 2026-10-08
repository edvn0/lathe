#include "serialisation/cooked_model.hxx"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <source_location>
#include <utility>

#include <meshoptimizer.h>

#include "assets/meshlet.hxx"
#include "serialisation/byte_stream.hxx"

static_assert(sizeof(CompressedModelVertex) == 20, "CompressedModelVertex changed: bump cooked_model_version");
static_assert(sizeof(GpuMeshlet) == 48, "GpuMeshlet changed: bump cooked_model_version");
static_assert(meshlet_max_vertices == 64 && meshlet_max_triangles == 124,
              "meshlet limits changed: bump cooked_model_version");
static_assert(sizeof(SkinVertex) == 16, "SkinVertex changed: bump cooked_model_version");
static_assert(lod_count <= 8, "the LOD mask is 8 bits");

namespace {

    constexpr auto no_index = std::numeric_limits<std::uint32_t>::max();

    enum class IndexEncoding : std::uint8_t {
        raw = 0,
        meshopt = 1,
    };

    auto make_error(LbfErrorType type, std::string_view message,
                    std::source_location location = std::source_location::current()) -> LbfError {
        return LbfError{
                .type = type,
                .cause = ErrorCause{ErrorContext{
                        .message = FlyString{message},
                        .location = location,
                }},
        };
    }

    auto write_vec3(ByteWriter &writer, glm::vec3 const &value) -> void {
        writer.write(value.x);
        writer.write(value.y);
        writer.write(value.z);
    }

    auto write_vec4(ByteWriter &writer, glm::vec4 const &value) -> void {
        writer.write(value.x);
        writer.write(value.y);
        writer.write(value.z);
        writer.write(value.w);
    }

    [[nodiscard]] auto read_vec3(ByteReader &reader) -> glm::vec3 {
        glm::vec3 value{};
        reader.read(value.x);
        reader.read(value.y);
        reader.read(value.z);
        return value;
    }

    [[nodiscard]] auto read_vec4(ByteReader &reader) -> glm::vec4 {
        glm::vec4 value{};
        reader.read(value.x);
        reader.read(value.y);
        reader.read(value.z);
        reader.read(value.w);
        return value;
    }

    auto write_image_index(ByteWriter &writer, std::optional<std::size_t> index) -> void {
        writer.write(index.has_value() ? static_cast<std::uint32_t>(*index) : no_index);
    }

    [[nodiscard]] auto read_image_index(ByteReader &reader, std::size_t image_count) -> std::optional<std::size_t> {
        auto const index = reader.read<std::uint32_t>();

        if (index == no_index) {
            return std::nullopt;
        }

        if (index >= image_count) {
            reader.fail();
            return std::nullopt;
        }

        return index;
    }

    [[nodiscard]]
    auto primitive_bounds(ModelCpuPrimitive const &primitive) -> std::pair<glm::vec3, glm::vec3> {
        if (primitive.bounds.has_value()) {
            return *primitive.bounds;
        }

        auto bounds_min = glm::vec3{std::numeric_limits<float>::max()};
        auto bounds_max = glm::vec3{std::numeric_limits<float>::lowest()};

        for (auto const &vertex: primitive.vertices) {
            bounds_min = glm::min(bounds_min, vertex.position);
            bounds_max = glm::max(bounds_max, vertex.position);
        }

        if (bounds_min.x > bounds_max.x) {
            return {glm::vec3{-0.5F}, glm::vec3{0.5F}};
        }

        return {bounds_min, bounds_max};
    }

    inline constexpr std::uint32_t max_primitive_vertices = 1U << 24U;
    inline constexpr std::uint32_t max_primitive_indices = 3U << 24U;
    inline constexpr std::uint32_t max_node_depth = 256;
    inline constexpr std::size_t max_debug_name_length = 256;

    template<glm::length_t N>
    [[nodiscard]] auto all_finite(glm::vec<N, float> const &value) noexcept -> bool {
        for (glm::length_t i = 0; i < N; ++i) {
            if (!std::isfinite(value[i])) {
                return false;
            }
        }

        return true;
    }

    inline constexpr std::uint32_t max_skin_joints = 4096;
    inline constexpr std::uint32_t max_skin_clips = 4096;
    inline constexpr std::uint32_t max_track_keys = 1U << 22U;

    auto write_pose(ByteWriter &writer, Animation::Pose const &pose) -> void {
        for (std::size_t i = 0; i < pose.size(); ++i) {
            auto const joint = pose.joint(i);
            write_vec3(writer, joint.translation);
            write_vec4(writer, glm::vec4{joint.rotation.x, joint.rotation.y, joint.rotation.z, joint.rotation.w});
            write_vec3(writer, joint.scale);
        }
    }

    auto read_pose(ByteReader &reader, std::size_t count) -> Animation::Pose {
        Animation::Pose pose{count};
        for (std::size_t i = 0; i < count && reader.ok(); ++i) {
            auto const t = read_vec3(reader);
            auto const q = read_vec4(reader);
            auto const s = read_vec3(reader);
            pose.set_joint(i, {t, glm::quat{q.w, q.x, q.y, q.z}, s});
        }
        return pose;
    }

    auto write_vec3_tracks(ByteWriter &writer, std::vector<Animation::Vec3Track> const &tracks) -> void {
        writer.write(static_cast<std::uint32_t>(tracks.size()));
        for (auto const &track: tracks) {
            writer.write(track.joint);
            writer.write_array(track.times);
            for (auto const &value: track.values) {
                write_vec3(writer, value);
            }
        }
    }

    auto write_quat_tracks(ByteWriter &writer, std::vector<Animation::QuatTrack> const &tracks) -> void {
        writer.write(static_cast<std::uint32_t>(tracks.size()));
        for (auto const &track: tracks) {
            writer.write(track.joint);
            writer.write_array(track.times);
            for (auto const &value: track.values) {
                write_vec4(writer, glm::vec4{value.x, value.y, value.z, value.w});
            }
        }
    }

    auto read_track_header(ByteReader &reader, std::size_t joint_count, std::uint32_t &joint,
                           std::vector<float> &times) -> bool {
        joint = reader.read<std::uint32_t>();
        reader.read_array(times);
        if (reader.failed() || joint >= joint_count || times.size() > max_track_keys || times.empty()) {
            reader.fail();
            return false;
        }
        for (std::size_t i = 0; i < times.size(); ++i) {
            if (!std::isfinite(times[i]) || (i > 0 && !(times[i] > times[i - 1]))) {
                reader.fail();
                return false;
            }
        }
        if (reader.remaining() < times.size() * 12) {
            reader.fail();
            return false;
        }
        return true;
    }

    auto read_vec3_tracks(ByteReader &reader, std::size_t joint_count, std::vector<Animation::Vec3Track> &tracks)
            -> void {
        auto const count = reader.read<std::uint32_t>();
        if (reader.failed() || count > joint_count) {
            reader.fail();
            return;
        }
        tracks.resize(count);
        for (auto &track: tracks) {
            if (!read_track_header(reader, joint_count, track.joint, track.times)) {
                return;
            }
            track.values.resize(track.times.size());
            for (auto &value: track.values) {
                value = read_vec3(reader);
                if (!all_finite(value)) {
                    reader.fail();
                }
            }
        }
    }

    auto read_quat_tracks(ByteReader &reader, std::size_t joint_count, std::vector<Animation::QuatTrack> &tracks)
            -> void {
        auto const count = reader.read<std::uint32_t>();
        if (reader.failed() || count > joint_count) {
            reader.fail();
            return;
        }
        tracks.resize(count);
        for (auto &track: tracks) {
            if (!read_track_header(reader, joint_count, track.joint, track.times)) {
                return;
            }
            track.values.resize(track.times.size());
            for (auto &value: track.values) {
                auto const v = read_vec4(reader);
                if (!all_finite(v)) {
                    reader.fail();
                }
                value = glm::quat{v.w, v.x, v.y, v.z};
            }
        }
    }

    auto write_skin_section(ByteWriter &writer, ModelCpuData const &cpu_data) -> std::optional<LbfError> {
        for (auto const &mesh: cpu_data.meshes) {
            for (auto const &primitive: mesh.primitives) {
                auto const &skin = primitive.skin;

                if (skin.empty()) {
                    writer.write(static_cast<std::uint8_t>(0));
                    continue;
                }

                if (skin.size() != primitive.compressed_vertices.size()) {
                    return make_error(LbfErrorType::cook_failed, "skin stream doesn't match the vertex count");
                }

                std::vector<std::byte> encoded(meshopt_encodeVertexBufferBound(skin.size(), sizeof(SkinVertex)));
                auto const encoded_size = meshopt_encodeVertexBuffer(
                        reinterpret_cast<unsigned char *>(encoded.data()), encoded.size(), skin.data(), skin.size(),
                        sizeof(SkinVertex));

                if (encoded_size == 0) {
                    return make_error(LbfErrorType::cook_failed, "meshopt_encodeVertexBuffer failed for skin");
                }

                writer.write(static_cast<std::uint8_t>(1));
                writer.write(static_cast<std::uint32_t>(encoded_size));
                writer.write_span(std::span<std::byte const>{encoded}.first(encoded_size));
            }
        }

        auto const *animation = cpu_data.animation.get();
        writer.write(static_cast<std::uint8_t>(animation != nullptr ? 1 : 0));

        if (animation == nullptr) {
            return std::nullopt;
        }

        auto const &skeleton = animation->skeleton;
        writer.write(static_cast<std::uint32_t>(skeleton.joint_count()));
        for (std::size_t i = 0; i < skeleton.joint_count(); ++i) {
            writer.write_string(skeleton.names()[i]);
            writer.write(skeleton.parents()[i]);
        }
        write_pose(writer, skeleton.bind_pose());
        for (auto const &matrix: skeleton.inverse_bind()) {
            writer.write_span(std::span<float const>{&matrix[0][0], 16});
        }

        writer.write(static_cast<std::uint32_t>(animation->clips.size()));
        for (auto const &clip: animation->clips) {
            writer.write_string(clip.name);
            writer.write(clip.duration);
            write_pose(writer, clip.base);
            write_vec3_tracks(writer, clip.translations);
            write_quat_tracks(writer, clip.rotations);
            write_vec3_tracks(writer, clip.scales);
        }

        return std::nullopt;
    }

    auto read_skin_section(ByteReader &reader, ModelCpuData &cpu_data) -> void {
        for (auto &mesh: cpu_data.meshes) {
            for (auto &primitive: mesh.primitives) {
                auto const present = reader.read<std::uint8_t>();

                if (reader.failed() || present > 1) {
                    reader.fail();
                    return;
                }

                if (present == 0) {
                    continue;
                }

                auto const encoded_size = reader.read<std::uint32_t>();
                auto const encoded = reader.read_bytes(encoded_size);
                auto const vertex_count = primitive.compressed_vertices.size();

                if (reader.failed() || vertex_count / 16 > encoded_size) {
                    reader.fail();
                    return;
                }

                primitive.skin.resize(vertex_count);

                if (meshopt_decodeVertexBuffer(primitive.skin.data(), vertex_count, sizeof(SkinVertex),
                                               reinterpret_cast<unsigned char const *>(encoded.data()),
                                               encoded.size()) != 0) {
                    reader.fail();
                    return;
                }
            }
        }

        auto const has_animation = reader.read<std::uint8_t>();

        if (reader.failed() || has_animation > 1) {
            reader.fail();
            return;
        }

        std::uint32_t joint_count = 0;

        if (has_animation == 1) {
            joint_count = reader.read<std::uint32_t>();

            if (reader.failed() || joint_count == 0 || joint_count > max_skin_joints) {
                reader.fail();
                return;
            }

            std::vector<std::string> names(joint_count);
            std::vector<std::int32_t> parents(joint_count);

            for (std::uint32_t i = 0; i < joint_count; ++i) {
                reader.read_string(names[i]);
                parents[i] = reader.read<std::int32_t>();

                if (reader.failed() || names[i].size() > max_debug_name_length ||
                    parents[i] >= static_cast<std::int32_t>(i) || parents[i] < Animation::no_parent) {
                    reader.fail();
                    return;
                }
            }

            auto bind = read_pose(reader, joint_count);
            std::vector<glm::mat4> inverse_bind(joint_count);

            for (auto &matrix: inverse_bind) {
                reader.read_span(std::span<float>{&matrix[0][0], 16});
            }

            if (reader.failed()) {
                return;
            }

            auto const finite = [](glm::mat4 const &m) {
                return all_finite(m[0]) && all_finite(m[1]) && all_finite(m[2]) && all_finite(m[3]);
            };

            for (std::uint32_t i = 0; i < joint_count; ++i) {
                auto const joint = bind.joint(i);

                if (!all_finite(joint.translation) || !all_finite(joint.scale) ||
                    !std::isfinite(joint.rotation.x + joint.rotation.y + joint.rotation.z + joint.rotation.w) ||
                    !finite(inverse_bind[i])) {
                    reader.fail();
                    return;
                }
            }

            auto animation = std::make_shared<ModelAnimationData>(ModelAnimationData{
                    Animation::Skeleton{std::move(names), std::move(parents), std::move(bind), std::move(inverse_bind)},
                    {},
            });

            auto const clip_count = reader.read<std::uint32_t>();

            if (reader.failed() || clip_count > max_skin_clips) {
                reader.fail();
                return;
            }

            animation->clips.resize(clip_count);

            for (auto &clip: animation->clips) {
                reader.read_string(clip.name);
                reader.read(clip.duration);
                clip.base = read_pose(reader, joint_count);
                read_vec3_tracks(reader, joint_count, clip.translations);
                read_quat_tracks(reader, joint_count, clip.rotations);
                read_vec3_tracks(reader, joint_count, clip.scales);

                if (reader.failed() || clip.name.size() > max_debug_name_length || !std::isfinite(clip.duration)) {
                    reader.fail();
                    return;
                }
            }

            cpu_data.animation = std::move(animation);
        }

        for (auto const &mesh: cpu_data.meshes) {
            for (auto const &primitive: mesh.primitives) {
                for (auto const &vertex: primitive.skin) {
                    std::uint32_t weight_sum = 0;

                    for (std::size_t k = 0; k < 4; ++k) {
                        weight_sum += vertex.weights[k];

                        if (vertex.weights[k] != 0 && vertex.joints[k] >= joint_count) {
                            reader.fail();
                            return;
                        }
                    }

                    if (weight_sum != 65535) {
                        reader.fail();
                        return;
                    }
                }
            }
        }
    }

    auto accumulate_bounds(ModelCpuData const &cpu_data, std::uint32_t node_index, glm::mat4 const &parent,
                           glm::vec3 &bounds_min, glm::vec3 &bounds_max, std::uint32_t depth) -> void {
        if (node_index >= cpu_data.nodes.size() || depth > 256) {
            return;
        }

        auto const &node = cpu_data.nodes[node_index];
        auto const local_to_model = parent * node.local_transform;

        if (node.mesh_index < cpu_data.meshes.size()) {
            for (auto const &primitive: cpu_data.meshes[node.mesh_index].primitives) {
                auto const accumulate = [&](glm::vec3 const &local) {
                    auto const position = glm::vec3{local_to_model * glm::vec4{local, 1.0F}};
                    bounds_min = glm::min(bounds_min, position);
                    bounds_max = glm::max(bounds_max, position);
                };

                if (!primitive.vertices.empty()) {
                    for (auto const &vertex: primitive.vertices) {
                        accumulate(vertex.position);
                    }
                    continue;
                }

                auto const [local_min, local_max] = primitive_bounds(primitive);

                for (std::uint32_t corner = 0; corner < 8; ++corner) {
                    accumulate(glm::vec3{(corner & 1U) != 0 ? local_max.x : local_min.x,
                                         (corner & 2U) != 0 ? local_max.y : local_min.y,
                                         (corner & 4U) != 0 ? local_max.z : local_min.z});
                }
            }
        }

        for (auto const child: node.children) {
            accumulate_bounds(cpu_data, child, local_to_model, bounds_min, bounds_max, depth + 1);
        }
    }

    [[nodiscard]] auto model_bounds(ModelCpuData const &cpu_data) -> std::pair<glm::vec3, glm::vec3> {
        if (cpu_data.bounds.has_value()) {
            return *cpu_data.bounds;
        }

        auto bounds_min = glm::vec3{std::numeric_limits<float>::max()};
        auto bounds_max = glm::vec3{std::numeric_limits<float>::lowest()};

        for (auto const root: cpu_data.scene_roots) {
            accumulate_bounds(cpu_data, root, glm::mat4{1.0F}, bounds_min, bounds_max, 0);
        }

        if (bounds_min.x > bounds_max.x) {
            return {glm::vec3{-0.5F}, glm::vec3{0.5F}};
        }

        return {bounds_min, bounds_max};
    }

    auto write_indices(ByteWriter &writer, std::vector<std::uint32_t> const &indices, std::size_t vertex_count)
            -> void {
        writer.write(static_cast<std::uint32_t>(indices.size()));

        if (!indices.empty() && indices.size() % 3 == 0) {
            std::vector<std::byte> encoded(meshopt_encodeIndexBufferBound(indices.size(), vertex_count));
            auto const size = meshopt_encodeIndexBuffer(reinterpret_cast<unsigned char *>(encoded.data()),
                                                        encoded.size(), indices.data(), indices.size());

            if (size != 0) {
                writer.write(std::to_underlying(IndexEncoding::meshopt));
                writer.write(static_cast<std::uint32_t>(size));
                writer.write_span(std::span<std::byte const>{encoded}.first(size));
                return;
            }
        }

        writer.write(std::to_underlying(IndexEncoding::raw));
        writer.write(static_cast<std::uint32_t>(indices.size() * sizeof(std::uint32_t)));
        writer.write_span(std::span<std::uint32_t const>{indices});
    }

    auto read_indices(ByteReader &reader, std::vector<std::uint32_t> &indices) -> bool {
        auto const count = reader.read<std::uint32_t>();
        auto const encoding = reader.read<std::uint8_t>();
        auto const size = reader.read<std::uint32_t>();
        auto const bytes = reader.read_bytes(size);

        if (reader.failed() || count % 3 != 0 || count > max_primitive_indices) {
            reader.fail();
            return false;
        }

        switch (static_cast<IndexEncoding>(encoding)) {
            case IndexEncoding::raw:
                if (size != static_cast<std::uint64_t>(count) * sizeof(std::uint32_t)) {
                    reader.fail();
                    return false;
                }

                indices.resize(count);

                if (count != 0) {
                    std::memcpy(indices.data(), bytes.data(), size);
                }
                return true;

            case IndexEncoding::meshopt:
                if (count / 3 > size) {
                    reader.fail();
                    return false;
                }

                indices.resize(count);

                if (meshopt_decodeIndexBuffer(indices.data(), count, sizeof(std::uint32_t),
                                              reinterpret_cast<unsigned char const *>(bytes.data()),
                                              bytes.size()) != 0) {
                    reader.fail();
                    return false;
                }
                return true;
        }

        reader.fail();
        return false;
    }

    auto write_meshlets(ByteWriter &writer, MeshletBuild const &build) -> void {
        writer.write_array(build.meshlets);
        writer.write_array(build.topology.data);
    }

    auto read_meshlets(ByteReader &reader, MeshletBuild &build, std::uint32_t primitive_vertex_count) -> bool {
        reader.read_array(build.meshlets);
        reader.read_array(build.topology.data);

        build.topology.meshlets.clear();
        build.topology.meshlets.reserve(build.meshlets.size());

        auto const &data = build.topology.data;
        auto const data_size = data.size();

        for (auto const &meshlet: build.meshlets) {
            if (static_cast<std::uint64_t>(meshlet.vertex_offset) + meshlet.vertex_count > data_size ||
                static_cast<std::uint64_t>(meshlet.triangle_offset) + meshlet.triangle_count > data_size) {
                reader.fail();
                return false;
            }

            if (meshlet.vertex_count == 0 || meshlet.vertex_count > meshlet_max_vertices ||
                meshlet.triangle_count == 0 || meshlet.triangle_count > meshlet_max_triangles) {
                reader.fail();
                return false;
            }

            auto const vertices =
                    std::span<std::uint32_t const>{data}.subspan(meshlet.vertex_offset, meshlet.vertex_count);

            if (!std::ranges::all_of(vertices, [&](std::uint32_t index) { return index < primitive_vertex_count; })) {
                reader.fail();
                return false;
            }

            auto const triangles =
                    std::span<std::uint32_t const>{data}.subspan(meshlet.triangle_offset, meshlet.triangle_count);

            auto const triangle_in_range = [&](std::uint32_t packed) {
                return (packed & 0xFFU) < meshlet.vertex_count && ((packed >> 8U) & 0xFFU) < meshlet.vertex_count &&
                       ((packed >> 16U) & 0xFFU) < meshlet.vertex_count;
            };

            if (!std::ranges::all_of(triangles, triangle_in_range)) {
                reader.fail();
                return false;
            }

            if (!all_finite(meshlet.centre) || !std::isfinite(meshlet.radius) || meshlet.radius < 0.0F ||
                !all_finite(meshlet.cone_axis) || !std::isfinite(meshlet.cone_cutoff)) {
                reader.fail();
                return false;
            }

            build.topology.meshlets.push_back(MeshletTopology::Range{
                    .vertex_offset = meshlet.vertex_offset,
                    .triangle_offset = meshlet.triangle_offset,
                    .vertex_count = meshlet.vertex_count,
                    .triangle_count = meshlet.triangle_count,
            });
        }

        return reader.ok();
    }

    [[nodiscard]] auto node_graph_is_a_forest(ModelCpuData const &cpu_data) -> bool {
        std::vector<bool> visited(cpu_data.nodes.size(), false);
        std::vector<std::pair<std::uint32_t, std::uint32_t>> stack;

        stack.reserve(cpu_data.scene_roots.size());

        for (auto const root: cpu_data.scene_roots) {
            stack.emplace_back(root, 0);
        }

        while (!stack.empty()) {
            auto const [node, depth] = stack.back();
            stack.pop_back();

            if (visited[node] || depth > max_node_depth) {
                return false;
            }

            visited[node] = true;

            auto const &children = cpu_data.nodes[node].children;
            stack.reserve(stack.size() + children.size());

            for (auto const child: children) {
                stack.emplace_back(child, depth + 1);
            }
        }

        return true;
    }

}

auto encode_cooked_model(ModelCpuData const &cpu_data, std::span<CookedImageRef const> images,
                         std::span<DefaultSampler const> material_samplers)
        -> std::expected<std::vector<std::byte>, LbfError> {
    ZoneScopedNC("encode_cooked_model", tracy::Color::Goldenrod);

    if (images.size() != cpu_data.image_sources.size() || material_samplers.size() != cpu_data.materials.size()) {
        return std::unexpected(make_error(LbfErrorType::cook_failed, "image/sampler tables don't match the model"));
    }

    ByteWriter writer;

    auto const [bounds_min, bounds_max] = model_bounds(cpu_data);
    write_vec3(writer, bounds_min);
    write_vec3(writer, bounds_max);

    writer.write(static_cast<std::uint32_t>(images.size()));

    for (auto const &image: images) {
        writer.write(image.texture.value);
        writer.write(std::to_underlying(image.slot));
        writer.write_string(image.debug_name.view());
    }

    writer.write(static_cast<std::uint32_t>(cpu_data.materials.size()));

    for (std::size_t index = 0; index < cpu_data.materials.size(); ++index) {
        auto const &material = cpu_data.materials[index];

        write_vec4(writer, material.base_colour_factor);
        write_vec3(writer, material.emissive_factor);
        writer.write(material.emissive_strength);
        writer.write(material.metallic_factor);
        writer.write(material.roughness_factor);
        writer.write(material.alpha_cutoff);
        writer.write(material.normal_scale);
        writer.write(material.occlusion_strength);
        writer.write(std::to_underlying(material.alpha_mode));
        writer.write(static_cast<std::uint8_t>(material.double_sided ? 1U : 0U));
        writer.write(std::to_underlying(material_samplers[index]));

        write_image_index(writer, material.base_colour_image);
        write_image_index(writer, material.metallic_roughness_image);
        write_image_index(writer, material.normal_image);
        write_image_index(writer, material.occlusion_image);
        write_image_index(writer, material.emissive_image);
    }

    writer.write(static_cast<std::uint32_t>(cpu_data.nodes.size()));

    for (auto const &node: cpu_data.nodes) {
        writer.write_span(std::span<float const>{&node.local_transform[0][0], 16});
        writer.write(node.mesh_index);
        writer.write_array(node.children);
    }

    writer.write_array(cpu_data.scene_roots);

    writer.write(static_cast<std::uint32_t>(cpu_data.lights.size()));

    for (auto const &light: cpu_data.lights) {
        writer.write(std::to_underlying(light.type));
        write_vec3(writer, light.position);
        write_vec3(writer, light.direction);
        write_vec3(writer, light.colour);
        writer.write(light.intensity);
        writer.write(light.range);
        writer.write(light.inner_cone_degrees);
        writer.write(light.outer_cone_degrees);
    }

    writer.write(static_cast<std::uint32_t>(cpu_data.meshes.size()));

    for (auto const &mesh: cpu_data.meshes) {
        writer.write(static_cast<std::uint32_t>(mesh.primitives.size()));

        for (auto const &primitive: mesh.primitives) {
            if (primitive.compressed_vertices.empty() || !primitive.meshlets[0].has_value()) {
                return std::unexpected(make_error(LbfErrorType::cook_failed, "primitive was not finalized"));
            }

            writer.write(primitive.material_index.value_or(no_index));

            auto const [primitive_min, primitive_max] = primitive_bounds(primitive);
            write_vec3(writer, primitive_min);
            write_vec3(writer, primitive_max);

            auto const &vertices = primitive.compressed_vertices;
            std::vector<std::byte> encoded(
                    meshopt_encodeVertexBufferBound(vertices.size(), sizeof(CompressedModelVertex)));
            auto const encoded_size =
                    meshopt_encodeVertexBuffer(reinterpret_cast<unsigned char *>(encoded.data()), encoded.size(),
                                               vertices.data(), vertices.size(), sizeof(CompressedModelVertex));

            if (encoded_size == 0) {
                return std::unexpected(make_error(LbfErrorType::cook_failed, "meshopt_encodeVertexBuffer failed"));
            }

            writer.write(static_cast<std::uint32_t>(vertices.size()));
            writer.write(static_cast<std::uint32_t>(encoded_size));
            writer.write_span(std::span<std::byte const>{encoded}.first(encoded_size));

            std::uint8_t level_mask = 1;

            for (std::uint32_t level = 1; level < lod_count; ++level) {
                if (primitive.reduced_indices[level - 1].has_value()) {
                    if (!primitive.meshlets[level].has_value()) {
                        return std::unexpected(make_error(LbfErrorType::cook_failed, "LOD without meshlets"));
                    }

                    level_mask = static_cast<std::uint8_t>(level_mask | (1U << level));
                }
            }

            writer.write(level_mask);

            for (std::uint32_t level = 0; level < lod_count; ++level) {
                if ((level_mask & (1U << level)) == 0) {
                    continue;
                }

                auto const &indices = level == 0 ? primitive.indices : *primitive.reduced_indices[level - 1];
                write_indices(writer, indices, vertices.size());
                write_meshlets(writer, *primitive.meshlets[level]);
            }
        }
    }

    if (auto const error = write_skin_section(writer, cpu_data)) {
        return std::unexpected(*error);
    }

    return writer.take();
}

auto decode_cooked_model(std::span<std::byte const> payload, std::uint16_t version)
        -> std::expected<CookedModel, LbfError> {
    ZoneScopedNC("decode_cooked_model", tracy::Color::Goldenrod);

    if (version < cooked_model_oldest_readable_version || version > cooked_model_version) {
        return std::unexpected(make_error(LbfErrorType::unsupported_version,
                                          std::format("MODL v{}, this build reads v{}-v{}", version,
                                                      cooked_model_oldest_readable_version, cooked_model_version)));
    }

    ByteReader reader{payload};
    CookedModel cooked;
    auto &cpu_data = cooked.cpu_data;

    auto const bounds_min = read_vec3(reader);
    auto const bounds_max = read_vec3(reader);
    cpu_data.bounds = std::pair{bounds_min, bounds_max};

    auto const bounded_count = [&](std::size_t minimum_record_size = 1) {
        auto const count = reader.read<std::uint32_t>();

        if (count > reader.remaining() / minimum_record_size) {
            reader.fail();
            return std::uint32_t{0};
        }

        return count;
    };

    auto const image_count = bounded_count(13);
    cooked.images.resize(image_count);
    cpu_data.image_sources.resize(image_count);

    for (std::size_t index = 0; index < image_count; ++index) {
        auto &image = cooked.images[index];
        image.texture.value = reader.read<std::uint64_t>();

        auto const slot = reader.read<std::uint8_t>();

        if (slot > std::to_underlying(ModelTextureSlot::emissive)) {
            reader.fail();
        }

        image.slot = static_cast<ModelTextureSlot>(slot);

        auto debug_name = reader.read_string();
        debug_name.resize(std::min(debug_name.size(), max_debug_name_length));
        image.debug_name = FlyString{debug_name};

        cpu_data.image_sources[index].slot = image.slot;
        cpu_data.image_sources[index].debug_name = image.debug_name;
    }

    auto const material_count = bounded_count(64);
    cpu_data.materials.resize(material_count);
    cooked.material_samplers.resize(material_count);

    for (std::size_t index = 0; index < material_count; ++index) {
        auto &material = cpu_data.materials[index];

        material.base_colour_factor = read_vec4(reader);
        material.emissive_factor = read_vec3(reader);
        reader.read(material.emissive_strength);
        reader.read(material.metallic_factor);
        reader.read(material.roughness_factor);
        reader.read(material.alpha_cutoff);
        reader.read(material.normal_scale);
        reader.read(material.occlusion_strength);

        auto const alpha_mode = reader.read<std::uint32_t>();
        material.double_sided = version >= 2 && reader.read<std::uint8_t>() != 0;
        auto const sampler = reader.read<std::uint8_t>();

        if (alpha_mode > std::to_underlying(AlphaMode::blend) ||
            sampler > std::to_underlying(DefaultSampler::nearest_clamp)) {
            reader.fail();
        }

        material.alpha_mode = static_cast<AlphaMode>(alpha_mode);
        cooked.material_samplers[index] = static_cast<DefaultSampler>(sampler);

        material.base_colour_image = read_image_index(reader, image_count);
        material.metallic_roughness_image = read_image_index(reader, image_count);
        material.normal_image = read_image_index(reader, image_count);
        material.occlusion_image = read_image_index(reader, image_count);
        material.emissive_image = read_image_index(reader, image_count);
    }

    auto const node_count = bounded_count(72);
    cpu_data.nodes.resize(node_count);

    for (auto &node: cpu_data.nodes) {
        reader.read_span(std::span<float>{&node.local_transform[0][0], 16});
        reader.read(node.mesh_index);
        reader.read_array(node.children);
    }

    reader.read_array(cpu_data.scene_roots);

    auto const light_count = bounded_count(53);
    cpu_data.lights.resize(light_count);

    for (auto &light: cpu_data.lights) {
        auto const type = reader.read<std::uint8_t>();

        if (type > std::to_underlying(ModelLightType::spot)) {
            reader.fail();
        }

        light.type = static_cast<ModelLightType>(type);
        light.position = read_vec3(reader);
        light.direction = read_vec3(reader);
        light.colour = read_vec3(reader);
        reader.read(light.intensity);
        reader.read(light.range);
        reader.read(light.inner_cone_degrees);
        reader.read(light.outer_cone_degrees);
    }

    auto const mesh_count = bounded_count(4);
    cpu_data.meshes.resize(mesh_count);

    for (auto &mesh: cpu_data.meshes) {
        mesh.primitives.resize(bounded_count(36));

        for (auto &primitive: mesh.primitives) {
            if (reader.failed()) {
                break;
            }

            auto const material_index = reader.read<std::uint32_t>();

            if (material_index != no_index) {
                if (material_index >= material_count) {
                    reader.fail();
                }
                primitive.material_index = material_index;
            }

            auto const primitive_min = read_vec3(reader);
            auto const primitive_max = read_vec3(reader);
            primitive.bounds = std::pair{primitive_min, primitive_max};

            auto const vertex_count = reader.read<std::uint32_t>();
            auto const encoded_size = reader.read<std::uint32_t>();
            auto const encoded = reader.read_bytes(encoded_size);

            if (reader.failed() || vertex_count == 0 || vertex_count > max_primitive_vertices ||
                vertex_count / 16 > encoded_size) {
                reader.fail();
                break;
            }

            primitive.compressed_vertices.resize(vertex_count);

            if (meshopt_decodeVertexBuffer(
                        primitive.compressed_vertices.data(), vertex_count, sizeof(CompressedModelVertex),
                        reinterpret_cast<unsigned char const *>(encoded.data()), encoded.size()) != 0) {
                reader.fail();
                break;
            }

            auto const level_mask = reader.read<std::uint8_t>();

            if ((level_mask & 1U) == 0) {
                reader.fail();
                break;
            }

            for (std::uint32_t level = 0; level < lod_count; ++level) {
                if ((level_mask & (1U << level)) == 0) {
                    continue;
                }

                auto &indices = level == 0 ? primitive.indices : primitive.reduced_indices[level - 1].emplace();
                read_indices(reader, indices);

                for (auto const index: indices) {
                    if (index >= vertex_count) {
                        reader.fail();
                        break;
                    }
                }

                read_meshlets(reader, primitive.meshlets[level].emplace(), vertex_count);
            }
        }
    }

    if (version >= 3 && !reader.failed()) {
        read_skin_section(reader, cpu_data);
    }

    if (reader.failed()) {
        return std::unexpected(make_error(LbfErrorType::malformed_payload, "MODL chunk is truncated or inconsistent"));
    }

    auto const valid_node = [&](std::uint32_t index) { return index < node_count; };

    auto const finite_matrix = [](glm::mat4 const &matrix) {
        return all_finite(matrix[0]) && all_finite(matrix[1]) && all_finite(matrix[2]) && all_finite(matrix[3]);
    };

    auto const finite_bounds = [](std::pair<glm::vec3, glm::vec3> const &bounds) {
        return all_finite(bounds.first) && all_finite(bounds.second);
    };

    auto const finite_material = [](auto const &material) {
        return all_finite(material.base_colour_factor) && all_finite(material.emissive_factor) &&
               std::isfinite(material.emissive_strength) && std::isfinite(material.metallic_factor) &&
               std::isfinite(material.roughness_factor) && std::isfinite(material.alpha_cutoff) &&
               std::isfinite(material.normal_scale) && std::isfinite(material.occlusion_strength);
    };

    auto const finite_light = [](auto const &light) {
        return all_finite(light.position) && all_finite(light.direction) && all_finite(light.colour) &&
               std::isfinite(light.intensity) && std::isfinite(light.range) &&
               std::isfinite(light.inner_cone_degrees) && std::isfinite(light.outer_cone_degrees);
    };

    auto const primitives_finite = std::ranges::all_of(cpu_data.meshes, [&](auto const &mesh) {
        return std::ranges::all_of(mesh.primitives,
                                   [&](auto const &primitive) { return finite_bounds(*primitive.bounds); });
    });

    if (!finite_bounds(*cpu_data.bounds) || !primitives_finite ||
        !std::ranges::all_of(cpu_data.materials, finite_material) ||
        !std::ranges::all_of(cpu_data.lights, finite_light) ||
        !std::ranges::all_of(cpu_data.nodes, [&](auto const &node) { return finite_matrix(node.local_transform); })) {
        return std::unexpected(make_error(LbfErrorType::malformed_payload, "MODL contains a non-finite value"));
    }

    for (auto const &node: cpu_data.nodes) {
        if ((node.mesh_index != no_index && node.mesh_index >= mesh_count) ||
            !std::ranges::all_of(node.children, valid_node)) {
            return std::unexpected(
                    make_error(LbfErrorType::malformed_payload, "node references a missing mesh or node"));
        }
    }

    if (!std::ranges::all_of(cpu_data.scene_roots, valid_node)) {
        return std::unexpected(make_error(LbfErrorType::malformed_payload, "scene root references a missing node"));
    }

    if (!node_graph_is_a_forest(cpu_data)) {
        return std::unexpected(
                make_error(LbfErrorType::malformed_payload, "node graph has a cycle, a shared child, or is too deep"));
    }

    return cooked;
}
