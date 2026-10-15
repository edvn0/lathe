#include "assets/assimp_import.hxx"

#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <stb_image_write.h>

#include <glm/geometric.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <filesystem>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "assets/model_formats.hxx"
#include "core/logger.hxx"
#include "gpu/sampler_storage.hxx"

namespace {

    // Assimp hands back right-handed, bottom-left-origin UV data, the same handedness glTF has, so the conversion
    // mirrors the glTF path: negate Z, flip the winding and mirror node matrices. Only V needs flipping on top.
    constexpr float z_sign = -1.0F;

    constexpr auto mirror_z = glm::mat4{1.0F, 0.0F, 0.0F,  0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                        0.0F, 0.0F, -1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};

    auto to_glm(aiMatrix4x4 const &matrix) -> glm::mat4 {
        // aiMatrix4x4 is row-major, glm column-major.
        auto const column_major = glm::transpose(glm::make_mat4(&matrix.a1));
        return mirror_z * column_major * mirror_z;
    }

    auto xml_attribute(std::string_view tag, std::string_view name) -> std::optional<std::string_view> {
        for (std::size_t from = 0;;) {
            auto const at = tag.find(name, from);

            if (at == std::string_view::npos) {
                return std::nullopt;
            }

            from = at + name.size();

            if (at == 0 || std::isspace(static_cast<unsigned char>(tag[at - 1])) == 0) {
                continue;
            }

            auto cursor = from;
            auto const skip_space = [&] {
                while (cursor < tag.size() && std::isspace(static_cast<unsigned char>(tag[cursor])) != 0) {
                    ++cursor;
                }
            };

            skip_space();

            if (cursor >= tag.size() || tag[cursor] != '=') {
                continue;
            }

            ++cursor;
            skip_space();

            if (cursor >= tag.size() || (tag[cursor] != '"' && tag[cursor] != '\'')) {
                continue;
            }

            auto const quote = tag[cursor++];
            auto const end = tag.find(quote, cursor);

            if (end == std::string_view::npos) {
                return std::nullopt;
            }

            return tag.substr(cursor, end - cursor);
        }
    }

    // Assimp resolves a Collada <instance_node> by recursing into the node it names, with no guard against a node that
    // (directly or through others) instances itself, so such a file runs the stack out. Build the "contains or
    // instances" graph from the XML and reject cycles, and chains too deep to recurse safely, before Assimp sees it.
    constexpr std::size_t max_collada_instance_depth = 256;

    auto find_collada_instance_problem(std::string_view xml) -> std::optional<std::string> {
        std::map<std::string, std::vector<std::string>> edges;

        struct OpenElement {
            bool has_id = false;
        };

        std::vector<OpenElement> open;
        std::vector<std::string> ids;

        auto const add_edge = [&](std::string_view target) {
            edges[ids.empty() ? std::string{} : ids.back()].emplace_back(target);
        };

        for (std::size_t position = 0;;) {
            auto const lt = xml.find('<', position);

            if (lt == std::string_view::npos) {
                break;
            }

            if (xml.substr(lt).starts_with("<!--")) {
                auto const end = xml.find("-->", lt + 4);
                position = end == std::string_view::npos ? xml.size() : end + 3;
                continue;
            }

            if (xml.substr(lt).starts_with("<![CDATA[")) {
                auto const end = xml.find("]]>", lt + 9);
                position = end == std::string_view::npos ? xml.size() : end + 3;
                continue;
            }

            auto const gt = xml.find('>', lt);

            if (gt == std::string_view::npos) {
                break;
            }

            auto const tag = xml.substr(lt + 1, gt - lt - 1);
            position = gt + 1;

            if (tag.empty() || tag.front() == '?' || tag.front() == '!') {
                continue;
            }

            if (tag.front() == '/') {
                if (!open.empty()) {
                    if (open.back().has_id) {
                        ids.pop_back();
                    }

                    open.pop_back();
                }

                continue;
            }

            auto const name = tag.substr(0, tag.find_first_of(" \t\r\n/"));
            auto const self_closing = tag.back() == '/';
            std::optional<std::string> node_id;

            if (name == "node") {
                if (auto const id = xml_attribute(tag, "id"); id.has_value() && !id->empty()) {
                    node_id = std::string{*id};

                    if (!ids.empty()) {
                        add_edge(*id); // an instance of the parent includes its children
                    }
                }
            } else if (name == "instance_node") {
                if (auto const url = xml_attribute(tag, "url"); url.has_value() && url->starts_with('#')) {
                    add_edge(url->substr(1));
                }
            }

            if (!self_closing) {
                open.push_back({.has_id = node_id.has_value()});

                if (node_id.has_value()) {
                    ids.push_back(*node_id);
                }
            }
        }

        enum class Mark : std::uint8_t { active, done };

        struct Frame {
            std::string id;
            std::size_t next = 0;
        };

        std::map<std::string, Mark> marks;

        for (auto const &[start, ignored]: edges) {
            if (marks.contains(start)) {
                continue;
            }

            std::vector<Frame> stack;
            marks[start] = Mark::active;
            stack.push_back({.id = start});

            while (!stack.empty()) {
                auto &frame = stack.back();
                auto const found = edges.find(frame.id);

                if (found == edges.end() || frame.next == found->second.size()) {
                    marks[frame.id] = Mark::done;
                    stack.pop_back();
                    continue;
                }

                auto const target = found->second[frame.next++];

                if (auto const mark = marks.find(target); mark != marks.end()) {
                    if (mark->second == Mark::active) {
                        return std::format("Collada node '{}' instances itself", target);
                    }

                    continue;
                }

                if (stack.size() >= max_collada_instance_depth) {
                    return std::format("Collada nodes are instanced more than {} levels deep",
                                       max_collada_instance_depth);
                }

                marks[target] = Mark::active;
                stack.push_back({.id = target});
            }
        }

        return std::nullopt;
    }

    auto import_error(std::string message) -> ModelLoadError {
        return ModelLoadError{
                .type = ModelLoadErrorType::parse_error,
                .cause = ErrorCause{ErrorContext{.message = FlyString{std::move(message)}}},
        };
    }

    auto extract_primitive(aiMesh const &mesh) -> std::expected<ModelCpuPrimitive, ModelLoadError> {
        if (mesh.mNumVertices == 0 || mesh.mVertices == nullptr) {
            return std::unexpected(ModelLoadError{.type = ModelLoadErrorType::missing_position});
        }

        std::vector<ModelVertex> vertices(mesh.mNumVertices);
        auto const *const uvs = mesh.HasTextureCoords(0) ? mesh.mTextureCoords[0] : nullptr;
        // Tangents the file carries are kept; anything else goes through MikkTSpace with the glTF meshes.
        auto const has_tangents = mesh.HasTangentsAndBitangents() && mesh.HasNormals();

        for (std::uint32_t index = 0; index < mesh.mNumVertices; ++index) {
            auto &vertex = vertices[index];
            auto const &position = mesh.mVertices[index];

            vertex.position = glm::vec3{position.x, position.y, z_sign * position.z};
            vertex.normal = glm::vec3{0.0F, 1.0F, 0.0F};
            vertex.tangent = glm::vec4{1.0F, 0.0F, 0.0F, 1.0F};
            vertex.texcoord = glm::vec2{0.0F};

            if (mesh.HasNormals()) {
                auto const &normal = mesh.mNormals[index];
                vertex.normal = glm::vec3{normal.x, normal.y, z_sign * normal.z};
            }

            if (uvs != nullptr) {
                vertex.texcoord = glm::vec2{uvs[index].x, 1.0F - uvs[index].y};
            }

            if (has_tangents) {
                auto const &tangent = mesh.mTangents[index];
                auto const &bitangent = mesh.mBitangents[index];
                auto const normal = aiVector3D{mesh.mNormals[index]};
                auto const handedness = ((normal ^ tangent) * bitangent) < 0.0F ? -1.0F : 1.0F;

                vertex.tangent = glm::vec4{tangent.x, tangent.y, z_sign * tangent.z, z_sign * handedness};
            }
        }

        std::vector<std::uint32_t> indices;
        indices.reserve(static_cast<std::size_t>(mesh.mNumFaces) * 3);

        for (std::uint32_t face_index = 0; face_index < mesh.mNumFaces; ++face_index) {
            auto const &face = mesh.mFaces[face_index];

            // Triangulate + SortByPType leave only triangles, but degenerate faces can survive as lines.
            if (face.mNumIndices != 3) {
                continue;
            }

            // Mirroring Z reverses the winding.
            indices.push_back(face.mIndices[0]);
            indices.push_back(face.mIndices[2]);
            indices.push_back(face.mIndices[1]);
        }

        if (indices.empty()) {
            return std::unexpected(ModelLoadError{.type = ModelLoadErrorType::unsupported_primitive});
        }

        return ModelCpuPrimitive{
                .vertices = std::move(vertices),
                .indices = std::move(indices),
                .material_index = mesh.mMaterialIndex,
                .has_tangents = has_tangents && uvs != nullptr,
        };
    }

    // aiTexel is BGRA in memory order; PNG wants RGBA.
    auto encode_raw_texture(aiTexture const &texture) -> std::optional<std::vector<std::byte>> {
        auto const texel_count = static_cast<std::size_t>(texture.mWidth) * texture.mHeight;
        std::vector<unsigned char> rgba(texel_count * 4);

        for (std::size_t index = 0; index < texel_count; ++index) {
            auto const &texel = texture.pcData[index];
            rgba[index * 4 + 0] = texel.r;
            rgba[index * 4 + 1] = texel.g;
            rgba[index * 4 + 2] = texel.b;
            rgba[index * 4 + 3] = texel.a;
        }

        std::vector<std::byte> encoded;

        auto const append = [](void *context, void *data, int size) {
            auto &sink = *static_cast<std::vector<std::byte> *>(context);
            auto const *bytes = static_cast<std::byte const *>(data);
            sink.insert(sink.end(), bytes, bytes + size);
        };

        auto const ok = stbi_write_png_to_func(append, &encoded, static_cast<int>(texture.mWidth),
                                               static_cast<int>(texture.mHeight), 4, rgba.data(),
                                               static_cast<int>(texture.mWidth * 4));

        if (ok == 0 || encoded.empty()) {
            return std::nullopt;
        }

        return encoded;
    }

    struct TextureRequest {
        aiTextureType type;
        ModelTextureSlot slot;
        std::string_view slot_name;
    };

    class ImageCollector {
    public:
        ImageCollector(aiScene const &scene, AssetPath const &model_path, ModelCpuData &cpu_data)
            : scene_(scene), model_path_(model_path), cpu_data_(cpu_data) {}

        // The first of `types` the material has a usable texture for, as an index into cpu_data.image_sources.
        auto resolve(aiMaterial const &material, std::string_view material_name,
                     std::initializer_list<TextureRequest> requests) -> std::optional<std::size_t> {
            for (auto const &request: requests) {
                if (material.GetTextureCount(request.type) == 0) {
                    continue;
                }

                aiString raw;
                if (material.GetTexture(request.type, 0, &raw) != aiReturn_SUCCESS || raw.length == 0) {
                    continue;
                }

                if (auto const found = add(raw.C_Str(), request.slot, request.slot_name, material_name)) {
                    return found;
                }
            }

            return std::nullopt;
        }

    private:
        auto add(std::string const &reference, ModelTextureSlot slot, std::string_view slot_name,
                 std::string_view material_name) -> std::optional<std::size_t> {
            auto const cache_id = std::pair{reference, slot};

            if (auto const cached = cache_.find(cache_id); cached != cache_.end()) {
                return cached->second;
            }

            ModelCpuImageSource source;
            source.slot = slot;
            source.debug_name = FlyString{material_name.empty() ? std::format("image_{}", cpu_data_.image_sources.size())
                                                                : std::format("{}_{}", material_name, slot_name)};

            if (auto const *embedded = scene_.GetEmbeddedTexture(reference.c_str())) {
                if (embedded->pcData == nullptr) {
                    warn("[load_model_cpu_assimp]: '{}' has an empty embedded texture; skipped", reference);
                    return std::nullopt;
                }

                if (embedded->mHeight == 0) {
                    auto const *bytes = reinterpret_cast<std::byte const *>(embedded->pcData);
                    source.encoded.assign(bytes, bytes + embedded->mWidth);
                } else if (auto png = encode_raw_texture(*embedded)) {
                    // USD hands textures over decoded; re-encode them so they take the streamer's usual path.
                    source.encoded = std::move(*png);
                } else {
                    warn("[load_model_cpu_assimp]: '{}' could not be re-encoded; skipped", reference);
                    return std::nullopt;
                }

                source.cache_key = std::format("{}#assimp:{}#{}", model_path_.key(), reference, source.encoded.size());
            } else {
                auto relative = reference;
                std::ranges::replace(relative, '\\', '/');

                auto path = model_path_.sibling(relative);

                // FBX in particular stores the artist's absolute path; fall back to the file name beside the model.
                if (!path || !std::filesystem::exists(path->absolute())) {
                    path = model_path_.sibling(std::filesystem::path{relative}.filename());
                }

                if (!path || !std::filesystem::exists(path->absolute())) {
                    warn("[load_model_cpu_assimp]: texture '{}' not found beside '{}'", reference, model_path_.key());
                    return std::nullopt;
                }

                source.path = std::move(path);
            }

            auto const index = cpu_data_.image_sources.size();
            cpu_data_.image_sources.push_back(std::move(source));
            cache_.emplace(cache_id, index);

            return index;
        }

        aiScene const &scene_;
        AssetPath const &model_path_;
        ModelCpuData &cpu_data_;
        std::map<std::pair<std::string, ModelTextureSlot>, std::size_t> cache_;
    };

    auto convert_material(aiMaterial const &material, std::size_t index, SamplerHandle sampler,
                          ImageCollector &images) -> ModelCpuMaterial {
        ModelCpuMaterial result{};
        result.sampler = sampler;

        aiString name_string;
        material.Get(AI_MATKEY_NAME, name_string);
        auto const name = name_string.length != 0 ? std::string{name_string.C_Str()} : std::format("material_{}", index);

        aiColor4D colour{1.0F, 1.0F, 1.0F, 1.0F};
        if (material.Get(AI_MATKEY_BASE_COLOR, colour) != aiReturn_SUCCESS) {
            material.Get(AI_MATKEY_COLOR_DIFFUSE, colour);
        }

        float opacity = 1.0F;
        material.Get(AI_MATKEY_OPACITY, opacity);

        result.base_colour_factor = glm::vec4{colour.r, colour.g, colour.b, colour.a * opacity};

        if (result.base_colour_factor.w < 0.999F) {
            result.alpha_mode = AlphaMode::blend;
        }

        aiColor3D emissive{0.0F, 0.0F, 0.0F};
        material.Get(AI_MATKEY_COLOR_EMISSIVE, emissive);
        result.emissive_factor = glm::vec3{emissive.r, emissive.g, emissive.b};

        int two_sided = 0;
        material.Get(AI_MATKEY_TWOSIDED, two_sided);
        result.double_sided = two_sided != 0;

        // Legacy Phong materials carry no PBR factors. Map the specular exponent onto roughness (the usual
        // Blinn-Phong equivalence) and treat them as dielectrics.
        float metallic = 0.0F;
        float roughness = 1.0F;
        auto const has_metallic = material.Get(AI_MATKEY_METALLIC_FACTOR, metallic) == aiReturn_SUCCESS;
        auto const has_roughness = material.Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness) == aiReturn_SUCCESS;

        if (!has_roughness) {
            float shininess = 0.0F;

            if (material.Get(AI_MATKEY_SHININESS, shininess) == aiReturn_SUCCESS && shininess > 0.0F) {
                roughness = std::clamp(std::sqrt(2.0F / (shininess + 2.0F)), 0.04F, 1.0F);
            }
        }

        result.metallic_factor = has_metallic ? metallic : 0.0F;
        result.roughness_factor = roughness;

        result.base_colour_image = images.resolve(material, name,
                                                  {{aiTextureType_BASE_COLOR, ModelTextureSlot::base_colour, "basecolor"},
                                                   {aiTextureType_DIFFUSE, ModelTextureSlot::base_colour, "basecolor"}});

        // OBJ's `map_Bump` / `bump` arrive as HEIGHT; they are tangent-space normal maps in practice.
        result.normal_image = images.resolve(material, name,
                                             {{aiTextureType_NORMAL_CAMERA, ModelTextureSlot::normal, "normal"},
                                              {aiTextureType_NORMALS, ModelTextureSlot::normal, "normal"},
                                              {aiTextureType_HEIGHT, ModelTextureSlot::normal, "normal"}});

        // The engine samples glTF's packed layout (G roughness, B metallic); only a texture already packed that way
        // is usable, which Assimp reports as UNKNOWN for glTF-derived FBX exports.
        result.metallic_roughness_image =
                images.resolve(material, name, {{aiTextureType_UNKNOWN, ModelTextureSlot::metallic_roughness, "metallic_roughness"}});

        result.occlusion_image = images.resolve(
                material, name,
                {{aiTextureType_AMBIENT_OCCLUSION, ModelTextureSlot::occlusion, "occlusion"},
                 {aiTextureType_LIGHTMAP, ModelTextureSlot::occlusion, "occlusion"}});

        result.emissive_image = images.resolve(
                material, name,
                {{aiTextureType_EMISSION_COLOR, ModelTextureSlot::emissive, "emissive"},
                 {aiTextureType_EMISSIVE, ModelTextureSlot::emissive, "emissive"}});

        if (result.emissive_image.has_value() && glm::dot(result.emissive_factor, result.emissive_factor) == 0.0F) {
            result.emissive_factor = glm::vec3{1.0F};
        }

        return result;
    }

    // Flattens the aiNode tree into cpu_data.nodes and returns the new node's index. A node holding several meshes
    // keeps the first and gets one identity child per extra mesh, since ModelNode references a single mesh.
    auto append_node(aiNode const &source, ModelCpuData &cpu_data) -> std::uint32_t {
        auto const index = static_cast<std::uint32_t>(cpu_data.nodes.size());
        cpu_data.nodes.emplace_back();
        cpu_data.nodes[index].local_transform = to_glm(source.mTransformation);

        if (source.mNumMeshes > 0) {
            cpu_data.nodes[index].mesh_index = source.mMeshes[0];
        }

        for (std::uint32_t extra = 1; extra < source.mNumMeshes; ++extra) {
            auto const child_index = static_cast<std::uint32_t>(cpu_data.nodes.size());
            cpu_data.nodes.emplace_back();
            cpu_data.nodes[child_index].mesh_index = source.mMeshes[extra];
            cpu_data.nodes[index].children.push_back(child_index);
        }

        for (std::uint32_t child = 0; child < source.mNumChildren; ++child) {
            auto const child_index = append_node(*source.mChildren[child], cpu_data);
            cpu_data.nodes[index].children.push_back(child_index);
        }

        return index;
    }

} // namespace

auto load_model_cpu_assimp(AssetPath const &path, SamplerStorage &sampler_storage,
                           std::shared_ptr<ModelLoadProfile> profile) -> std::expected<ModelCpuData, ModelLoadError> {
    ZoneScopedNC("LoadModelCpuAssimp", tracy::Color::Goldenrod);

    if (!std::filesystem::exists(path.absolute())) {
        return std::unexpected(ModelLoadError{.type = ModelLoadErrorType::file_not_found});
    }

    if (model_extension_of(path.absolute()) == ".dae") {
        std::ifstream file{path.absolute(), std::ios::binary};
        std::string xml{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};

        if (auto const problem = find_collada_instance_problem(xml)) {
            return std::unexpected(import_error(*problem));
        }
    }

    auto *const profile_ptr = profile.get();
    ScopedProfileSample parse_sample{profile_ptr != nullptr ? &profile_ptr->gltf_parse_ns : nullptr};

    Assimp::Importer importer;
    importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);
    // FBX is authored in centimetres by default; the engine works in metres like glTF.
    importer.SetPropertyBool(AI_CONFIG_FBX_CONVERT_TO_M, true);
    importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_PRESERVE_PIVOTS, false);

    constexpr unsigned int flags = aiProcess_Triangulate | aiProcess_SortByPType | aiProcess_GenSmoothNormals |
                                   aiProcess_GenUVCoords | aiProcess_TransformUVCoords | aiProcess_FindDegenerates |
                                   aiProcess_FindInvalidData | aiProcess_ValidateDataStructure;

    auto const *const scene = importer.ReadFile(path.absolute().string().c_str(), flags);
    parse_sample.stop();

    if (scene == nullptr || scene->mRootNode == nullptr || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) != 0) {
        return std::unexpected(import_error(std::format("assimp: {}", importer.GetErrorString())));
    }

    debug("[load_model_cpu_assimp]: parsed {} meshes, {} materials, {} embedded textures", scene->mNumMeshes,
          scene->mNumMaterials, scene->mNumTextures);

    ModelCpuData cpu_data;
    cpu_data.profile = std::move(profile);
    cpu_data.meshes.reserve(scene->mNumMeshes);
    cpu_data.materials.reserve(scene->mNumMaterials);

    ImageCollector images{*scene, path, cpu_data};
    auto const sampler = sampler_storage.linear_repeat();

    for (std::uint32_t index = 0; index < scene->mNumMaterials; ++index) {
        ScopedProfileSample const material_sample{profile_ptr != nullptr ? &profile_ptr->material_resolve_ns : nullptr};

        cpu_data.materials.push_back(convert_material(*scene->mMaterials[index], index, sampler, images));
    }

    // One ModelCpuMesh per aiMesh, so a mesh instanced by several nodes is still uploaded once.
    for (std::uint32_t index = 0; index < scene->mNumMeshes; ++index) {
        ScopedProfileSample const extract_sample{profile_ptr != nullptr ? &profile_ptr->primitive_extract_ns : nullptr};

        ModelCpuMesh mesh;
        auto primitive = extract_primitive(*scene->mMeshes[index]);

        if (primitive) {
            if (primitive->material_index.has_value() && *primitive->material_index >= cpu_data.materials.size()) {
                return std::unexpected(ModelLoadError{.type = ModelLoadErrorType::invalid_material_index});
            }

            mesh.primitives.push_back(std::move(*primitive));
        } else {
            // A mesh with nothing drawable (e.g. all lines) must still occupy its slot so node mesh indices hold.
            warn("[load_model_cpu_assimp]: mesh {} of '{}' skipped: {}", index, path.key(), primitive.error().type);
        }

        cpu_data.meshes.push_back(std::move(mesh));
    }

    cpu_data.scene_roots.push_back(append_node(*scene->mRootNode, cpu_data));

    return cpu_data;
}
