#include "assets/shader_pack.hxx"

#include <atomic>
#include <bit>
#include <cstring>
#include <fstream>
#include <mutex>

namespace renderer {

    namespace {
        constexpr char magic[4] = {'L', 'S', 'H', 'P'};
        constexpr std::uint32_t format_version = 1;

        template<typename T>
        auto write_value(std::ofstream &out, T value) -> void {
            out.write(reinterpret_cast<char const *>(&value), sizeof(T)); // NOLINT
        }

        template<typename T>
        auto read_value(std::ifstream &in, T &value) -> bool {
            return static_cast<bool>(in.read(reinterpret_cast<char *>(&value), sizeof(T))); // NOLINT
        }

        auto write_string(std::ofstream &out, std::string const &text) -> void {
            write_value(out, static_cast<std::uint32_t>(text.size()));
            out.write(text.data(), static_cast<std::streamsize>(text.size()));
        }

        auto read_string(std::ifstream &in, std::string &text) -> bool {
            std::uint32_t size = 0;

            if (!read_value(in, size) || size > (1U << 20)) {
                return false;
            }

            text.resize(size);

            return static_cast<bool>(in.read(text.data(), static_cast<std::streamsize>(size)));
        }

        std::atomic<std::shared_ptr<ShaderPack const>> installed_pack;

        struct Recording {
            std::mutex mutex;
            std::optional<ShaderPack> pack;
        };

        auto recording() -> Recording & {
            static Recording instance;

            return instance;
        }
    }

    auto ShaderPack::add(std::string key, CompiledShader const &shader) -> void {
        entries_.insert_or_assign(std::move(key), Entry{
                                                           .stage = shader.stage,
                                                           .entry_point = std::string{shader.entry_point.view()},
                                                           .spirv = shader.spirv,
                                           });
    }

    auto ShaderPack::find(std::string_view key) const -> std::optional<CompiledShader> {
        auto const it = entries_.find(std::string{key});

        if (it == entries_.end()) {
            return std::nullopt;
        }

        return CompiledShader{
                .stage = it->second.stage,
                .entry_point = FlyString{it->second.entry_point},
                .spirv = it->second.spirv,
        };
    }

    auto ShaderPack::save(std::filesystem::path const &path) const -> std::expected<void, std::string> {
        static_assert(std::endian::native == std::endian::little, "shader packs are little-endian");

        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);

        std::ofstream out{path, std::ios::binary | std::ios::trunc};

        if (!out) {
            return std::unexpected{std::format("could not open '{}' for writing", path.string())};
        }

        out.write(magic, sizeof(magic));
        write_value(out, format_version);
        write_value(out, static_cast<std::uint32_t>(entries_.size()));

        for (auto const &[key, entry]: entries_) {
            write_string(out, key);
            write_value(out, static_cast<std::uint8_t>(entry.stage));
            write_string(out, entry.entry_point);
            write_value(out, static_cast<std::uint32_t>(entry.spirv.size()));
            out.write(reinterpret_cast<char const *>(entry.spirv.data()), // NOLINT
                      static_cast<std::streamsize>(entry.spirv.size() * sizeof(std::uint32_t)));
        }

        if (!out) {
            return std::unexpected{std::format("failed while writing '{}'", path.string())};
        }

        return {};
    }

    auto ShaderPack::load(std::filesystem::path const &path) -> std::expected<ShaderPack, std::string> {
        std::ifstream in{path, std::ios::binary};

        if (!in) {
            return std::unexpected{std::format("could not open '{}'", path.string())};
        }

        char header[4]{};
        std::uint32_t version = 0;
        std::uint32_t count = 0;

        if (!in.read(header, sizeof(header)) || std::memcmp(header, magic, sizeof(magic)) != 0 ||
            !read_value(in, version) || version != format_version || !read_value(in, count)) {
            return std::unexpected{std::format("'{}' is not a shader pack of version {}", path.string(), format_version)};
        }

        ShaderPack pack;

        for (std::uint32_t index = 0; index < count; ++index) {
            std::string key;
            Entry entry;
            std::uint8_t stage = 0;
            std::uint32_t words = 0;

            if (!read_string(in, key) || !read_value(in, stage) || !read_string(in, entry.entry_point) ||
                !read_value(in, words) || words > (1U << 26)) {
                return std::unexpected{std::format("'{}' is truncated or corrupt", path.string())};
            }

            entry.stage = static_cast<ShaderStage>(stage);
            entry.spirv.resize(words);

            if (!in.read(reinterpret_cast<char *>(entry.spirv.data()), // NOLINT
                         static_cast<std::streamsize>(words * sizeof(std::uint32_t)))) {
                return std::unexpected{std::format("'{}' is truncated or corrupt", path.string())};
            }

            pack.entries_.insert_or_assign(std::move(key), std::move(entry));
        }

        return pack;
    }

    auto shader_request_key(ShaderCompileRequest const &request) -> std::string {
        auto key = std::format("{}|{}|{}|o{}d{}", request.source_path.logical(), request.entry_point,
                               static_cast<int>(std::to_underlying(request.stage)), request.optimize ? 1 : 0,
                               request.generate_debug_info ? 1 : 0);

        for (auto const &define: request.defines) {
            key += std::format("|{}={}", define.name, define.value);
        }

        return key;
    }

    auto install_shader_pack(std::shared_ptr<ShaderPack const> pack) -> void { installed_pack.store(std::move(pack)); }

    auto installed_shader_pack() -> std::shared_ptr<ShaderPack const> { return installed_pack.load(); }

    auto start_shader_recording() -> void {
        auto &state = recording();
        std::scoped_lock const lock{state.mutex};

        state.pack.emplace();
    }

    auto shader_recording() -> bool {
        auto &state = recording();
        std::scoped_lock const lock{state.mutex};

        return state.pack.has_value();
    }

    auto record_compiled_shader(ShaderCompileRequest const &request, CompiledShader const &shader) -> void {
        auto &state = recording();
        std::scoped_lock const lock{state.mutex};

        if (state.pack) {
            state.pack->add(shader_request_key(request), shader);
        }
    }

    auto finish_shader_recording(std::filesystem::path const &path) -> std::expected<std::size_t, std::string> {
        auto &state = recording();
        std::scoped_lock const lock{state.mutex};

        if (!state.pack) {
            return std::unexpected{"shader recording was not started"};
        }

        auto const count = state.pack->size();
        auto saved = state.pack->save(path);
        state.pack.reset();

        if (!saved) {
            return std::unexpected{std::move(saved.error())};
        }

        return count;
    }

}
