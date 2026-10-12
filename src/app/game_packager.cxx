#include "app/game_packager.hxx"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <format>
#include <fstream>
#include <system_error>
#include <utility>
#include <vector>

#include "assets/shader_bake.hxx"

namespace {

    namespace fs = std::filesystem;

    constexpr auto bake_end = 0.25F;
    constexpr auto assets_end = 0.80F;
    constexpr auto runtime_end = 0.85F;

    struct Context {
        std::mutex *mutex = nullptr;
        std::string *step = nullptr;
        std::atomic<float> *fraction = nullptr;
        std::atomic<bool> *cancelled = nullptr;

        auto set(std::string text, float value) const -> void {
            {
                std::scoped_lock lock{*mutex};
                *step = std::move(text);
            }
            fraction->store(value, std::memory_order_release);
        }

        [[nodiscard]] auto is_cancelled() const -> bool { return cancelled->load(std::memory_order_acquire); }
    };

    auto fail(std::string message) -> std::unexpected<std::string> { return std::unexpected{std::move(message)}; }

    auto describe(std::error_code const &error) -> std::string { return error.message(); }

    struct FileEntry {
        fs::path from;
        fs::path to;
        std::uintmax_t size = 0;
    };

    [[nodiscard]] auto list_files(fs::path const &root, fs::path const &target)
            -> std::expected<std::vector<FileEntry>, std::string> {
        std::vector<FileEntry> files;
        std::error_code error;

        if (!fs::is_directory(root, error)) {
            return files;
        }

        for (fs::recursive_directory_iterator it{root, error}, end; it != end; it.increment(error)) {
            if (error) {
                return fail(std::format("could not read '{}': {}", root.string(), describe(error)));
            }

            if (!it->is_regular_file(error)) {
                continue;
            }

            auto const size = it->file_size(error);
            files.push_back({.from = it->path(), .to = target / fs::relative(it->path(), root), .size = size});
        }

        return files;
    }

    // A package left by an earlier run is replaced, but only if it looks like one, so a mistyped output directory
    // cannot delete anything else.
    [[nodiscard]] auto remove_previous(fs::path const &directory) -> std::expected<void, std::string> {
        std::error_code error;

        if (!fs::exists(directory, error)) {
            return {};
        }

        if (!fs::exists(directory / "data" / "game.toml", error)) {
            return fail(std::format("'{}' exists and is not a package built by the editor; choose another name or "
                                    "output directory",
                                    directory.string()));
        }

        fs::remove_all(directory, error);

        if (error) {
            return fail(std::format("could not replace '{}': {}", directory.string(), describe(error)));
        }

        return {};
    }

    [[nodiscard]] auto run(PackageOptions const &options, Context const &context)
            -> std::expected<PackageResult, std::string> {
        auto const started = std::chrono::steady_clock::now();
        auto const &name = options.manifest.name;

        if (name.empty() || name.find_first_of("/\\") != std::string::npos || name == "." || name == "..") {
            return fail(std::format("'{}' is not a valid package name", name));
        }

        auto const final_dir = options.output_dir / name;
        auto const stage = options.output_dir / (name + ".partial");
        auto const data = stage / "data";
        std::error_code error;

        auto const clean_up = [&] {
            std::error_code ignored;
            fs::remove_all(stage, ignored);
        };

        auto const abort = [&](std::string message) {
            clean_up();
            return fail(std::move(message));
        };

        fs::remove_all(stage, error);
        fs::create_directories(data, error);

        if (error) {
            return abort(std::format("could not create '{}': {}", data.string(), describe(error)));
        }

        // 1. Shaders.
        context.set("Compiling shaders", 0.0F);

        auto const shader_pack = data / "shaders.lsp";
        std::string bake_failure = "no shader compiler is available";

        if (options.compiler != nullptr && options.compiler->valid()) {
            if (auto const baked = renderer::bake_shaders(*options.compiler, shader_pack)) {
                bake_failure.clear();
            } else {
                bake_failure = baked.error();
            }
        }

        if (!bake_failure.empty()) {
            auto const existing = options.data_root / "shaders.lsp";

            if (!fs::exists(existing, error) || !fs::copy_file(existing, shader_pack, error)) {
                return abort(std::format("could not bake shaders ({}) and there is no shaders.lsp to reuse",
                                         bake_failure));
            }
        }

        if (context.is_cancelled()) {
            return abort("cancelled");
        }

        // 2. Game data: assets/ and the cooked scene.
        context.set("Collecting assets", bake_end);

        auto assets = list_files(options.data_root / "assets", data / "assets");

        if (!assets) {
            return abort(assets.error());
        }

        std::uintmax_t total_bytes = 0;

        for (auto const &file: *assets) {
            total_bytes += file.size;
        }

        std::uintmax_t copied_bytes = 0;
        std::size_t file_count = 0;

        for (auto const &file: *assets) {
            if (context.is_cancelled()) {
                return abort("cancelled");
            }

            context.set(std::format("Copying {}", fs::relative(file.from, options.data_root).generic_string()),
                        bake_end + (assets_end - bake_end) * static_cast<float>(static_cast<double>(copied_bytes) /
                                                                                static_cast<double>(std::max<std::uintmax_t>(total_bytes, 1))));

            fs::create_directories(file.to.parent_path(), error);
            fs::copy_file(file.from, file.to, fs::copy_options::overwrite_existing, error);

            if (error) {
                return abort(std::format("could not copy '{}': {}", file.from.string(), describe(error)));
            }

            copied_bytes += file.size;
            ++file_count;
        }

        if (!options.manifest.scene.empty()) {
            context.set("Adding the scene", assets_end);

            auto const scene_target = data / options.manifest.scene;
            fs::create_directories(scene_target.parent_path(), error);
            fs::rename(options.cooked_scene, scene_target, error);

            if (error) {
                // Staging and output can be on different filesystems.
                error.clear();
                fs::copy_file(options.cooked_scene, scene_target, fs::copy_options::overwrite_existing, error);

                if (error) {
                    return abort(std::format("could not add the scene: {}", describe(error)));
                }

                fs::remove(options.cooked_scene, error);
            }

            ++file_count;
        }

        // 3. The runtime: this executable and the shared libraries beside it.
        context.set("Copying the runtime", assets_end);

        auto const executable_dir = options.executable.parent_path();
        fs::copy_file(options.executable, stage / name, fs::copy_options::overwrite_existing, error);

        if (error) {
            return abort(std::format("could not copy the executable: {}", describe(error)));
        }

        fs::permissions(stage / name, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
                        fs::perm_options::add, error);
        ++file_count;

        for (fs::directory_iterator it{executable_dir, error}, end; !error && it != end; it.increment(error)) {
            auto const filename = it->path().filename().string();

            if (it->is_regular_file() && (filename.ends_with(".so") || filename.find(".so.") != std::string::npos)) {
                fs::copy_file(it->path(), stage / filename, fs::copy_options::overwrite_existing, error);
                ++file_count;
            }
        }

        if (error) {
            return abort(std::format("could not copy runtime libraries: {}", describe(error)));
        }

        // 4. Manifest.
        context.set("Writing game.toml", runtime_end);

        if (std::ofstream manifest{data / "game.toml", std::ios::trunc}; !(manifest << options.manifest.to_text())) {
            return abort("could not write game.toml");
        }

        ++file_count;

        if (context.is_cancelled()) {
            return abort("cancelled");
        }

        // 5. Move into place, then archive.
        if (auto removed = remove_previous(final_dir); !removed) {
            return abort(removed.error());
        }

        fs::rename(stage, final_dir, error);

        if (error) {
            return abort(std::format("could not move the package into '{}': {}", final_dir.string(), describe(error)));
        }

        PackageResult result{.directory = final_dir, .files = file_count};

        if (options.archive) {
            context.set("Archiving", runtime_end);

            auto archive = options.output_dir / (name + ".tar");

            if (auto written = write_tar(final_dir, archive); !written) {
                return fail(std::format("the package is at '{}' but archiving failed: {}", final_dir.string(),
                                        written.error()));
            }

            result.archive = std::move(archive);
        }

        for (fs::recursive_directory_iterator it{final_dir, error}, end; !error && it != end; it.increment(error)) {
            if (it->is_regular_file()) {
                result.bytes += it->file_size();
            }
        }

        context.set("Done", 1.0F);
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return result;
    }

    auto octal(char *field, std::size_t width, std::uintmax_t value) -> void {
        // `width` includes the terminating NUL.
        auto const text = std::format("{:0{}o}", value, width - 1);
        std::memcpy(field, text.data(), width - 1);
        field[width - 1] = '\0';
    }

}

auto write_tar(fs::path const &directory, fs::path const &output) -> std::expected<void, std::string> {
    std::ofstream out{output, std::ios::binary | std::ios::trunc};

    if (!out) {
        return fail(std::format("could not open '{}'", output.string()));
    }

    constexpr std::size_t block = 512;
    constexpr std::uintmax_t max_size = 077777777777ULL;
    auto const base = directory.parent_path();
    std::error_code error;
    std::vector<fs::path> files;

    for (fs::recursive_directory_iterator it{directory, error}, end; !error && it != end; it.increment(error)) {
        if (it->is_regular_file()) {
            files.push_back(it->path());
        }
    }

    if (error) {
        return fail(std::format("could not read '{}': {}", directory.string(), describe(error)));
    }

    std::ranges::sort(files);

    for (auto const &file: files) {
        auto const name = fs::relative(file, base).generic_string();
        auto const size = fs::file_size(file, error);

        if (error || size > max_size) {
            return fail(std::format("'{}' is too large for a tar archive", file.string()));
        }

        std::string prefix;
        std::string short_name = name;

        if (name.size() > 100) {
            auto const split = name.rfind('/', 155);

            if (split == std::string::npos || name.size() - split - 1 > 100 || split > 155) {
                return fail(std::format("'{}' has too long a path for a tar archive", name));
            }

            prefix = name.substr(0, split);
            short_name = name.substr(split + 1);
        }

        std::array<char, block> header{};
        std::memcpy(header.data(), short_name.data(), short_name.size());
        octal(header.data() + 100, 8, static_cast<std::uintmax_t>(fs::status(file, error).permissions() & fs::perms::mask) & 0777U);
        octal(header.data() + 108, 8, 0);
        octal(header.data() + 116, 8, 0);
        octal(header.data() + 124, 12, size);
        octal(header.data() + 136, 12, 0);
        std::memset(header.data() + 148, ' ', 8);
        header[156] = '0';
        std::memcpy(header.data() + 257, "ustar", 6);
        std::memcpy(header.data() + 263, "00", 2);
        std::memcpy(header.data() + 345, prefix.data(), prefix.size());

        std::uint32_t checksum = 0;

        for (auto const byte: header) {
            checksum += static_cast<unsigned char>(byte);
        }

        octal(header.data() + 148, 7, checksum);
        header[155] = ' ';
        out.write(header.data(), block);

        std::ifstream in{file, std::ios::binary};
        std::array<char, 1 << 16> buffer{};
        std::uintmax_t remaining = size;

        while (remaining > 0) {
            in.read(buffer.data(), static_cast<std::streamsize>(std::min<std::uintmax_t>(remaining, buffer.size())));
            auto const got = in.gcount();

            if (got <= 0) {
                return fail(std::format("could not read '{}'", file.string()));
            }

            out.write(buffer.data(), got);
            remaining -= static_cast<std::uintmax_t>(got);
        }

        if (auto const padding = (block - size % block) % block; padding != 0) {
            std::array<char, block> zeros{};
            out.write(zeros.data(), static_cast<std::streamsize>(padding));
        }
    }

    std::array<char, block * 2> end{};
    out.write(end.data(), end.size());

    if (!out) {
        return fail(std::format("could not write '{}'", output.string()));
    }

    return {};
}

auto GamePackageJob::start(PackageOptions options) -> GamePackageJob {
    GamePackageJob job;
    job.shared_ = std::make_shared<Shared>();
    job.shared_->step = "Starting";

    std::promise<std::expected<PackageResult, std::string>> promise;
    job.future_ = promise.get_future();

    // Detached: the shared state outlives the job object, and the destructor cancels and waits on the future.
    std::thread{[shared = job.shared_, options = std::move(options), promise = std::move(promise)]() mutable {
        Context context{.mutex = &shared->mutex,
                        .step = &shared->step,
                        .fraction = &shared->fraction,
                        .cancelled = &shared->cancelled};

        promise.set_value(run(options, context));
    }}.detach();

    return job;
}

GamePackageJob::~GamePackageJob() {
    if (shared_ != nullptr && future_.valid()) {
        shared_->cancelled.store(true, std::memory_order_release);
        future_.wait();
    }
}

auto GamePackageJob::ready() const -> bool {
    return future_.valid() && future_.wait_for(std::chrono::seconds{0}) == std::future_status::ready;
}

auto GamePackageJob::progress() const -> PackageProgress {
    std::scoped_lock lock{shared_->mutex};
    return {.fraction = shared_->fraction.load(std::memory_order_acquire), .step = shared_->step};
}

auto GamePackageJob::cancel() -> void { shared_->cancelled.store(true, std::memory_order_release); }

auto GamePackageJob::take() -> std::expected<PackageResult, std::string> { return future_.get(); }
