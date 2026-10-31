#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "assets/gpu_texture_service.hxx"
#include "assets/texture_cache.hxx"
#include "assets/texture_pipeline.hxx"
#include "core/paths.hxx"

#ifndef TEST_ASSETS_DIR
#error "TEST_ASSETS_DIR must be defined by the build"
#endif

namespace {

    constexpr auto marker = std::byte{0x5A};

    // Stands in for the GPU: every block is the marker, so a test can tell whose blocks it got.
    auto fake_blocks(std::span<GpuTextureLevel const> levels) -> GpuTextureService::Outcome {
        GpuTextureService::Blocks blocks;

        for (auto const &level: levels) {
            blocks.emplace_back(block_count_4x4(level.width, level.height) * 16, marker);
        }

        return blocks;
    }

    struct Recorder {
        std::mutex mutex;
        std::vector<std::size_t> batch_levels;
        std::vector<GpuBlockFormat> formats;
        int calls = 0;
    };

    auto recording_backend(std::shared_ptr<Recorder> const &recorder) -> GpuTextureService::Backend {
        return [recorder](std::span<GpuTextureLevel const> levels, GpuBlockFormat format) {
            {
                std::scoped_lock const lock{recorder->mutex};
                recorder->batch_levels.push_back(levels.size());
                recorder->formats.push_back(format);
                ++recorder->calls;
            }

            return fake_blocks(levels);
        };
    }

    // Holds the backend until released, so requests pile up behind it.
    struct Gate {
        std::mutex mutex;
        std::condition_variable changed;
        bool entered = false;
        bool open = false;

        auto wait_until_entered() -> void {
            std::unique_lock lock{mutex};
            changed.wait(lock, [this] { return entered; });
        }

        auto release() -> void {
            {
                std::scoped_lock const lock{mutex};
                open = true;
            }

            changed.notify_all();
        }

        auto pass() -> void {
            std::unique_lock lock{mutex};
            entered = true;
            changed.notify_all();
            changed.wait(lock, [this] { return open; });
        }
    };

    struct Image {
        std::vector<std::byte> pixels;
        GpuTextureLevel level;

        Image(std::uint32_t width, std::uint32_t height) :
            pixels(static_cast<std::size_t>(width) * height * 4),
            level{.rgba = pixels, .width = width, .height = height} {}
    };

    auto make_temp_dir(std::string_view label) -> std::filesystem::path {
        auto const dir = std::filesystem::temp_directory_path() /
                         std::format("gpu_texture_test_{}_{}", label,
                                     std::chrono::steady_clock::now().time_since_epoch().count());

        std::filesystem::create_directories(dir);

        return dir;
    }

    struct ScopedService {
        explicit ScopedService(std::shared_ptr<GpuTextureService> service) {
            install_gpu_texture_service(std::move(service));
        }

        ~ScopedService() { install_gpu_texture_service(nullptr); }

        ScopedService(ScopedService const &) = delete;
        auto operator=(ScopedService const &) -> ScopedService & = delete;
    };

} // namespace

TEST_SUITE("unit") {

    TEST_CASE("block_count_4x4 rounds up to whole blocks") {
        CHECK(block_count_4x4(4, 4) == 1);
        CHECK(block_count_4x4(5, 4) == 2);
        CHECK(block_count_4x4(1, 1) == 1);
        CHECK(block_count_4x4(2048, 2048) == 512U * 512U);
        CHECK(block_count_4x4(6, 10) == 2U * 3U);
    }

    TEST_CASE("GpuTextureService: a request gets back the blocks for its own levels") {
        auto const recorder = std::make_shared<Recorder>();
        GpuTextureService service{recording_backend(recorder)};

        Image first{16, 16};
        Image second{4, 4};
        std::array const levels{first.level, second.level};

        auto outcome = service.submit(levels, GpuBlockFormat::bc7).get();

        REQUIRE(outcome.has_value());
        REQUIRE(outcome->size() == 2);
        CHECK((*outcome)[0].size() == 16U * 16);
        CHECK((*outcome)[1].size() == 16);
        CHECK(service.usable());
    }

    TEST_CASE("GpuTextureService: requests that queue up behind a running batch go to the GPU together") {
        auto const gate = std::make_shared<Gate>();
        auto const recorder = std::make_shared<Recorder>();
        GpuTextureService service{[gate, recorder](std::span<GpuTextureLevel const> levels, GpuBlockFormat format) {
            {
                std::scoped_lock const lock{recorder->mutex};
                recorder->batch_levels.push_back(levels.size());
                recorder->formats.push_back(format);
            }

            // Only the first batch waits.
            if (recorder->batch_levels.size() == 1) {
                gate->pass();
            }

            return fake_blocks(levels);
        }};

        Image a{8, 8};
        Image b{8, 8};
        Image c{8, 8};
        Image d{8, 8};

        auto first = service.submit(std::span{&a.level, 1}, GpuBlockFormat::bc7);
        gate->wait_until_entered();

        auto second = service.submit(std::span{&b.level, 1}, GpuBlockFormat::bc7);
        auto third = service.submit(std::span{&c.level, 1}, GpuBlockFormat::bc7);
        auto fourth = service.submit(std::span{&d.level, 1}, GpuBlockFormat::bc7);

        gate->release();

        for (auto *future: {&first, &second, &third, &fourth}) {
            CHECK(future->get().has_value());
        }

        REQUIRE(recorder->batch_levels.size() == 2);
        CHECK(recorder->batch_levels[0] == 1);
        CHECK(recorder->batch_levels[1] == 3);

        auto const stats = service.stats();
        CHECK(stats.requests == 4);
        CHECK(stats.batches == 2);
        CHECK(stats.largest_batch == 3);
    }

    TEST_CASE("GpuTextureService: a batch never mixes formats") {
        auto const gate = std::make_shared<Gate>();
        auto const recorder = std::make_shared<Recorder>();
        GpuTextureService service{[gate, recorder](std::span<GpuTextureLevel const> levels, GpuBlockFormat format) {
            std::size_t call = 0;

            {
                std::scoped_lock const lock{recorder->mutex};
                recorder->batch_levels.push_back(levels.size());
                recorder->formats.push_back(format);
                call = recorder->batch_levels.size();
            }

            if (call == 1) {
                gate->pass();
            }

            return fake_blocks(levels);
        }};

        Image a{8, 8};
        Image b{8, 8};
        Image c{8, 8};
        Image d{8, 8};

        auto first = service.submit(std::span{&a.level, 1}, GpuBlockFormat::bc7);
        gate->wait_until_entered();

        auto colour = service.submit(std::span{&b.level, 1}, GpuBlockFormat::bc7);
        auto normal = service.submit(std::span{&c.level, 1}, GpuBlockFormat::bc5);
        auto more_colour = service.submit(std::span{&d.level, 1}, GpuBlockFormat::bc7);

        gate->release();

        CHECK(first.get().has_value());
        CHECK(colour.get().has_value());
        CHECK(normal.get().has_value());
        CHECK(more_colour.get().has_value());

        // The first request, then the colour request alone (the normal map is next in line), then the normal map, then
        // the last colour request.
        REQUIRE(recorder->formats.size() == 4);
        CHECK(recorder->formats[1] == GpuBlockFormat::bc7);
        CHECK(recorder->formats[2] == GpuBlockFormat::bc5);
        CHECK(recorder->formats[3] == GpuBlockFormat::bc7);
    }

    TEST_CASE("GpuTextureService: the batch budget splits a large backlog") {
        auto const gate = std::make_shared<Gate>();
        auto const recorder = std::make_shared<Recorder>();
        // Room for two 8x8 images (256 bytes each) per batch.
        GpuTextureService service{[gate, recorder](std::span<GpuTextureLevel const> levels, GpuBlockFormat format) {
                                      std::size_t call = 0;

                                      {
                                          std::scoped_lock const lock{recorder->mutex};
                                          recorder->batch_levels.push_back(levels.size());
                                          recorder->formats.push_back(format);
                                          call = recorder->batch_levels.size();
                                      }

                                      if (call == 1) {
                                          gate->pass();
                                      }

                                      return fake_blocks(levels);
                                  },
                                  512};

        std::vector<std::unique_ptr<Image>> images;
        std::vector<std::future<GpuTextureService::Outcome>> futures;

        images.push_back(std::make_unique<Image>(8, 8));
        futures.push_back(service.submit(std::span{&images.back()->level, 1}, GpuBlockFormat::bc7));
        gate->wait_until_entered();

        for (int index = 0; index < 5; ++index) {
            images.push_back(std::make_unique<Image>(8, 8));
            futures.push_back(service.submit(std::span{&images.back()->level, 1}, GpuBlockFormat::bc7));
        }

        gate->release();

        for (auto &future: futures) {
            CHECK(future.get().has_value());
        }

        REQUIRE(recorder->batch_levels.size() == 4);
        CHECK(recorder->batch_levels[1] == 2);
        CHECK(recorder->batch_levels[2] == 2);
        CHECK(recorder->batch_levels[3] == 1);
    }

    TEST_CASE("GpuTextureService: after a failure every request fails at once and the backend is not asked again") {
        std::atomic<int> calls{0};
        GpuTextureService service{
                [&calls](std::span<GpuTextureLevel const>, GpuBlockFormat) -> GpuTextureService::Outcome {
                    ++calls;

                    return std::unexpected{std::string{"device lost"}};
                }};

        Image image{8, 8};

        auto failed = service.submit(std::span{&image.level, 1}, GpuBlockFormat::bc7).get();
        REQUIRE_FALSE(failed.has_value());
        CHECK(failed.error() == "device lost");
        CHECK_FALSE(service.usable());

        auto later = service.submit(std::span{&image.level, 1}, GpuBlockFormat::bc7).get();
        REQUIRE_FALSE(later.has_value());
        CHECK(later.error().find("failed earlier") != std::string::npos);
        CHECK(calls.load() == 1);
        CHECK(service.stats().failed_requests == 1);
    }

    TEST_CASE("GpuTextureService: a backend that returns the wrong number of levels counts as failed") {
        GpuTextureService service{[](std::span<GpuTextureLevel const>, GpuBlockFormat) -> GpuTextureService::Outcome {
            return GpuTextureService::Blocks{};
        }};

        Image image{8, 8};

        CHECK_FALSE(service.submit(std::span{&image.level, 1}, GpuBlockFormat::bc7).get().has_value());
        CHECK_FALSE(service.usable());
    }

    TEST_CASE("GpuTextureService: shutting down fails what is queued, and later requests") {
        auto const gate = std::make_shared<Gate>();
        auto const recorder = std::make_shared<Recorder>();
        GpuTextureService service{[gate, recorder](std::span<GpuTextureLevel const> levels, GpuBlockFormat) {
            gate->pass();

            return fake_blocks(levels);
        }};

        Image a{8, 8};
        Image b{8, 8};

        auto running = service.submit(std::span{&a.level, 1}, GpuBlockFormat::bc7);
        gate->wait_until_entered();
        auto queued = service.submit(std::span{&b.level, 1}, GpuBlockFormat::bc7);

        std::jthread shutdown{[&service] { service.shutdown(); }};
        gate->release();
        shutdown.join();

        CHECK(running.get().has_value());

        auto const queued_outcome = queued.get();
        // Taken by the thread before it saw the shutdown, or failed by it: both are fine, but it must not hang.
        if (!queued_outcome) {
            CHECK(queued_outcome.error().find("shut down") != std::string::npos);
        }

        CHECK_FALSE(service.usable());
        CHECK_FALSE(service.submit(std::span{&b.level, 1}, GpuBlockFormat::bc7).get().has_value());

        service.shutdown();
    }

    TEST_CASE("GpuTextureService: many threads can submit at once") {
        auto const recorder = std::make_shared<Recorder>();
        GpuTextureService service{recording_backend(recorder)};
        std::atomic<int> ok{0};

        {
            std::vector<std::jthread> workers;

            for (int worker = 0; worker < 8; ++worker) {
                workers.emplace_back([&] {
                    for (int round = 0; round < 25; ++round) {
                        Image image{16, 16};
                        auto outcome = service.submit(std::span{&image.level, 1}, GpuBlockFormat::bc7).get();

                        if (outcome && outcome->size() == 1 && outcome->front().size() == 16U * 16) {
                            ++ok;
                        }
                    }
                });
            }
        }

        CHECK(ok.load() == 200);
        CHECK(service.stats().requests == 200);
    }

    TEST_CASE("install_gpu_texture_service: installing replaces and shuts down the previous service") {
        auto first = std::make_shared<GpuTextureService>(recording_backend(std::make_shared<Recorder>()));
        auto second = std::make_shared<GpuTextureService>(recording_backend(std::make_shared<Recorder>()));

        CHECK(gpu_texture_service() == nullptr);

        install_gpu_texture_service(first);
        CHECK(gpu_texture_service() == first);
        CHECK(first->usable());

        install_gpu_texture_service(second);
        CHECK(gpu_texture_service() == second);
        CHECK_FALSE(first->usable());

        install_gpu_texture_service(nullptr);
        CHECK(gpu_texture_service() == nullptr);
        CHECK_FALSE(second->usable());
    }

    TEST_CASE("texture pipeline: an installed GPU service compresses colour textures to BC7 and normal maps to BC5") {
        auto const dir = make_temp_dir("pipeline");
        auto const recorder = std::make_shared<Recorder>();
        ScopedService const installed{std::make_shared<GpuTextureService>(recording_backend(recorder))};

        auto const albedo = AssetPath::external(std::filesystem::path{TEST_ASSETS_DIR} /
                                                "assets/textures/terrain/terrain_albedo.jpg")
                                    .value();
        auto const normal = AssetPath::external(std::filesystem::path{TEST_ASSETS_DIR} /
                                                "assets/textures/terrain/terrain_normal.jpg")
                                    .value();

        auto colour = load_compressed_texture(albedo, TextureRole::colour, TextureCache{dir});
        REQUIRE(colour.has_value());
        CHECK(colour->format == VK_FORMAT_BC7_SRGB_BLOCK);
        CHECK_FALSE(validate_compressed_texture(*colour).has_value());

        // The blocks are the fake GPU's, not the CPU encoder's.
        auto const &base = colour->mips.front();
        CHECK(base.byte_length == block_count_4x4(base.width, base.height) * 16);
        CHECK(std::ranges::all_of(std::span{colour->data}.subspan(base.byte_offset, base.byte_length),
                                  [](std::byte value) { return value == marker; }));

        auto data = load_compressed_texture(albedo, TextureRole::generic, TextureCache{dir});
        REQUIRE(data.has_value());
        CHECK(data->format == VK_FORMAT_BC7_UNORM_BLOCK);

        auto normal_map = load_compressed_texture(normal, TextureRole::normal_map, TextureCache{dir});
        REQUIRE(normal_map.has_value());
        CHECK(normal_map->format == VK_FORMAT_BC5_UNORM_BLOCK);

        {
            std::scoped_lock const lock{recorder->mutex};
            REQUIRE(recorder->formats.size() == 3);
            CHECK(recorder->formats[0] == GpuBlockFormat::bc7);
            CHECK(recorder->formats[2] == GpuBlockFormat::bc5);
        }

        // What the GPU wrote is what the cache serves next time, with no GPU involved.
        install_gpu_texture_service(nullptr);

        auto again = load_compressed_texture(albedo, TextureRole::colour, TextureCache{dir});
        REQUIRE(again.has_value());
        CHECK(again->data == colour->data);

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("texture pipeline: a GPU service that fails falls back to the CPU encoder") {
        auto const dir = make_temp_dir("fallback");
        ScopedService const installed{std::make_shared<GpuTextureService>(
                [](std::span<GpuTextureLevel const>, GpuBlockFormat) -> GpuTextureService::Outcome {
                    return std::unexpected{std::string{"no queue"}};
                })};

        auto const albedo = AssetPath::external(std::filesystem::path{TEST_ASSETS_DIR} /
                                                "assets/textures/terrain/terrain_albedo.jpg")
                                    .value();

        auto texture = load_compressed_texture(albedo, TextureRole::colour, TextureCache{dir});

        REQUIRE(texture.has_value());
        CHECK(texture->format == VK_FORMAT_BC7_SRGB_BLOCK);
        CHECK_FALSE(validate_compressed_texture(*texture).has_value());

        // Real BC7 from the CPU encoder, not the fake's constant blocks.
        auto const &base = texture->mips.front();
        CHECK_FALSE(std::ranges::all_of(std::span{texture->data}.subspan(base.byte_offset, base.byte_length),
                                        [](std::byte value) { return value == marker; }));

        std::filesystem::remove_all(dir);
    }
}
