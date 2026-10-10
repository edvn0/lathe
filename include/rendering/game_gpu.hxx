#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "core/renderer_error.hxx"
#include "gpu/buffer.hxx"
#include "rendering/game_graph.hxx"

struct Renderer;

struct GameComputeShaderInfo {
    // Relative to the data directory, e.g. "assets/shaders/game/my_pass.slang".
    std::string source;
    std::string entry_point = "main_cs";
    std::string debug_name;
};

struct GameGraphicsShaderInfo {
    std::string source;
    std::string vertex_entry_point = "main_vs";
    std::string fragment_entry_point = "main_fs";
    // Alpha-blended, depth-tested, no depth writes: for transparent draws.
    bool blending = false;
    std::string debug_name;
};

// What a game may ask of the GPU: shaders that run through the game graph and buffers that survive between frames.
// It replaces reaching for Renderer::context() and the bindless table, which hand out the whole device.
class GameGpu {
public:
    explicit GameGpu(Renderer &renderer) noexcept : renderer_{&renderer} {}

    [[nodiscard]] auto register_compute(GameComputeShaderInfo const &info)
            -> std::expected<GameComputeShader, RendererError>;
    [[nodiscard]] auto register_graphics(GameGraphicsShaderInfo const &info)
            -> std::expected<GameGraphicsShader, RendererError>;

    // The engine side of GameGraph::create_buffer and persistent_buffer. A game never calls these: it names a buffer
    // through the graph and the graph asks for it here, so no Buffer or VkBuffer ever reaches game code.

    // Called once per record_frame, before the game declares anything. Everything `frame_slot` acquired two frames ago
    // is known to be finished with, so unused frame buffers are freed and the rest become available again; retired
    // persistent buffers are destroyed once every frame that could have used them is done.
    auto begin_frame(std::uint32_t frame_slot) -> void;

    // A device-local storage buffer with an address that only `frame_slot` uses, valid until the slot's next
    // begin_frame. Contents are whatever the buffer held before. At least `size` bytes.
    [[nodiscard]] auto acquire_frame_buffer(std::uint32_t frame_slot, VkDeviceSize size, std::string_view name)
            -> std::expected<Buffer const *, RendererError>;

    // The buffer of this name, shared by all frames in flight and zero-filled when created. Asking for another size
    // recreates it (zero-filled again) and retires the old one until the frames in flight are done with it.
    [[nodiscard]] auto acquire_persistent_buffer(std::string_view name, VkDeviceSize size)
            -> std::expected<Buffer const *, RendererError>;

    // Forgets a persistent buffer that nothing will ask for again. It is destroyed once the frames in flight are done
    // with it; asking for the name later creates a fresh zero-filled buffer.
    auto release_persistent_buffer(std::string_view name) -> void;

    [[nodiscard]] auto persistent_buffer_count() const noexcept -> std::size_t { return persistent_.size(); }
    [[nodiscard]] auto retired_buffer_count() const noexcept -> std::size_t { return retired_.size(); }
    [[nodiscard]] auto frame_buffer_count() const noexcept -> std::size_t;

    [[nodiscard]] auto memory() noexcept -> GameGraphMemory & { return memory_; }

private:
    struct FrameBuffer {
        Buffer buffer;
        bool used = false;
    };

    struct Retired {
        Buffer buffer;
        std::uint32_t frames_left = 0;
    };

    auto retire(Buffer buffer) -> void;
    [[nodiscard]] auto create_storage_buffer(VkDeviceSize size, std::string_view name, bool zero)
            -> std::expected<Buffer, RendererError>;

    Renderer *renderer_;
    std::vector<std::vector<FrameBuffer>> frame_buffers_;
    std::map<std::string, Buffer, std::less<>> persistent_;
    std::vector<Retired> retired_;
    GameGraphMemory memory_;
};
