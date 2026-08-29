#include "model_streamer.hxx"

#include <chrono>

#include "logger.hxx"
#include "renderer.hxx"

auto ModelStreamer::request(Renderer &renderer, std::filesystem::path source_path, ModelHandle fallback,
                            std::string debug_name) -> ModelHandle {
    auto pending_handle = renderer.create_pending_model(fallback);

    if (!pending_handle) {
        warn("model_streamer: could not reserve a slot for '{}', staying on its fallback model", debug_name);
        return fallback;
    }

    auto future = renderer.load_model_cpu_async(std::move(source_path));

    pending_.push_back(PendingRequest{
            .handle = *pending_handle,
            .debug_name = std::move(debug_name),
            .future = std::move(future),
    });

    return *pending_handle;
}

auto ModelStreamer::process_ready(Renderer &renderer, VkCommandBuffer command_buffer) -> void {
    ZoneScopedNC("ProcessReadyModels", tracy::Color::Goldenrod);

    using namespace std::chrono_literals;

    std::erase_if(pending_, [&](PendingRequest &request) {
        if (request.future.wait_for(0s) != std::future_status::ready) {
            return false;
        }

        auto cpu_data = request.future.get();

        if (!cpu_data) {
            error("model_streamer: '{}' failed to load ({}); staying on its fallback model", request.debug_name,
                  cpu_data.error().type);

            return true;
        }

        auto finished = renderer.finish_model_load(request.handle, *cpu_data, command_buffer);

        if (!finished) {
            error("model_streamer: '{}' failed to finish loading ({}); staying on its fallback model",
                  request.debug_name, finished.error().type);

            return true;
        }

        debug("model_streamer: '{}' uploaded to GPU", request.debug_name);

        return true;
    });
}

auto ModelStreamer::wait_all() -> void {
    for (auto &request: pending_) {
        request.future.wait();
    }

    pending_.clear();
}
