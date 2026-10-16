// The installed game: a bare Runtime -- no editor, no benchmark runner. It always runs as a packaged game would
// (fullscreen play, no way back into an editor); startup and the frame loop are run_engine, shared with the editor
// (src/main.cxx).

#include <memory>
#include <string_view>

#include "runtime/engine_main.hxx"
#include "runtime/runtime.hxx"

namespace {

    class PlayerProfile final : public EngineProfile {
    public:
        [[nodiscard]] auto name() const -> std::string_view override { return "lathe-player"; }
        [[nodiscard]] auto description() const -> std::string_view override { return "Lathe game player"; }

        [[nodiscard]] auto always_player() const -> bool override { return true; }

        [[nodiscard]] auto create_runtime(VulkanContext &context) -> std::unique_ptr<Runtime> override {
            return std::make_unique<Runtime>(context);
        }
    };

}

auto main(int argc, char **argv) -> int {
    PlayerProfile profile;

    return run_engine(argc, argv, profile);
}
