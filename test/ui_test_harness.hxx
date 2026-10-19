#pragma once

#include <string>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>

// A Dear ImGui context with no window system and no renderer, so UI code can be run frame by frame in a test. ImGui's
// asserts still fire (build the tests in Debug to have them), and the mismatches it recovers from (a missing End,
// an unbalanced Push or Pop) are collected in `errors` instead of showing a red tooltip.
class UiTestHarness {
public:
    UiTestHarness() {
        context_ = ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.DisplaySize = ImVec2{1280.0F, 720.0F};
        io.ConfigErrorRecoveryEnableAssert = false;
        io.ConfigErrorRecoveryEnableTooltip = false;
        context_->ErrorCallback = [](ImGuiContext *, void *user_data, char const *message) {
            static_cast<std::vector<std::string> *>(user_data)->emplace_back(message);
        };
        context_->ErrorCallbackUserData = &errors;

        unsigned char *pixels = nullptr;
        int width = 0;
        int height = 0;
        io.Fonts->AddFontDefault();
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    }

    ~UiTestHarness() { ImGui::DestroyContext(context_); }

    UiTestHarness(UiTestHarness const &) = delete;
    auto operator=(UiTestHarness const &) -> UiTestHarness & = delete;

    // One frame: `body` runs inside a window named "test".
    template<typename Body>
    auto frame(Body &&body) -> void {
        ImGui::GetIO().DeltaTime = 1.0F / 60.0F;
        ImGui::NewFrame();
        ImGui::SetNextWindowSize(ImVec2{600.0F, 500.0F});
        ImGui::Begin("test");
        body();
        ImGui::End();
        ImGui::Render();
    }

    // One frame with `body` inside an open popup, as a combo or a menu draws its list.
    template<typename Body>
    auto popup_frame(Body &&body) -> void {
        frame([&] {
            if (first_popup_frame_) {
                ImGui::OpenPopup("popup");
                first_popup_frame_ = false;
            }
            if (ImGui::BeginPopup("popup")) {
                body();
                ImGui::EndPopup();
            }
        });
    }

    std::vector<std::string> errors;

private:
    ImGuiContext *context_ = nullptr;
    bool first_popup_frame_ = true;
};
