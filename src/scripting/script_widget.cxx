#include "scripting/script_widget.hxx"

#include <imgui.h>

#include <cfloat>
#include <chrono>
#include <cstddef>
#include <format>
#include <string>
#include <string_view>
#include <utility>

#include "core/human_readable_bytes.hxx"
#include "rendering/toast.hxx"

namespace gui {
    namespace {
        constexpr std::string_view example_script = "e = scene.get_entity(\"Helmets\")\n"
                                                    "for _, c in ipairs(e.get_children_or_empty()) do\n"
                                                    "    c.get_transform().translation = Vec3.random(-30, 30)\n"
                                                    "end\n";

        constexpr char const *api_help = "scene.get_entity(name)       entity by name (duplicates: lowest entity id)\n"
                                         "entity.get_children_or_empty()  list of child entities\n"
                                         "entity.get_transform()       the entity's transform\n"
                                         "transform.translation        read: a Vec3 copy; write: a whole Vec3\n"
                                         "Vec3(x, y, z), Vec3.random(lo, hi), + - *, .x .y .z\n"
                                         "print(...)                   to the Console\n\n"
                                         "Globals don't persist between runs. Writes made before an error are kept.";

        constexpr std::size_t bytes_per_mib = 1024ULL * 1024;

        // imgui_stdlib.cpp's approach: let ImGui grow the std::string it edits in place.
        auto resize_callback(ImGuiInputTextCallbackData *data) -> int {
            if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
                auto *const text = static_cast<std::string *>(data->UserData);
                text->resize(static_cast<std::size_t>(data->BufTextLen));
                data->Buf = text->data();
            }
            return 0;
        }

        auto input_text_multiline(char const *label, std::string &text, ImVec2 size,
                                  ImGuiInputTextFlags flags) -> bool {
            return ImGui::InputTextMultiline(label, text.data(), text.capacity() + 1, size,
                                             flags | ImGuiInputTextFlags_CallbackResize, &resize_callback, &text);
        }

        [[nodiscard]] auto error_status(ScriptError const &error) -> std::string {
            if (error.line.has_value()) {
                return std::format("{} at line {}: {}", script_error_kind_name(error.kind), *error.line, error.message);
            }
            return std::format("{}: {}", script_error_kind_name(error.kind), error.message);
        }
    } // namespace

    ScriptWidget::ScriptWidget(ScriptEngineSettings settings) : settings_(settings), source_(example_script) {}

    auto ScriptWidget::draw(std::optional<ScriptWorld> world) -> ScriptWidgetFrame {
        bool const can_run = world.has_value();

        // Toolbar.
        ImGui::BeginDisabled(!can_run);
        bool const run_clicked = ImGui::Button("Run");
        ImGui::EndDisabled();

        ImGui::SameLine();
        if (ImGui::Button("Reset VM")) {
            engine_.reset();
            last_error_.reset();
        }
        ImGui::SetItemTooltip("Discards the Lua state and everything a script left in it.");

        ImGui::SameLine();
        ImGui::TextDisabled("Ctrl+Enter");
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        ImGui::SetItemTooltip("%s", api_help);

        if (!can_run) {
            ImGui::TextDisabled("Stop play mode to run scripts on the editor scene.");
        }

        if (ImGui::CollapsingHeader("Limits")) {
            auto timeout_ms = static_cast<int>(settings_.timeout.count());
            if (ImGui::SliderInt("Timeout (ms)", &timeout_ms, 10, 5000)) {
                settings_.timeout = std::chrono::milliseconds{timeout_ms};
                if (engine_.has_value()) {
                    engine_->set_timeout(settings_.timeout);
                }
            }

            auto memory_mib = static_cast<int>(settings_.memory_limit_bytes / bytes_per_mib);
            if (ImGui::SliderInt("Memory (MiB)", &memory_mib, 8, 512)) {
                settings_.memory_limit_bytes = static_cast<std::size_t>(memory_mib) * bytes_per_mib;
                if (engine_.has_value()) {
                    engine_->set_memory_limit(settings_.memory_limit_bytes);
                }
            }
        }

        // Editor. Ctrl+Enter validates a multiline field (plain Enter inserts a newline), which
        // EnterReturnsTrue reports as `submitted`.
        bool const submitted = input_text_multiline(
                "##source", source_, ImVec2{-FLT_MIN, -ImGui::GetTextLineHeightWithSpacing() * 3.0F},
                ImGuiInputTextFlags_AllowTabInput | ImGuiInputTextFlags_EnterReturnsTrue);
        bool const chord = ImGui::IsItemFocused() && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Enter);
        // One request, so the button and Ctrl+Enter in the same frame run once.
        bool const run_requested = run_clicked || submitted || chord;

        // Status.
        if (last_error_.has_value()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0F, 0.4F, 0.4F, 1.0F});
            ImGui::TextWrapped("%s", error_status(*last_error_).c_str());
            ImGui::PopStyleColor();
        } else if (last_duration_.has_value()) {
            ImGui::TextUnformatted("OK");
        }

        if (last_duration_.has_value()) {
            auto const milliseconds = std::chrono::duration<double, std::milli>{*last_duration_}.count();
            ImGui::Text("Last run: %.2f ms, %u transforms", milliseconds, last_transforms_written_);
        }
        if (engine_.has_value()) {
            ImGui::SameLine();
            ImGui::TextDisabled("Lua heap: %s", human_readable_bytes(engine_->stats().lua_bytes_in_use).c_str());
        }

        ScriptWidgetFrame frame{};
        if (run_requested && world.has_value()) {
            frame = run(*world);
        }
        return frame;
    }

    auto ScriptWidget::run(ScriptWorld world) -> ScriptWidgetFrame {
        if (!engine_.has_value()) {
            auto created = ScriptEngine::create(settings_);
            if (!created.has_value()) {
                last_error_ = created.error();
                report_error(created.error());
                return {};
            }
            engine_.emplace(std::move(*created));
        }

        auto const report = engine_->run(source_, world);
        last_error_ = report.error;
        last_duration_ = report.duration;
        last_transforms_written_ = report.transforms_written;
        if (report.error.has_value()) {
            report_error(*report.error);
        }

        return {.ran = true, .transforms_written = report.transforms_written};
    }

    // A successful run only updates the inline status, so repeated Ctrl+Enter doesn't flood the corner.
    auto ScriptWidget::report_error(ScriptError const &error) -> void {
        auto const message = describe(error);
        switch (error.kind) {
            case ScriptErrorKind::syntax:
                toast_error("Script syntax error", message);
                break;
            case ScriptErrorKind::runtime:
                toast_error("Script error", message);
                break;
            case ScriptErrorKind::timeout:
                toast_error("Script timed out", message);
                break;
            case ScriptErrorKind::memory:
                toast_error("Script out of memory", message);
                break;
            case ScriptErrorKind::invalid_entity:
                toast_error("Script: invalid entity", message);
                break;
            case ScriptErrorKind::busy:
                toast_warn("Script busy", message);
                break;
        }
    }
} // namespace gui
