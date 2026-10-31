#include "app/cache_menu.hxx"

#include <imgui.h>

#include <format>
#include <string_view>

#include <spawn.h>
#include <unistd.h>

#include "core/error_describe.hxx"
#include "core/human_readable_bytes.hxx"
#include "core/logger.hxx"
#include "storage/caches.hxx"

extern char **environ; // NOLINT(readability-redundant-declaration)

namespace {
    constexpr auto clear_all_popup = "Clear the whole cache?";

    auto describe_reclaimed(std::string_view what, CacheReclaimed const &reclaimed) -> std::string {
        return std::format("{}: {} entries, {}", what, reclaimed.entries, human_readable_bytes(reclaimed.bytes));
    }

    auto open_folder(std::filesystem::path const &directory) -> bool {
        auto const path = directory.string();
        char const *argv[] = {"xdg-open", path.c_str(), nullptr};
        pid_t child = 0;

        return posix_spawnp(&child, "xdg-open", nullptr, nullptr, const_cast<char *const *>(argv), environ) == 0;
    }
}

auto CacheMenu::refresh() -> void {
    stats_.reset();

    if (auto const store = caches::store()) {
        if (auto stats = store->stats()) {
            stats_ = std::move(*stats);
        } else {
            status_ = describe(stats.error());
        }
    }
}

auto CacheMenu::draw_popups() -> void {
    if (confirm_clear_all_) {
        confirm_clear_all_ = false;
        ImGui::OpenPopup(clear_all_popup);
    }

    if (!ImGui::BeginPopupModal(clear_all_popup, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        return;
    }

    ImGui::TextUnformatted("This deletes every cached texture, shader binary and shader pack.");
    ImGui::TextUnformatted("They are rebuilt on demand, so the next loads are slow.");
    ImGui::Spacing();

    if (ImGui::Button("Clear everything")) {
        status_ = describe_reclaimed("Cleared the cache", caches::clear_all());
        info("{}", status_);
        refresh();
        ImGui::CloseCurrentPopup();
    }

    ImGui::SameLine();

    if (ImGui::Button("Cancel")) {
        ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
}

auto CacheMenu::draw() -> void {
    if (!ImGui::BeginMenu("Advanced")) {
        return;
    }

    if (ImGui::BeginMenu("Cache")) {
        auto const store = caches::store();

        if (store == nullptr) {
            ImGui::TextDisabled("Caching is off");
            ImGui::EndMenu();
            ImGui::EndMenu();
            return;
        }

        // Counting entries reads the whole index, so it is done once when the menu opens, not every frame.
        if (ImGui::IsWindowAppearing() || !stats_.has_value()) {
            refresh();
        }

        if (stats_.has_value()) {
            ImGui::TextDisabled("%s", std::format("{} entries, {} (index {})", stats_->total_entries(),
                                                  human_readable_bytes(stats_->total_bytes()),
                                                  human_readable_bytes(stats_->database_bytes))
                                              .c_str());
            ImGui::Separator();

            for (auto const &entry: stats_->namespaces) {
                auto const label = std::format("Clear {} ({} entries, {})", entry.name, entry.entries,
                                               human_readable_bytes(entry.bytes));

                if (ImGui::MenuItem(label.c_str(), nullptr, false, entry.entries != 0)) {
                    status_ = describe_reclaimed(std::format("Cleared {}", entry.name), store->clear(entry.name));
                    info("{}", status_);
                    refresh();
                }
            }

            ImGui::Separator();
        }

        if (ImGui::MenuItem("Remove orphaned files")) {
            status_ = describe_reclaimed("Removed orphans", store->remove_orphans());
            info("{}", status_);
            refresh();
        }

        if (ImGui::MenuItem("Prune entries whose source is gone")) {
            CacheReclaimed total;

            if (stats_.has_value()) {
                for (auto const &entry: stats_->namespaces) {
                    if (entry.kind == CacheKind::file) {
                        auto const reclaimed = store->prune_missing_sources(entry.name);
                        total.entries += reclaimed.entries;
                        total.bytes += reclaimed.bytes;
                    }
                }
            }

            status_ = describe_reclaimed("Pruned", total);
            info("{}", status_);
            refresh();
        }

        if (ImGui::MenuItem("Open cache folder")) {
            if (!open_folder(store->directory())) {
                status_ = "Could not start xdg-open";
            }
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Clear everything...")) {
            confirm_clear_all_ = true;
        }

        if (!status_.empty()) {
            ImGui::Separator();
            ImGui::TextDisabled("%s", status_.c_str());
        }

        ImGui::EndMenu();
    }

    ImGui::EndMenu();
}
