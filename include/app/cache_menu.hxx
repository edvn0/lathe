#pragma once

#include <optional>
#include <string>

#include "storage/cache_store.hxx"

// Debug > Advanced > Cache: what the on-disk cache store holds, and buttons to empty or tidy it. Call draw() while the
// Debug menu is open.
class CacheMenu {
public:
    auto draw() -> void;

    // The confirmation for "Clear everything"; call outside any menu, every frame.
    auto draw_popups() -> void;

private:
    auto refresh() -> void;

    std::optional<CacheStats> stats_;
    std::string status_;
    bool confirm_clear_all_ = false;
};
