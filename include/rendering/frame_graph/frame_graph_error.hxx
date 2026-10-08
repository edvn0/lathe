#pragma once

#include <cstdint>
#include <format>
#include <string>
#include <string_view>

namespace frame_graph {

    enum class FrameGraphErrorType : std::uint8_t {
        duplicate_pass_name,
        invalid_handle,
        stale_version,
        read_before_write,
        write_to_read_only_import,
        wrong_resource_kind,
        missing_shader_stages,
        conflicting_use,
        attachment_in_non_raster_pass,
        raster_pass_with_compute_affinity,
        token_on_compute_pass,
        import_exit_not_on_graphics,
    };

    struct FrameGraphError {
        FrameGraphErrorType type = FrameGraphErrorType::invalid_handle;
        std::string pass;
        std::string resource;
    };

}

template<>
struct std::formatter<frame_graph::FrameGraphErrorType> : std::formatter<std::string_view> {
    constexpr auto format(frame_graph::FrameGraphErrorType error, std::format_context &context) const {
        using enum frame_graph::FrameGraphErrorType;
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case duplicate_pass_name:
                    return "duplicate_pass_name";
                case invalid_handle:
                    return "invalid_handle";
                case stale_version:
                    return "stale_version";
                case read_before_write:
                    return "read_before_write";
                case write_to_read_only_import:
                    return "write_to_read_only_import";
                case wrong_resource_kind:
                    return "wrong_resource_kind";
                case missing_shader_stages:
                    return "missing_shader_stages";
                case conflicting_use:
                    return "conflicting_use";
                case attachment_in_non_raster_pass:
                    return "attachment_in_non_raster_pass";
                case raster_pass_with_compute_affinity:
                    return "raster_pass_with_compute_affinity";
                case token_on_compute_pass:
                    return "token_on_compute_pass";
                case import_exit_not_on_graphics:
                    return "import_exit_not_on_graphics";
            }
            return "unknown_frame_graph_error";
        }();
        return std::formatter<std::string_view>::format(name, context);
    }
};

template<>
struct std::formatter<frame_graph::FrameGraphError> : std::formatter<std::string_view> {
    auto format(frame_graph::FrameGraphError const &error, std::format_context &context) const {
        auto const text = std::format("{} (pass '{}', resource '{}')", error.type, error.pass, error.resource);
        return std::formatter<std::string_view>::format(text, context);
    }
};
