// describe() overloads for the rendering module's error types.
#include "core/error_describe.hxx"

#include <format>

#include "rendering/forward_target.hxx"
#include "rendering/pipeline_graph_repository.hxx"

auto describe(PipelineGraphError const &error) -> std::string {
    auto head = std::format("PipelineGraphError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}

auto describe(ForwardTargetError const &error) -> std::string {
    auto head = std::format("ForwardTargetError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}
