#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rendering/frame_graph/frame_graph_error.hxx"
#include "rendering/frame_graph/physical.hxx"
#include "rendering/frame_graph/types.hxx"
#include "rendering/frame_graph/use_table.hxx"

namespace frame_graph {

    struct PassContext;
    using RecordFn = std::move_only_function<void(PassContext &)>;

    struct TransientImageDesc {
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent3D extent{};
        std::uint32_t mip_levels = 1;
        std::uint32_t array_layers = 1;
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
        std::string_view debug_name;
    };

    struct ImportedDesc {
        ResourceState entry; // state the graph may assume at frame start
        ResourceState exit; // state the graph must leave; exit.queue must be graphics
        Sharing sharing = Sharing::exclusive;
        bool read_only = false;
        bool swapchain = false; // touching it roots the pass; the present transition is the epilogue
        std::string_view debug_name;
        // Handles for an imported image or buffer; the executor needs them to translate barriers.
        PhysicalImage image{};
        PhysicalBuffer buffer{};
    };

    // Where a pass runs and how it appears in profiles. `name_id` is the stable benchmark id, `label` the UI text.
    struct PassProfile {
        std::string_view name_id;
        std::string_view label;
        std::uint32_t color = 0;
    };

    enum class ResourceKind : std::uint8_t { image, buffer, token };

    struct ResourceDesc {
        std::string name;
        ResourceKind kind = ResourceKind::image;
        bool imported = false;
        bool swapchain = false;
        bool read_only = false;
        Sharing sharing = Sharing::exclusive;
        ResourceState entry;
        ResourceState exit;
        std::optional<TransientImageDesc> transient_image;
        PhysicalImage image{};
        PhysicalBuffer buffer{};
    };

    struct AccessDesc {
        std::uint32_t resource = 0;
        std::uint32_t version = 0; // the version consumed (read, or written over)
        Use use = Use::sampled;
        ShaderStages stages = 0;
        bool discard = false; // the previous contents are not needed
        bool produces = false; // writes the next version
        // The pass leaves the whole resource in this use's layout and scope instead of the one it entered with.
        std::optional<Use> exit_use;
    };

    struct PassDesc {
        std::string name;
        PassType type = PassType::compute;
        QueueAffinity affinity = QueueAffinity::graphics;
        PassProfile profile;
        std::vector<AccessDesc> accesses;
        bool side_effect = false;
        bool legacy = false;
        bool pinned = false;
    };

    struct GraphDesc {
        std::vector<ResourceDesc> resources;
        std::vector<PassDesc> passes;
        // For each resource, the latest version and, per version, the pass that produced it (-1 for imports and
        // unwritten transients). producers[resource][version] where version is 1-based, index 0 unused.
        std::vector<std::vector<std::int64_t>> producers;
    };

    // For a pass that manages subresources itself: it enters in one use and leaves the resource in another.
    struct ExitUse {
        Use use = Use::sampled;
    };

    class FrameGraph;

    // Declares one pass's accesses. Handles are versioned: a write returns the next version, and using anything but the
    // latest version is a stale_version error reported by compile().
    class PassBuilder {
    public:
        auto queue(QueueAffinity affinity) -> void;
        auto side_effect() -> void;
        auto legacy() -> void;
        auto pinned() -> void;

        [[nodiscard]] auto read(ImageId image, Use use, ShaderStages stages = 0) -> ImageId;
        [[nodiscard]] auto write(ImageId image, Use use, ShaderStages stages = 0) -> ImageId;
        [[nodiscard]] auto read(BufferId buffer, Use use, ShaderStages stages = 0) -> BufferId;
        // Enters as `use` and leaves the image as `exit.use` (e.g. bloom writes it as storage and leaves it sampled).
        [[nodiscard]] auto write(ImageId image, Use use, ShaderStages stages, ExitUse exit) -> ImageId;
        [[nodiscard]] auto write(BufferId buffer, Use use, ShaderStages stages = 0) -> BufferId;
        // A buffer write that overwrites the whole buffer, so earlier contents are not needed.
        [[nodiscard]] auto write_discard(BufferId buffer, Use use, ShaderStages stages = 0) -> BufferId;
        [[nodiscard]] auto color(ImageId image, LoadOp load, StoreOp store) -> ImageId;
        [[nodiscard]] auto write_depth(ImageId image, LoadOp load, StoreOp store) -> ImageId;
        [[nodiscard]] auto create(TransientImageDesc const &desc) -> ImageId;

    private:
        friend class FrameGraph;
        PassBuilder(FrameGraph &graph, PassDesc &pass) : graph_(&graph), pass_(&pass) {}

        auto access(std::uint32_t resource, std::uint32_t version, Use use, ShaderStages stages, bool discard) -> void;

        FrameGraph *graph_;
        PassDesc *pass_;
    };

    class FrameGraph {
    public:
        auto reset() -> void;

        [[nodiscard]] auto import_image(ImportedDesc const &desc) -> ImageId;
        [[nodiscard]] auto import_buffer(ImportedDesc const &desc) -> BufferId;
        [[nodiscard]] auto import_token(std::string_view name, ResourceState entry, ResourceState exit) -> BufferId;

        // `setup` receives a PassBuilder, declares the pass's accesses, and returns the record function.
        template<std::invocable<PassBuilder &> Setup>
        auto add_pass(std::string_view name, PassType type, PassProfile profile, Setup &&setup) -> PassId {
            auto &pass = begin_pass(name, type, profile);
            auto builder = PassBuilder{*this, pass};
            records_.push_back(std::forward<Setup>(setup)(builder));
            return PassId{.index = static_cast<std::uint32_t>(desc_.passes.size() - 1), .generation = 1};
        }

        [[nodiscard]] auto description() const -> GraphDesc const & { return desc_; }
        [[nodiscard]] auto declaration_errors() const -> std::vector<FrameGraphError> const & { return errors_; }

        // The record lambdas, in pass declaration order. The compiler never calls them; the executor does.
        [[nodiscard]] auto records() -> std::span<RecordFn> { return records_; }

    private:
        friend class PassBuilder;

        auto begin_pass(std::string_view name, PassType type, PassProfile profile) -> PassDesc &;
        auto add_resource(ResourceDesc resource, bool produced) -> std::uint32_t;
        // The latest version of `resource`; 0 when it is a transient that nothing has written.
        [[nodiscard]] auto latest_version(std::uint32_t resource) const -> std::uint32_t;
        auto record_error(FrameGraphErrorType type, PassDesc const &pass, std::uint32_t resource) -> void;
        // Checks a handle's slot and version against the latest, recording the error that applies.
        auto validate_access(PassDesc const &pass, std::uint32_t resource, std::uint32_t version) -> bool;
        auto produce(std::uint32_t resource) -> std::uint32_t;

        GraphDesc desc_;
        std::vector<RecordFn> records_;
        std::vector<std::uint32_t> latest_;
        std::vector<bool> written_;
        std::vector<FrameGraphError> errors_;
    };

} // namespace frame_graph
