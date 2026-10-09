#include "rendering/frame_graph/resource_info.hxx"

#include <algorithm>

#include "gpu/image.hxx"

namespace frame_graph {

    auto previewable(ResourceDesc const &resource) -> bool {
        return resource.kind == ResourceKind::image && !resource.imported && resource.transient_image &&
               resource.transient_image->samples == VK_SAMPLE_COUNT_1_BIT &&
               has_image_descriptor_view(resource.transient_image->descriptor_views, ImageDescriptorView::sampled_2d);
    }

    auto describe_resource(FrameGraphView const &view, std::uint32_t resource) -> ResourceInfo {
        auto info = ResourceInfo{};

        for (auto pass = std::uint32_t{0}; pass < view.graph.passes.size(); ++pass) {
            for (auto const &access: view.graph.passes[pass].accesses) {
                if (access.resource != resource) {
                    continue;
                }

                (access.produces ? info.producers : info.consumers).push_back(pass);
                info.first_pass = info.first_pass ? std::min(*info.first_pass, pass) : pass;
                info.last_pass = info.last_pass ? std::max(*info.last_pass, pass) : pass;
            }
        }

        auto const *placement = view.transients.placement_of(resource);
        if (placement == nullptr) {
            return info;
        }

        info.placement = *placement;
        for (auto const &other: view.transients.placements) {
            auto const overlaps = other.block == placement->block && other.offset < placement->offset + placement->size &&
                                  placement->offset < other.offset + other.size;
            if (other.resource != resource && overlaps) {
                info.shares_memory_with.push_back(other.resource);
            }
        }

        return info;
    }

}
