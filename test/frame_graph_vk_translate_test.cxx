#include <doctest/doctest.h>

#include "rendering/frame_graph/vk_translate.hxx"

using namespace frame_graph;

namespace {

    // Fake handles: the translation only copies them.
    template<typename Handle>
    auto handle(std::uintptr_t value) -> Handle {
        return reinterpret_cast<Handle>(value); // NOLINT(performance-no-int-to-ptr)
    }

    auto make_resources() -> PhysicalResources {
        auto resources = PhysicalResources{};
        resources.images.resize(4);
        resources.buffers.resize(4);
        resources.images[0] = PhysicalImage{
                .image = handle<VkImage>(0x10),
                .view = handle<VkImageView>(0x11),
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .extent = {64, 32, 1},
                .mip_levels = 5,
                .array_layers = 2,
        };
        resources.images[1] = PhysicalImage{
                .image = handle<VkImage>(0x20),
                .view = handle<VkImageView>(0x21),
                .format = VK_FORMAT_D32_SFLOAT,
                .extent = {64, 32, 1},
        };
        resources.images[2] = PhysicalImage{
                .image = handle<VkImage>(0x30),
                .view = handle<VkImageView>(0x31),
                .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                .extent = {64, 32, 1},
        };
        resources.buffers[3] = PhysicalBuffer{.buffer = handle<VkBuffer>(0x40), .address = 0x1000, .size = 4096};
        return resources;
    }

} // namespace

TEST_SUITE("unit") {
    TEST_CASE("an image barrier keeps its scopes and layouts and covers every mip and layer") {
        auto barriers = BarrierSet{};
        barriers.images.push_back(ImageBarrier{
                .resource = 0,
                .src_stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .src_access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .dst_stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                .dst_access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                .old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        });

        auto const storage = translate(barriers, make_resources());
        REQUIRE(storage.has_value());
        REQUIRE(storage->images.size() == 1);

        auto const &barrier = storage->images.front();
        CHECK(barrier.sType == VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2);
        CHECK(barrier.srcStageMask == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
        CHECK(barrier.srcAccessMask == VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        CHECK(barrier.dstStageMask == VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
        CHECK(barrier.dstAccessMask == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        CHECK(barrier.oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        CHECK(barrier.newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        CHECK(barrier.image == handle<VkImage>(0x10));
        CHECK(barrier.subresourceRange.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT);
        CHECK(barrier.subresourceRange.baseMipLevel == 0);
        CHECK(barrier.subresourceRange.levelCount == 5);
        CHECK(barrier.subresourceRange.baseArrayLayer == 0);
        CHECK(barrier.subresourceRange.layerCount == 2);
    }

    TEST_CASE("queue family indices are IGNORED unless the barrier is an ownership transfer") {
        auto barriers = BarrierSet{};
        barriers.images.push_back(ImageBarrier{.resource = 0});
        barriers.buffers.push_back(BufferBarrier{.resource = 3});
        barriers.images.push_back(ImageBarrier{
                .resource = 2,
                .src_family = 0,
                .dst_family = 2,
                .op = OwnershipOp::release,
        });
        barriers.buffers.push_back(BufferBarrier{
                .resource = 3,
                .src_family = 2,
                .dst_family = 0,
                .op = OwnershipOp::acquire,
        });

        auto const storage = translate(barriers, make_resources());
        REQUIRE(storage.has_value());
        REQUIRE(storage->images.size() == 2);
        REQUIRE(storage->buffers.size() == 2);

        CHECK(storage->images[0].srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
        CHECK(storage->images[0].dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
        CHECK(storage->buffers[0].srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
        CHECK(storage->buffers[0].dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);

        CHECK(storage->images[1].srcQueueFamilyIndex == 0);
        CHECK(storage->images[1].dstQueueFamilyIndex == 2);
        CHECK(storage->buffers[1].srcQueueFamilyIndex == 2);
        CHECK(storage->buffers[1].dstQueueFamilyIndex == 0);
    }

    TEST_CASE("buffer barriers are always the whole buffer") {
        auto barriers = BarrierSet{};
        barriers.buffers.push_back(BufferBarrier{
                .resource = 3,
                .src_stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .src_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .dst_stages = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                .dst_access = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
        });

        auto const storage = translate(barriers, make_resources());
        REQUIRE(storage.has_value());
        REQUIRE(storage->buffers.size() == 1);
        auto const &barrier = storage->buffers.front();
        CHECK(barrier.sType == VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2);
        CHECK(barrier.buffer == handle<VkBuffer>(0x40));
        CHECK(barrier.offset == 0);
        CHECK(barrier.size == VK_WHOLE_SIZE);
        CHECK(barrier.srcStageMask == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        CHECK(barrier.dstAccessMask == VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    }

    TEST_CASE("memory barriers translate directly") {
        auto barriers = BarrierSet{};
        barriers.memory.push_back(MemoryBarrier{
                .src_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .src_access = VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dst_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .dst_access = VK_ACCESS_2_MEMORY_READ_BIT,
        });
        auto const storage = translate(barriers, PhysicalResources{});
        REQUIRE(storage.has_value());
        REQUIRE(storage->memory.size() == 1);
        CHECK(storage->memory.front().sType == VK_STRUCTURE_TYPE_MEMORY_BARRIER_2);
        CHECK(storage->memory.front().srcAccessMask == VK_ACCESS_2_MEMORY_WRITE_BIT);
        CHECK(storage->memory.front().dstAccessMask == VK_ACCESS_2_MEMORY_READ_BIT);
    }

    TEST_CASE("the dependency info points at the stored arrays") {
        auto barriers = BarrierSet{};
        barriers.memory.push_back(MemoryBarrier{});
        barriers.images.push_back(ImageBarrier{.resource = 0});
        barriers.images.push_back(ImageBarrier{.resource = 1});
        barriers.buffers.push_back(BufferBarrier{.resource = 3});

        auto const storage = translate(barriers, make_resources());
        REQUIRE(storage.has_value());
        auto const info = storage->info();
        CHECK(info.sType == VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
        CHECK(info.memoryBarrierCount == 1);
        CHECK(info.imageMemoryBarrierCount == 2);
        CHECK(info.bufferMemoryBarrierCount == 1);
        CHECK(info.pMemoryBarriers == storage->memory.data());
        CHECK(info.pImageMemoryBarriers == storage->images.data());
        CHECK(info.pBufferMemoryBarriers == storage->buffers.data());
        CHECK_FALSE(storage->empty());
        CHECK(translate(BarrierSet{}, make_resources())->empty());
    }

    TEST_CASE("the aspect follows the format") {
        CHECK(image_aspect(VK_FORMAT_R8G8B8A8_UNORM) == VK_IMAGE_ASPECT_COLOR_BIT);
        CHECK(image_aspect(VK_FORMAT_R16G16B16A16_SFLOAT) == VK_IMAGE_ASPECT_COLOR_BIT);
        CHECK(image_aspect(VK_FORMAT_D32_SFLOAT) == VK_IMAGE_ASPECT_DEPTH_BIT);
        CHECK(image_aspect(VK_FORMAT_D16_UNORM) == VK_IMAGE_ASPECT_DEPTH_BIT);
        CHECK(image_aspect(VK_FORMAT_D24_UNORM_S8_UINT) == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));
        CHECK(image_aspect(VK_FORMAT_D32_SFLOAT_S8_UINT) == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));
        CHECK(image_aspect(VK_FORMAT_S8_UINT) == VK_IMAGE_ASPECT_STENCIL_BIT);

        auto barriers = BarrierSet{};
        barriers.images.push_back(ImageBarrier{.resource = 1});
        auto const storage = translate(barriers, make_resources());
        REQUIRE(storage.has_value());
        CHECK(storage->images.front().subresourceRange.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    TEST_CASE("a barrier on a resource with no handle is a failure that names it") {
        auto barriers = BarrierSet{};
        barriers.images.push_back(ImageBarrier{.resource = 3}); // slot 3 only holds a buffer
        auto const image = translate(barriers, make_resources());
        REQUIRE_FALSE(image.has_value());
        CHECK(image.error().kind == TranslateFailureKind::missing_image);
        CHECK(image.error().resource == 3);

        auto buffers = BarrierSet{};
        buffers.buffers.push_back(BufferBarrier{.resource = 0}); // slot 0 only holds an image
        auto const buffer = translate(buffers, make_resources());
        REQUIRE_FALSE(buffer.has_value());
        CHECK(buffer.error().kind == TranslateFailureKind::missing_buffer);

        auto out_of_range = BarrierSet{};
        out_of_range.images.push_back(ImageBarrier{.resource = 99});
        CHECK_FALSE(translate(out_of_range, make_resources()).has_value());
    }

    TEST_CASE("physical resources are collected from the imports by slot") {
        auto graph = FrameGraph{};
        [[maybe_unused]] auto const image = graph.import_image({
                .debug_name = "image",
                .image = PhysicalImage{.image = handle<VkImage>(0x50), .format = VK_FORMAT_R8_UNORM},
        });
        [[maybe_unused]] auto const buffer = graph.import_buffer({
                .debug_name = "buffer",
                .buffer = PhysicalBuffer{.buffer = handle<VkBuffer>(0x60), .size = 256},
        });

        auto const resources = physical_resources_of(graph.description());
        REQUIRE(resources.images.size() == 2);
        REQUIRE(resources.buffers.size() == 2);
        REQUIRE(resources.image(0) != nullptr);
        CHECK(resources.image(0)->image == handle<VkImage>(0x50));
        CHECK(resources.image(1) == nullptr);
        REQUIRE(resources.buffer(1) != nullptr);
        CHECK(resources.buffer(1)->size == 256);
        CHECK(resources.buffer(0) == nullptr);
    }

    TEST_CASE("rendering: load and store ops, clear values and layouts") {
        auto rendering = RenderingDesc{
                .render_area = VkRect2D{.offset = {0, 0}, .extent = {64, 32}},
                .layer_count = 1,
        };
        rendering.colors.push_back(AttachmentDesc{
                .resource = 0,
                .load = LoadOp::clear,
                .store = StoreOp::store,
                .clear = VkClearValue{.color = {{0.25F, 0.5F, 0.75F, 1.0F}}},
        });
        rendering.colors.push_back(AttachmentDesc{.resource = 2, .load = LoadOp::load, .store = StoreOp::dont_care});
        rendering.depth = AttachmentDesc{
                .resource = 1,
                .load = LoadOp::dont_care,
                .store = StoreOp::store,
                .clear = VkClearValue{.depthStencil = {0.0F, 0}},
        };

        auto const storage = translate(rendering, make_resources());
        REQUIRE(storage.has_value());
        REQUIRE(storage->colors.size() == 2);
        REQUIRE(storage->has_depth);

        CHECK(storage->colors[0].sType == VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO);
        CHECK(storage->colors[0].imageView == handle<VkImageView>(0x11));
        CHECK(storage->colors[0].imageLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        CHECK(storage->colors[0].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
        CHECK(storage->colors[0].storeOp == VK_ATTACHMENT_STORE_OP_STORE);
        CHECK(storage->colors[0].clearValue.color.float32[1] == 0.5F);
        CHECK(storage->colors[0].resolveMode == VK_RESOLVE_MODE_NONE);

        CHECK(storage->colors[1].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
        CHECK(storage->colors[1].storeOp == VK_ATTACHMENT_STORE_OP_DONT_CARE);

        CHECK(storage->depth.imageView == handle<VkImageView>(0x21));
        CHECK(storage->depth.imageLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        CHECK(storage->depth.loadOp == VK_ATTACHMENT_LOAD_OP_DONT_CARE);
        CHECK(storage->depth.clearValue.depthStencil.depth == 0.0F);

        auto const info = storage->info();
        CHECK(info.sType == VK_STRUCTURE_TYPE_RENDERING_INFO);
        CHECK(info.colorAttachmentCount == 2);
        CHECK(info.pColorAttachments == storage->colors.data());
        CHECK(info.pDepthAttachment == &storage->depth);
        CHECK(info.pStencilAttachment == nullptr);
        CHECK(info.renderArea.extent.width == 64);
        CHECK(info.layerCount == 1);
    }

    TEST_CASE("rendering: a resolve names the target view, mode and layout") {
        auto rendering = RenderingDesc{.render_area = VkRect2D{.extent = {64, 32}}};
        rendering.colors.push_back(AttachmentDesc{
                .resource = 0,
                .load = LoadOp::clear,
                .store = StoreOp::dont_care,
                .resolve = AttachmentResolve{.resource = 2, .mode = VK_RESOLVE_MODE_AVERAGE_BIT},
        });
        rendering.depth = AttachmentDesc{
                .resource = 1,
                .load = LoadOp::clear,
                .store = StoreOp::dont_care,
                .resolve = AttachmentResolve{.resource = 2, .mode = VK_RESOLVE_MODE_MIN_BIT},
        };

        auto const storage = translate(rendering, make_resources());
        REQUIRE(storage.has_value());
        CHECK(storage->colors[0].resolveMode == VK_RESOLVE_MODE_AVERAGE_BIT);
        CHECK(storage->colors[0].resolveImageView == handle<VkImageView>(0x31));
        CHECK(storage->colors[0].resolveImageLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        CHECK(storage->depth.resolveMode == VK_RESOLVE_MODE_MIN_BIT);
        CHECK(storage->depth.resolveImageLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    }

    TEST_CASE("rendering: no depth means no depth attachment pointer") {
        auto rendering = RenderingDesc{};
        rendering.colors.push_back(AttachmentDesc{.resource = 0});
        auto const storage = translate(rendering, make_resources());
        REQUIRE(storage.has_value());
        CHECK_FALSE(storage->has_depth);
        CHECK(storage->info().pDepthAttachment == nullptr);
    }

    TEST_CASE("rendering: a missing attachment or resolve target fails and names the resource") {
        auto missing_color = RenderingDesc{};
        missing_color.colors.push_back(AttachmentDesc{.resource = 3});
        auto const color = translate(missing_color, make_resources());
        REQUIRE_FALSE(color.has_value());
        CHECK(color.error().resource == 3);

        auto missing_resolve = RenderingDesc{};
        missing_resolve.colors.push_back(AttachmentDesc{
                .resource = 0,
                .resolve = AttachmentResolve{.resource = 3},
        });
        auto const resolve = translate(missing_resolve, make_resources());
        REQUIRE_FALSE(resolve.has_value());
        CHECK(resolve.error().kind == TranslateFailureKind::missing_image);
        CHECK(resolve.error().resource == 3);
    }
}
