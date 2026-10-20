#include "rendering/imgui_sdf_font.hxx"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <new>

#include <imgui.h>
#include <imgui_internal.h>

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_SIZES_H
#include FT_MODULE_H

namespace gui {

    namespace {

        struct LoaderData {
            FT_Library library = nullptr;
        };

        struct SourceData {
            FT_Face face = nullptr;
            ImFontBaked const *last_activated = nullptr;
        };

        struct BakedData {
            FT_Size size = nullptr;
            // Logical pixels per pixel of the rasterised field.
            float to_logical = 1.0F;
        };

        // The size ImGui wants this source at: a merged source scales with its own SizePixels against the main one.
        auto logical_size(ImFontConfig const *src, ImFontBaked const *baked) -> float {
            auto size = baked->Size;
            auto const reference = baked->OwnerFont->Sources[0]->SizePixels;
            if (src->MergeMode && src->SizePixels != 0.0F) {
                size *= src->SizePixels / reference;
            }
            return size;
        }

        auto loader_init(ImFontAtlas *atlas) -> bool {
            auto *data = IM_NEW(LoaderData)();
            if (FT_Init_FreeType(&data->library) != 0) {
                IM_DELETE(data);
                return false;
            }
            // Variable and composite fonts draw glyphs as overlapping contours, which the default SDF generation turns
            // into specks along the strokes.
            FT_Bool overlaps = 1;
            FT_Property_Set(data->library, "sdf", "overlaps", &overlaps);
            atlas->FontLoaderData = data;
            return true;
        }

        auto loader_shutdown(ImFontAtlas *atlas) -> void {
            auto *data = static_cast<LoaderData *>(atlas->FontLoaderData);
            FT_Done_FreeType(data->library);
            IM_DELETE(data);
            atlas->FontLoaderData = nullptr;
        }

        auto source_init(ImFontAtlas *atlas, ImFontConfig *src) -> bool {
            auto *loader = static_cast<LoaderData *>(atlas->FontLoaderData);
            auto *source = IM_NEW(SourceData)();
            if (auto const error = FT_New_Memory_Face(loader->library, static_cast<FT_Byte const *>(src->FontData),
                                                      static_cast<FT_Long>(src->FontDataSize), src->FontNo,
                                                      &source->face);
                error != 0) {
                std::fprintf(stderr, "[ImGui] FreeType could not open a font (error %d)\n", error);
                IM_DELETE(source);
                return false;
            }
            FT_Select_Charmap(source->face, FT_ENCODING_UNICODE);
            src->FontLoaderData = source;
            return true;
        }

        auto source_destroy(ImFontAtlas *, ImFontConfig *src) -> void {
            auto *source = static_cast<SourceData *>(src->FontLoaderData);
            if (source == nullptr) {
                return;
            }
            FT_Done_Face(source->face);
            IM_DELETE(source);
            src->FontLoaderData = nullptr;
        }

        auto source_contains_glyph(ImFontAtlas *, ImFontConfig *src, ImWchar codepoint) -> bool {
            return FT_Get_Char_Index(static_cast<SourceData *>(src->FontLoaderData)->face, codepoint) != 0;
        }

        auto baked_init(ImFontAtlas *, ImFontConfig *src, ImFontBaked *baked, void *baked_data) -> bool {
            auto *source = static_cast<SourceData *>(src->FontLoaderData);
            auto *data = ::new (baked_data) BakedData{};
            data->to_logical = logical_size(src, baked) / sdf_pixels_per_em;

            FT_New_Size(source->face, &data->size);
            FT_Activate_Size(data->size);
            source->last_activated = baked;

            auto request = FT_Size_RequestRec{};
            request.type = FT_SIZE_REQUEST_TYPE_REAL_DIM;
            request.height = static_cast<FT_Long>(sdf_pixels_per_em * 64.0F);
            FT_Request_Size(source->face, &request);

            if (!src->MergeMode) {
                // Whole logical pixels, so lines of text sit on the pixel grid whatever the field's resolution.
                auto const &metrics = data->size->metrics;
                baked->Ascent = std::ceil(static_cast<float>(metrics.ascender) / 64.0F * data->to_logical);
                baked->Descent = std::floor(static_cast<float>(metrics.descender) / 64.0F * data->to_logical);
            }
            return true;
        }

        auto baked_destroy(ImFontAtlas *, ImFontConfig *, ImFontBaked *, void *baked_data) -> void {
            auto *data = static_cast<BakedData *>(baked_data);
            FT_Done_Size(data->size);
            data->~BakedData();
        }

        auto baked_load_glyph(ImFontAtlas *atlas, ImFontConfig *src, ImFontBaked *baked, void *baked_data,
                              ImWchar codepoint, ImFontGlyph *out_glyph, float *out_advance_x) -> bool {
            auto *source = static_cast<SourceData *>(src->FontLoaderData);
            auto const *data = static_cast<BakedData *>(baked_data);

            auto const index = FT_Get_Char_Index(source->face, codepoint);
            if (index == 0) {
                return false;
            }

            if (source->last_activated != baked) {
                FT_Activate_Size(data->size);
                source->last_activated = baked;
            }

            if (FT_Load_Glyph(source->face, index, FT_LOAD_NO_HINTING | FT_LOAD_NO_BITMAP) != 0) {
                return false;
            }

            auto *const slot = source->face->glyph;
            // The linear advance is unrounded; the integer one would round to the field's pixel grid.
            auto const advance_x = static_cast<float>(slot->linearHoriAdvance) / 65536.0F * data->to_logical;
            if (out_advance_x != nullptr) {
                *out_advance_x = advance_x;
                return true;
            }

            // Rendering coverage first makes the SDF renderer work from that bitmap (FreeType's "bsdf"). Its outline
            // based generator gets the sign wrong along the centre of thin strokes, which shows as dashes.
            if (FT_Render_Glyph(slot, FT_RENDER_MODE_NORMAL) != 0 || FT_Render_Glyph(slot, FT_RENDER_MODE_SDF) != 0) {
                return false;
            }

            auto const &bitmap = slot->bitmap;
            auto const width = static_cast<int>(bitmap.width);
            auto const height = static_cast<int>(bitmap.rows);

            out_glyph->Codepoint = static_cast<unsigned int>(codepoint) & 0x3FFFFFFU;
            out_glyph->AdvanceX = advance_x;

            if (width == 0 || height == 0) {
                return true;
            }

            auto const pack_id = ImFontAtlasPackAddRect(atlas, width, height);
            if (pack_id == ImFontAtlasRectId_Invalid) {
                return false;
            }
            auto *const rect = ImFontAtlasPackGetRect(atlas, pack_id);

            atlas->Builder->TempBuffer.resize(width * height * 4);
            auto *const pixels = reinterpret_cast<std::uint32_t *>(atlas->Builder->TempBuffer.Data);
            for (auto y = 0; y < height; ++y) {
                auto const *const row = bitmap.buffer + static_cast<std::ptrdiff_t>(y) * bitmap.pitch;
                for (auto x = 0; x < width; ++x) {
                    pixels[y * width + x] = 0x00FFFFFFU | (static_cast<std::uint32_t>(row[x]) << 24U);
                }
            }

            auto const reference = baked->OwnerFont->Sources[0]->SizePixels;
            auto const offset_scale = reference != 0.0F ? baked->Size / reference : 1.0F;
            auto const offset_x = ImFloor(src->GlyphOffset.x * offset_scale + 0.5F);
            auto const offset_y = ImFloor(src->GlyphOffset.y * offset_scale + 0.5F) + baked->Ascent;

            auto const left = static_cast<float>(slot->bitmap_left);
            auto const top = static_cast<float>(-slot->bitmap_top);
            out_glyph->X0 = left * data->to_logical + offset_x;
            out_glyph->Y0 = top * data->to_logical + offset_y;
            out_glyph->X1 = (left + static_cast<float>(width)) * data->to_logical + offset_x;
            out_glyph->Y1 = (top + static_cast<float>(height)) * data->to_logical + offset_y;
            out_glyph->Visible = true;
            out_glyph->Colored = false;
            out_glyph->PackId = pack_id;

            ImFontAtlasBakedSetFontGlyphBitmap(atlas, baked, src, out_glyph, rect,
                                               reinterpret_cast<unsigned char const *>(pixels), ImTextureFormat_RGBA32,
                                               width * 4);
            return true;
        }

    }

    auto sdf_font_loader() -> ImFontLoader const * {
        static auto const loader = [] {
            auto result = ImFontLoader{};
            result.Name = "FreeType SDF";
            result.LoaderInit = loader_init;
            result.LoaderShutdown = loader_shutdown;
            result.FontSrcInit = source_init;
            result.FontSrcDestroy = source_destroy;
            result.FontSrcContainsGlyph = source_contains_glyph;
            result.FontBakedInit = baked_init;
            result.FontBakedDestroy = baked_destroy;
            result.FontBakedLoadGlyph = baked_load_glyph;
            result.FontBakedSrcLoaderDataSize = sizeof(BakedData);
            return result;
        }();
        return &loader;
    }

}
