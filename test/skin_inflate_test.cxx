#include <doctest/doctest.h>

#include "assets/load_model.hxx"
#include "gpu/skinning.hxx"

namespace {

    auto make_skinned_data(bool with_clip) -> ModelCpuData {
        Animation::Pose bind{2};
        auto animation = std::make_shared<ModelAnimationData>(ModelAnimationData{
                .skeleton = Animation::Skeleton{{"root", "tip"}, {Animation::no_parent, 0}, bind},
                .clips = {},
        });

        if (with_clip) {
            ImportedClip clip;
            clip.name = "slide";
            clip.duration = 1.0F;
            clip.base = bind;
            clip.translations.push_back(Animation::Vec3Track{
                    .joint = 1,
                    .times = {0.0F, 1.0F},
                    .values = {glm::vec3{0.0F}, glm::vec3{2.0F, 0.0F, 0.0F}},
            });
            animation->clips.push_back(std::move(clip));
        }

        ModelCpuPrimitive primitive;
        primitive.vertices = {ModelVertex{.position = {0.0F, 0.0F, 0.0F}, .normal = {0.0F, 0.0F, 1.0F},
                                          .tangent = {1.0F, 0.0F, 0.0F, 1.0F}},
                              ModelVertex{.position = {1.0F, 0.0F, 0.0F}, .normal = {0.0F, 0.0F, 1.0F},
                                          .tangent = {1.0F, 0.0F, 0.0F, 1.0F}}};
        primitive.compressed_vertices = compress_vertices(primitive.vertices);
        primitive.skin = {SkinVertex{.joints = {1, 0, 0, 0}, .weights = {65535, 0, 0, 0}},
                          SkinVertex{.joints = {1, 0, 0, 0}, .weights = {65535, 0, 0, 0}}};

        ModelCpuData data;
        data.meshes.push_back(ModelCpuMesh{.primitives = {std::move(primitive)}});
        data.bounds = std::make_pair(glm::vec3{0.0F}, glm::vec3{1.0F, 0.0F, 0.0F});
        data.animation = animation;
        return data;
    }

}

TEST_SUITE("unit") {
    TEST_CASE("compute_skin_inflate bounds the sampled clip displacement with a margin") {
        auto const inflate = compute_skin_inflate(make_skinned_data(true));
        CHECK(inflate == doctest::Approx(2.0F * 1.25F).epsilon(0.02));
    }

    TEST_CASE("compute_skin_inflate falls back to the model radius without clips") {
        auto const inflate = compute_skin_inflate(make_skinned_data(false));
        CHECK(inflate == doctest::Approx(0.5F));
    }

    TEST_CASE("compute_skin_inflate is zero without a skinned primitive") {
        auto data = make_skinned_data(true);
        data.meshes[0].primitives[0].skin.clear();
        CHECK(compute_skin_inflate(data) == 0.0F);
    }
}
