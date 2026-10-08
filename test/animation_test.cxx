#include <doctest/doctest.h>

#include "animation/batch.hxx"
#include "animation/clip_state_table.hxx"
#include "animation/humanoid.hxx"

#include <glm/gtc/epsilon.hpp>

#include <BS_thread_pool.hpp>

#include <chrono>
#include <memory>
#include <span>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

using namespace Animation;

namespace {
    auto matrices_equal(glm::mat4 const &a, glm::mat4 const &b, float const eps = 1e-5F) -> bool {
        for (int c = 0; c < 4; ++c) {
            if (!glm::all(glm::epsilonEqual(a[c], b[c], eps))) {
                return false;
            }
        }
        return true;
    }

    auto two_joint_skeleton() -> Skeleton {
        auto bind = Pose{2};
        bind.set_joint(0, {{0.0F, 1.0F, 0.0F}, glm::quat{1, 0, 0, 0}, glm::vec3{1.0F}});
        bind.set_joint(1, {{0.0F, 1.0F, 0.0F}, glm::quat{1, 0, 0, 0}, glm::vec3{1.0F}});
        return Skeleton{{"root", "child"}, {no_parent, 0}, std::move(bind)};
    }

    auto make_input(std::size_t const i, std::size_t const frame) -> AnimInputs {
        AnimInputs in;
        in.horizontal_speed = static_cast<float>((i * 7 + frame) % 50) * 0.1F;
        in.grounded = (i + frame / 10) % 5 != 0;
        in.vertical_velocity = in.grounded ? 0.0F : 3.0F - static_cast<float>(frame % 12) * 0.6F;
        in.prone_requested = i % 11 == 0 && frame > 20 && frame < 60;
        in.lod = static_cast<Lod>(i % 3);
        return in;
    }
} // namespace

TEST_SUITE("unit") {
    TEST_CASE("Animation: skeleton finds joints and derives inverse bind") {
        auto const skeleton = two_joint_skeleton();
        CHECK(skeleton.find_joint("child") == 1);
        CHECK_FALSE(skeleton.find_joint("nope").has_value());
        CHECK(matrices_equal(skeleton.inverse_bind()[1], glm::translate(glm::mat4{1.0F}, {0.0F, -2.0F, 0.0F})));
    }

    TEST_CASE("Animation: palette is identity at bind pose") {
        auto const rig = Humanoid::make_rig();
        auto palette = std::vector<glm::mat4>(rig->skeleton.joint_count());
        compute_skinning_palette(rig->skeleton, rig->skeleton.bind_pose().view(), palette);
        for (auto const &matrix : palette) {
            CHECK(matrices_equal(matrix, glm::mat4{1.0F}));
        }
    }

    TEST_CASE("Animation: model matrices compose along the hierarchy") {
        auto const skeleton = two_joint_skeleton();
        auto pose = Pose{2};
        pose.set_joint(0, {{1.0F, 0.0F, 0.0F}, glm::angleAxis(glm::half_pi<float>(), glm::vec3{0, 0, 1}), glm::vec3{1.0F}});
        pose.set_joint(1, {{0.0F, 2.0F, 0.0F}, glm::quat{1, 0, 0, 0}, glm::vec3{1.0F}});
        auto model = std::vector<glm::mat4>(2);
        compute_model_matrices(skeleton, pose.view(), model);
        // Child offset (0,2,0) rotated 90 degrees about Z becomes (-2,0,0), then translated by (1,0,0).
        CHECK(model[1][3].x == doctest::Approx(-1.0F));
        CHECK(model[1][3].y == doctest::Approx(0.0F).epsilon(1e-5));
    }

    TEST_CASE("Animation: keyframe clip interpolates and clamps") {
        auto base = Pose{2};
        auto clip = KeyframeClip{base, 2.0F};
        clip.add_translation_track({1, {0.0F, 1.0F}, {{0, 0, 0}, {2, 0, 0}}});
        clip.add_rotation_track({1, {0.0F, 2.0F}, {glm::quat{1, 0, 0, 0}, glm::angleAxis(glm::pi<float>(), glm::vec3{0, 1, 0})}});

        auto out = Pose{2};
        clip.sample(0.25F, out.view()); // t = 0.5 s
        CHECK(out.joint(1).translation.x == doctest::Approx(1.0F));
        CHECK(out.joint(1).rotation.w == doctest::Approx(std::cos(glm::pi<float>() / 8.0F)));
        clip.sample(1.0F, out.view()); // beyond the last translation key: clamped
        CHECK(out.joint(1).translation.x == doctest::Approx(2.0F));
        CHECK(out.joint(0).translation.x == doctest::Approx(0.0F)); // untracked joint keeps base
    }

    TEST_CASE("Animation: procedural clip goes through the Clip interface") {
        auto const clip = ProceduralClip{1.0F, [](float phase, PoseView out) { out.translation[0].x = phase; }};
        Clip const &as_clip = clip;
        auto out = Pose{1};
        as_clip.sample(0.5F, out.view());
        CHECK(out.joint(0).translation.x == doctest::Approx(0.5F));
    }

    TEST_CASE("Animation: blend interpolates per joint and tolerates aliasing") {
        auto a = Pose{1};
        auto b = Pose{1};
        b.set_joint(0, {{2, 0, 0}, glm::angleAxis(glm::half_pi<float>(), glm::vec3{0, 1, 0}), glm::vec3{3.0F}});
        blend_poses(a.view(), b.view(), 0.5F, a.view());
        auto const result = a.joint(0);
        CHECK(result.translation.x == doctest::Approx(1.0F));
        CHECK(result.scale.x == doctest::Approx(2.0F));
        CHECK(glm::angle(result.rotation) == doctest::Approx(glm::quarter_pi<float>()));
    }

    TEST_CASE("Animation: locomotion blend weight follows speed and phase follows distance") {
        auto const rig = Humanoid::make_rig();
        auto const blend = LocomotionBlend{rig->walk.get(), rig->run.get(), 1.4F, 4.5F, 1.6F, 3.6F};
        CHECK(blend.weight(0.0F) == 0.0F);
        CHECK(blend.weight(10.0F) == 1.0F);
        CHECK(blend.weight(2.95F) == doctest::Approx(0.5F));

        auto out = Pose{Humanoid::JointCount};
        auto scratch = Pose{Humanoid::JointCount};
        blend.sample(0.25F, 3.0F, out.view(), scratch.view());
        CHECK(glm::length(out.joint(Humanoid::ThighL).rotation) == doctest::Approx(1.0F));
    }

    TEST_CASE("Animation: phase advance is distance-based so feet do not skate") {
        auto const rig = Humanoid::make_rig();
        auto const machine = AnimStateMachine{Humanoid::make_state_table(*rig)};
        auto const &walk = machine.definition(State::Walk);

        // Same distance covered at different speeds/frame rates yields the same phase.
        auto phase_after = [&](float speed, float dt, float distance) {
            auto const steps = static_cast<int>(std::round(distance / (speed * dt)));
            auto phase = 0.0F;
            for (int i = 0; i < steps; ++i) {
                phase += speed * dt / walk.stride;
            }
            return phase;
        };
        CHECK(phase_after(1.0F, 1.0F / 60.0F, 0.8F) == doctest::Approx(0.5F).epsilon(1e-3));
        CHECK(phase_after(2.0F, 1.0F / 30.0F, 0.8F) == doctest::Approx(0.5F).epsilon(1e-3));

        // And the machine itself obeys it: a stride of walking distance advances one cycle.
        auto state = StateMachineState{};
        state.current = State::Walk;
        auto pose = Pose{Humanoid::JointCount};
        auto scratch = Pose{Humanoid::JointCount};
        AnimInputs in;
        in.horizontal_speed = 2.0F;
        auto const dt = 1.0F / 60.0F;
        auto const frames = static_cast<int>(std::round(0.5F * walk.stride / (in.horizontal_speed * dt)));
        for (int i = 0; i < frames; ++i) {
            machine.update(state, in, dt, pose.view(), scratch.view());
        }
        CHECK(state.current == State::Walk);
        CHECK(state.phase == doctest::Approx(0.5F).epsilon(1e-2));

        // Standing still: the gait does not advance at all.
        auto still = StateMachineState{};
        still.current = State::Walk;
        in.horizontal_speed = 0.0F;
        auto const before = still.phase;
        machine.update(still, in, dt, pose.view(), scratch.view());
        CHECK(still.phase == doctest::Approx(before));
    }

    namespace {
        struct Harness {
            std::unique_ptr<Humanoid::Rig> rig{Humanoid::make_rig()};
            AnimStateMachine machine{Humanoid::make_state_table(*rig)};
            StateMachineState state{};
            Pose pose{Humanoid::JointCount};
            Pose scratch{Humanoid::JointCount};

            void step(AnimInputs const &in, float seconds) {
                for (float t = 0.0F; t < seconds; t += 1.0F / 60.0F) {
                    machine.update(state, in, 1.0F / 60.0F, pose.view(), scratch.view());
                }
            }
        };
    } // namespace

    TEST_CASE("Animation: state machine ground locomotion with hysteresis") {
        Harness h;
        AnimInputs in;
        h.step(in, 0.1F);
        CHECK(h.state.current == State::Idle);
        in.horizontal_speed = 1.5F;
        h.step(in, 0.1F);
        CHECK(h.state.current == State::Walk);
        in.horizontal_speed = 5.0F;
        h.step(in, 0.1F);
        CHECK(h.state.current == State::Run);
        in.horizontal_speed = 3.2F; // between run_to_walk and walk_to_run: stays running
        h.step(in, 0.1F);
        CHECK(h.state.current == State::Run);
        in.horizontal_speed = 0.0F;
        h.step(in, 0.1F);
        CHECK(h.state.current == State::Idle);
    }

    TEST_CASE("Animation: state machine airborne phases and landing") {
        Harness h;
        AnimInputs in;
        in.grounded = false;
        in.vertical_velocity = 4.0F;
        h.step(in, 0.05F);
        CHECK(h.state.current == State::JumpRise);
        in.vertical_velocity = 0.2F;
        h.step(in, 0.05F);
        CHECK(h.state.current == State::JumpApex);
        in.vertical_velocity = -4.0F;
        h.step(in, 0.05F);
        CHECK(h.state.current == State::JumpFall);
        in.grounded = true;
        in.vertical_velocity = 0.0F;
        h.step(in, 0.05F);
        CHECK(h.state.current == State::Idle);
    }

    TEST_CASE("Animation: state machine prone transitions and crossfade") {
        Harness h;
        AnimInputs in;
        in.prone_requested = true;
        h.step(in, 0.05F);
        CHECK(h.state.current == State::LyingDown);
        CHECK(h.state.fade < 1.0F); // crossfading from idle
        h.step(in, 2.0F);
        CHECK(h.state.current == State::Prone);
        CHECK(h.state.fade == doctest::Approx(1.0F));
        in.prone_requested = false;
        h.step(in, 0.05F);
        CHECK(h.state.current == State::GettingUp);
        h.step(in, 2.0F);
        CHECK(h.state.current == State::Idle);
    }

    TEST_CASE("Animation: humanoid prone pose lies near the ground") {
        Harness h;
        AnimInputs in;
        in.prone_requested = true;
        h.step(in, 3.0F);
        auto model = std::vector<glm::mat4>(Humanoid::JointCount);
        compute_model_matrices(h.rig->skeleton, h.pose.view(), model);
        CHECK(model[Humanoid::Head][3].y < 0.5F);
        CHECK(model[Humanoid::Head][3].z > 0.5F);
    }

    TEST_CASE("Animation: batch equals per-character serial evaluation" * doctest::description("also benchmarks")) {
        constexpr std::size_t count = 5000;
        constexpr std::size_t frames = 90;
        constexpr float dt = 1.0F / 60.0F;

        auto const rig = Humanoid::make_rig();
        auto const machine = AnimStateMachine{Humanoid::make_state_table(*rig)};
        BS::priority_thread_pool pool{4};

        AnimationBatch serial{rig->skeleton, machine, count, nullptr};
        AnimationBatch pooled{rig->skeleton, machine, count, &pool, {.chunk_size = 64}};

        // Reference: one standalone character at a time, replicating the batch's LOD policy.
        auto const joints = rig->skeleton.joint_count();
        std::vector<StateMachineState> ref_states(count);
        std::vector<float> ref_pending(count, 0.0F);
        std::vector<glm::mat4> ref_palette(count * joints);
        Pose pose{joints};
        Pose scratch{joints};
        auto const bind = rig->skeleton.bind_pose().view();
        for (std::size_t i = 0; i < count; ++i) {
            compute_skinning_palette(rig->skeleton, bind, std::span{ref_palette}.subspan(i * joints, joints));
        }

        std::vector<AnimInputs> inputs(count);
        std::chrono::duration<double, std::milli> serial_time{};
        std::chrono::duration<double, std::milli> pooled_time{};
        for (std::size_t frame = 0; frame < frames; ++frame) {
            for (std::size_t i = 0; i < count; ++i) {
                inputs[i] = make_input(i, frame);
            }
            auto t0 = std::chrono::steady_clock::now();
            serial.update(inputs, dt);
            auto t1 = std::chrono::steady_clock::now();
            pooled.update(inputs, dt);
            auto t2 = std::chrono::steady_clock::now();
            serial_time += t1 - t0;
            pooled_time += t2 - t1;

            for (std::size_t i = 0; i < count; ++i) {
                if (inputs[i].lod == Lod::Hold) {
                    ref_pending[i] = 0.0F;
                    continue;
                }
                ref_pending[i] += dt;
                if (inputs[i].lod == Lod::Reduced && ref_pending[i] < BatchSettings{}.reduced_period) {
                    continue;
                }
                machine.update(ref_states[i], inputs[i], ref_pending[i], pose.view(), scratch.view());
                ref_pending[i] = 0.0F;
                compute_skinning_palette(rig->skeleton, pose.view(), std::span{ref_palette}.subspan(i * joints, joints));
            }
        }
        std::printf("[animation bench] %zu characters x %zu frames: serial %.2f ms, pooled(4) %.2f ms\n", count, frames,
                    serial_time.count(), pooled_time.count());

        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < count * joints; ++i) {
            if (!matrices_equal(serial.palette()[i], ref_palette[i], 1e-6F) ||
                !matrices_equal(pooled.palette()[i], ref_palette[i], 1e-6F)) {
                ++mismatches;
            }
        }
        CHECK(mismatches == 0);
        CHECK(pooled.state(7).current == serial.state(7).current);
    }

    TEST_CASE("Animation: Hold LOD freezes the palette") {
        auto const rig = Humanoid::make_rig();
        auto const machine = AnimStateMachine{Humanoid::make_state_table(*rig)};
        AnimationBatch batch{rig->skeleton, machine, 2, nullptr};
        std::vector<AnimInputs> inputs(2);
        inputs[0].horizontal_speed = 2.0F;
        inputs[1].horizontal_speed = 2.0F;
        inputs[1].lod = Lod::Hold;
        for (int i = 0; i < 30; ++i) {
            batch.update(inputs, 1.0F / 60.0F);
        }
        CHECK(batch.state(0).current == State::Walk);
        CHECK(batch.state(1).current == State::Idle);
        CHECK(matrices_equal(batch.palette(1)[Humanoid::ThighL], glm::mat4{1.0F}));
        CHECK_FALSE(matrices_equal(batch.palette(0)[Humanoid::ThighL], glm::mat4{1.0F}));
    }

    TEST_CASE("Animation: LocomotionStateTable maps arbitrary clips and falls back") {
        // Each clip writes its phase (and its id) into joint 0, so the table's wiring can be read back.
        auto const marker = [](float id) {
            return ProceduralClip{1.0F, [id](float phase, PoseView out) {
                                      out.translation[0] = {phase, id, 0.0F};
                                  }};
        };
        auto const idle = marker(1.0F);
        auto const walk = marker(2.0F);
        auto const jump = marker(3.0F);
        auto const lie = marker(4.0F);

        auto const sample = [](StateDefinition const &definition, float phase) {
            auto pose = Pose{1};
            definition.clip->sample(phase, pose.view());
            return pose.joint(0).translation;
        };

        LocomotionStateTable const table{LocomotionClipSet{
                .idle = &idle, .walk = &walk, .jump = &jump, .lie_down = &lie, .walk_stride = 1.2F}};
        auto const &t = table.table();

        CHECK(t[static_cast<std::size_t>(State::Idle)].clip == &idle);
        CHECK(t[static_cast<std::size_t>(State::Walk)].stride == doctest::Approx(1.2F));
        // No run clip: run reuses walk (and its stride).
        CHECK(t[static_cast<std::size_t>(State::Run)].clip == &walk);
        CHECK(t[static_cast<std::size_t>(State::Run)].stride == doctest::Approx(1.2F));
        // Airborne states are held poses of the jump clip, whatever phase the machine asks for.
        auto const rise = sample(t[static_cast<std::size_t>(State::JumpRise)], 0.9F);
        CHECK(rise.y == doctest::Approx(3.0F));
        CHECK(rise.x == doctest::Approx(0.3F));
        CHECK(sample(t[static_cast<std::size_t>(State::JumpFall)], 0.1F).x == doctest::Approx(0.7F));
        // Prone is the last frame of the lie-down clip; getting up plays it backwards.
        CHECK(sample(t[static_cast<std::size_t>(State::Prone)], 0.2F).x == doctest::Approx(1.0F));
        CHECK(sample(t[static_cast<std::size_t>(State::GettingUp)], 0.25F).x == doctest::Approx(0.75F));
        CHECK(sample(t[static_cast<std::size_t>(State::LyingDown)], 0.25F).y == doctest::Approx(4.0F));

        // Only idle: every state falls back to it, and no state is left without a clip.
        LocomotionStateTable const bare{LocomotionClipSet{.idle = &idle}};
        for (auto const &definition : bare.table()) {
            CHECK(definition.clip == &idle);
        }
    }
}
