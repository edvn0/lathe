#include "animation/humanoid.hxx"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace Animation::Humanoid {
    namespace {
        constexpr glm::vec3 axis_x{1.0F, 0.0F, 0.0F};
        constexpr glm::vec3 axis_z{0.0F, 0.0F, 1.0F};
        constexpr float two_pi = glm::two_pi<float>();

        auto swing_forward(float const radians) -> glm::quat { return glm::angleAxis(-radians, axis_x); }
        auto bend_back(float const radians) -> glm::quat { return glm::angleAxis(radians, axis_x); }
        auto raise_sideways(float const radians, float const side) -> glm::quat {
            return glm::angleAxis(-radians * side, axis_z);
        }

        struct Gait {
            float hip;
            float knee;
            float arm;
            float elbow;
            float bob;
            float lean;
            float twist;
        };

        constexpr Gait walk_gait{0.45F, 0.7F, 0.35F, 0.2F, 0.02F, 0.03F, 0.08F};
        constexpr Gait run_gait{0.9F, 1.4F, 0.8F, 1.4F, 0.05F, 0.22F, 0.15F};

        struct Side {
            std::uint32_t thigh, shin, foot, upper_arm, forearm;
            float sign;
        };
        constexpr Side left{ThighL, ShinL, FootL, UpperArmL, ForearmL, 1.0F};
        constexpr Side right{ThighR, ShinR, FootR, UpperArmR, ForearmR, -1.0F};

        void set_bind(Pose const &bind, PoseView const out) { copy_pose(bind.view(), out); }

        void apply_leg(PoseView const out, Side const &side, Gait const &gait, float const angle) {
            auto const hip = gait.hip * std::sin(angle);
            auto const knee = gait.knee * std::max(0.0F, std::cos(angle));
            out.rotation[side.thigh] = swing_forward(hip);
            out.rotation[side.shin] = bend_back(knee);
            out.rotation[side.foot] = swing_forward(0.5F * (hip - knee));
        }

        void apply_arm(PoseView const out, Side const &side, Gait const &gait, float const angle) {
            auto const swing = -gait.arm * std::sin(angle);
            out.rotation[side.upper_arm] = swing_forward(swing) * raise_sideways(0.05F, side.sign);
            out.rotation[side.forearm] = swing_forward(gait.elbow * (0.6F + 0.4F * std::max(0.0F, std::cos(angle))));
        }

        void gait_pose(Pose const &bind, Gait const &gait, float const phase, PoseView const out) {
            set_bind(bind, out);
            auto const theta = phase * two_pi;
            apply_leg(out, left, gait, theta);
            apply_leg(out, right, gait, theta + glm::pi<float>());
            apply_arm(out, left, gait, theta);
            apply_arm(out, right, gait, theta + glm::pi<float>());
            out.translation[Pelvis].y += gait.bob * std::cos(2.0F * theta);
            out.rotation[Pelvis] = swing_forward(0.0F) * glm::angleAxis(0.5F * gait.twist * std::sin(theta), glm::vec3{0, 1, 0});
            out.rotation[Spine] = swing_forward(gait.lean) *
                                  glm::angleAxis(-gait.twist * std::sin(theta), glm::vec3{0, 1, 0});
        }

        void idle_pose(Pose const &bind, float const phase, PoseView const out) {
            set_bind(bind, out);
            auto const breath = std::sin(phase * two_pi);
            out.rotation[Chest] = swing_forward(0.02F * breath);
            out.translation[Pelvis].y += 0.003F * breath;
            out.rotation[UpperArmL] = raise_sideways(0.08F + 0.01F * breath, 1.0F);
            out.rotation[UpperArmR] = raise_sideways(0.08F + 0.01F * breath, -1.0F);
        }

        void jump_pose(Pose const &bind, float const phase, float const tuck, float const arms, PoseView const out) {
            set_bind(bind, out);
            auto const flutter = 0.03F * std::sin(phase * two_pi);
            for (auto const &side : {left, right}) {
                out.rotation[side.thigh] = swing_forward(tuck * 0.9F + flutter * side.sign);
                out.rotation[side.shin] = bend_back(tuck * 1.3F);
                out.rotation[side.foot] = swing_forward(tuck * 0.2F);
                out.rotation[side.upper_arm] = swing_forward(arms) * raise_sideways(0.5F + 0.3F * arms, side.sign);
                out.rotation[side.forearm] = swing_forward(0.4F);
            }
        }

        void lie_pose(Pose const &bind, float const s, float const kneel, PoseView const out) {
            set_bind(bind, out);
            auto const eased = s * s * (3.0F - 2.0F * s);
            auto const bind_height = bind.joint(Pelvis).translation.y;
            out.translation[Pelvis].y = glm::mix(bind_height, 0.15F, eased) - 0.3F * kneel;
            out.rotation[Pelvis] = glm::angleAxis(glm::half_pi<float>() * eased, axis_x);
            for (auto const &side : {left, right}) {
                out.rotation[side.thigh] = swing_forward(1.1F * kneel);
                out.rotation[side.shin] = bend_back(1.6F * kneel);
                out.rotation[side.upper_arm] = swing_forward(1.2F * eased) * raise_sideways(0.3F * eased, side.sign);
            }
        }

        auto make_clip(float const duration, ProceduralClip::Function function) -> std::unique_ptr<ProceduralClip> {
            return std::make_unique<ProceduralClip>(duration, std::move(function));
        }
    }

    auto make_skeleton() -> Skeleton {
        struct JointDef {
            char const *name;
            std::int32_t parent;
            glm::vec3 offset;
        };
        std::array<JointDef, JointCount> const defs{{
                {"pelvis", no_parent, {0.0F, 1.0F, 0.0F}},
                {"spine", Pelvis, {0.0F, 0.15F, 0.0F}},
                {"chest", Spine, {0.0F, 0.2F, 0.0F}},
                {"head", Chest, {0.0F, 0.35F, 0.0F}},
                {"upper_arm_l", Chest, {0.2F, 0.2F, 0.0F}},
                {"forearm_l", UpperArmL, {0.0F, -0.28F, 0.0F}},
                {"upper_arm_r", Chest, {-0.2F, 0.2F, 0.0F}},
                {"forearm_r", UpperArmR, {0.0F, -0.28F, 0.0F}},
                {"thigh_l", Pelvis, {0.09F, -0.05F, 0.0F}},
                {"shin_l", ThighL, {0.0F, -0.45F, 0.0F}},
                {"foot_l", ShinL, {0.0F, -0.45F, 0.0F}},
                {"thigh_r", Pelvis, {-0.09F, -0.05F, 0.0F}},
                {"shin_r", ThighR, {0.0F, -0.45F, 0.0F}},
                {"foot_r", ShinR, {0.0F, -0.45F, 0.0F}},
        }};

        auto names = std::vector<std::string>{};
        auto parents = std::vector<std::int32_t>{};
        auto bind = Pose{JointCount};
        for (std::size_t i = 0; i < defs.size(); ++i) {
            names.emplace_back(defs[i].name);
            parents.push_back(defs[i].parent);
            bind.set_joint(i, {defs[i].offset, glm::quat{1.0F, 0.0F, 0.0F, 0.0F}, glm::vec3{1.0F}});
        }
        return Skeleton{std::move(names), std::move(parents), std::move(bind)};
    }

    Rig::Rig(Skeleton skeleton_in) : skeleton{std::move(skeleton_in)} {
        auto const bind = skeleton.bind_pose();
        idle = make_clip(4.0F, [bind](float p, PoseView out) { idle_pose(bind, p, out); });
        walk = make_clip(walk_stride / 1.4F, [bind](float p, PoseView out) { gait_pose(bind, walk_gait, p, out); });
        run = make_clip(run_stride / 4.5F, [bind](float p, PoseView out) { gait_pose(bind, run_gait, p, out); });
        jump_rise = make_clip(1.0F, [bind](float p, PoseView out) { jump_pose(bind, p, 0.9F, -1.4F, out); });
        jump_apex = make_clip(1.0F, [bind](float p, PoseView out) { jump_pose(bind, p, 0.6F, -0.4F, out); });
        jump_fall = make_clip(1.0F, [bind](float p, PoseView out) { jump_pose(bind, p, 0.25F, 0.3F, out); });
        lying_down = make_clip(0.9F, [bind](float p, PoseView out) { lie_pose(bind, p, std::sin(glm::pi<float>() * p), out); });
        prone = make_clip(4.0F, [bind](float p, PoseView out) {
            lie_pose(bind, 1.0F, 0.0F, out);
            out.translation[Pelvis].y += 0.004F * std::sin(p * two_pi);
        });
        getting_up = make_clip(1.1F, [bind](float p, PoseView out) { lie_pose(bind, 1.0F - p, std::sin(glm::pi<float>() * p), out); });
    }

    auto make_rig() -> std::unique_ptr<Rig> { return std::make_unique<Rig>(make_skeleton()); }

    auto make_state_table(Rig const &rig) -> StateTable {
        StateTable table{};
        auto const set = [&table](State const state, StateDefinition definition) {
            table[static_cast<std::size_t>(state)] = definition;
        };
        set(State::Idle, {"idle", rig.idle.get(), 0.0F, true, false, 0.25F, Transitions::from_ground});
        set(State::Walk, {"walk", rig.walk.get(), walk_stride, true, true, 0.2F, Transitions::from_ground});
        set(State::Run, {"run", rig.run.get(), run_stride, true, true, 0.2F, Transitions::from_ground});
        set(State::JumpRise, {"jump_rise", rig.jump_rise.get(), 0.0F, true, false, 0.1F, Transitions::from_air});
        set(State::JumpApex, {"jump_apex", rig.jump_apex.get(), 0.0F, true, false, 0.15F, Transitions::from_air});
        set(State::JumpFall, {"jump_fall", rig.jump_fall.get(), 0.0F, true, false, 0.15F, Transitions::from_air});
        set(State::LyingDown, {"lying_down", rig.lying_down.get(), 0.0F, false, false, 0.1F, Transitions::from_lying_down});
        set(State::Prone, {"prone", rig.prone.get(), 0.0F, true, false, 0.1F, Transitions::from_prone});
        set(State::GettingUp, {"getting_up", rig.getting_up.get(), 0.0F, false, false, 0.1F, Transitions::from_getting_up});
        return table;
    }
}
