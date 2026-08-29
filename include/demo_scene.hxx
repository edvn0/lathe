#pragma once

#include "forward.hxx"

// Builds the sample level into app.editor_scene: a player capsule, a
// scattering of instanced helmet models, a physics-cube grid, a textured
// floor, a few point/spot lights, and a field of wind-swaying grass clumps.
//
// This is placeholder content for exercising the engine's rendering and
// physics systems end-to-end -- it deliberately doesn't live as a method on
// Application, so that changing what the sample level contains never
// requires touching the runtime class itself.
//
// Called once from Application::on_startup() and again whenever the user
// presses Ctrl+R to reload it. Clears app.editor_scene's registry first, so
// it's safe to call repeatedly.
auto build_demo_scene(Application &app) -> void;
