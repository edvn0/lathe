#include <memory>

#include "basic_game.hxx"
#include "app/game.hxx"

// Declared (not defined) by the engine's main.cxx, which calls this once
// during startup to obtain the game instance it drives.
auto create_game() -> std::unique_ptr<IGame> { return std::make_unique<BasicGame>(); }
