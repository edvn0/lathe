#include <memory>

#include "app/game.hxx"
#include "basic_game.hxx"

// Called once by the engine's main.cxx at startup.
auto create_game() -> std::unique_ptr<IGame> { return std::make_unique<BasicGame>(); }
