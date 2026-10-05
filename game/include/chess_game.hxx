#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <entt/entt.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "app/game.hxx"
#include "assets/model.hxx"

// "Chess": the voxel chess set on a board, with pawn movement and nothing else yet. Every piece is placed, but only
// pawns can be picked up: one square forward, two from the starting rank, diagonal captures, turns alternating. No
// en passant, no promotion (a pawn that reaches the far rank just stays there), no check.
//
// The board is built in code on every on_populate(); the rules live in this class and find the pieces by entity name
// when a scene first shows up in on_update(), which is every time the editor starts play.
//
// Controls: the mouse points at a square (a ray from the camera onto the board plane) and a left click picks up or
// puts down; a right click or Backspace puts back. The arrows or WASD move the same cursor, and Enter or Space act
// on it. R starts over. The picked-up pawn gets the renderer's outline (Components::Outlined). Escape isn't ours:
// while the cursor is free the editor takes it to stop playing.
class ChessGame final : public IGame {
public:
    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;

    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void override;
    auto on_mouse_button_pressed(Scene &scene, MouseButtonPressedEvent const &event) -> void override;

    [[nodiscard]] auto wants_cursor() const -> bool override { return true; }
    auto on_cursor_position(Scene &scene, CursorPositionEvent const &event) -> void override;

    auto on_ui(Scene &scene, Renderer &renderer) -> void override;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

    [[nodiscard]] auto benchmark_camera_path() const -> std::vector<CameraKeyframe> override;

private:
    enum class Side : std::uint8_t { white, black };
    enum class Kind : std::uint8_t { pawn, rook, knight, bishop, queen, king };

    // Files a..h are x 0..7 and ranks 1..8 are y 0..7, so white starts on y 0 and 1.
    using Square = glm::ivec2;

    static constexpr int board_size = 8;

    struct Piece {
        Side side{};
        Kind kind{};
        entt::entity entity{entt::null};
    };

    // Index into pieces_, or -1 for an empty square.
    using Board = std::array<int, board_size * board_size>;

    [[nodiscard]] static auto on_board(Square square) noexcept -> bool;
    [[nodiscard]] static auto index_of(Square square) noexcept -> std::size_t;
    [[nodiscard]] static auto square_name(Square square) -> std::string;

    // The centre of `square` on top of the board, in world space.
    [[nodiscard]] static auto world_position(Square square) noexcept -> glm::vec3;
    [[nodiscard]] static auto model_path(Side side, Kind kind) -> std::string;

    // All 32 pieces and their starting squares, in a fixed order: a piece's index is the number in its entity name.
    [[nodiscard]] static auto starting_layout() -> std::vector<std::pair<Piece, Square>>;

    // Finds the pieces by name in `scene` and lays them out for a new game.
    auto bind_to(Scene &scene) -> void;
    auto restart(Scene &scene) -> void;

    [[nodiscard]] auto piece_at(Square square) const noexcept -> Piece const *;

    // Where the pawn on `from` can go.
    [[nodiscard]] auto pawn_targets(Square from) const -> std::vector<Square>;

    auto activate_cursor(Scene &scene) -> void;
    auto move_piece(Scene &scene, Square from, Square to) -> void;

    // Moves the cursor, selection and move markers to match the state, and outlines the selected piece.
    auto place_markers(Scene &scene) -> void;

    // The square under the cursor's ray, if it hits the board.
    [[nodiscard]] static auto pick_square(CursorPositionEvent const &event, float aspect_ratio)
            -> std::optional<Square>;

    // The scene bind_to() last ran against, so play() is noticed without the engine telling us.
    Scene const *bound_scene_ = nullptr;

    std::vector<Piece> pieces_;
    Board board_{};

    entt::entity cursor_entity_{entt::null};
    entt::entity selection_entity_{entt::null};
    std::vector<entt::entity> target_entities_;

    // The entity carrying Components::Outlined, so it can be taken off when the selection moves.
    entt::entity outlined_entity_{entt::null};

    Square cursor_{4, 1};
    std::optional<Square> selected_;
    std::vector<Square> targets_;

    Side to_move_ = Side::white;
    std::string status_;

    // Set by the input callbacks, consumed by the next on_update(), which is where the scene is known.
    glm::ivec2 cursor_step_{0};
    bool activate_requested_ = false;
    bool click_requested_ = false;
    bool cancel_requested_ = false;
    bool restart_requested_ = false;

    // The aspect ratio camera() was last asked for, which is the image the cursor position is relative to.
    mutable float aspect_ratio_ = 16.0F / 9.0F;

    // The square under the mouse, if it is over the board.
    std::optional<Square> hovered_;

    // on_ui() is called whether or not the editor is playing, and isn't told which; the panel belongs to the game, so
    // it only draws on frames just after an on_update().
    std::uint32_t frames_since_update_ = 0;
};
