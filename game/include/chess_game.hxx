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

#include "chess_engine.hxx"

class ChessGame final : public IGame {
public:
    auto attach_host(GameHost host) -> void override { host_ = std::move(host); }

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
    enum class Screen : std::uint8_t {
        playing,
        loading,
        menu,
        paused,
        game_over,
    };

    auto draw_player_ui(Renderer &renderer) -> void;
    auto draw_loading(Renderer &renderer) -> void;
    auto draw_menu() -> void;
    auto draw_pause() -> void;
    auto draw_game_over() -> void;
    auto draw_hud() -> void;
    auto draw_state_banner() -> void;
    auto draw_promotion_choice() -> void;

    auto start_new_game() -> void;

    using Square = glm::ivec2;

    static constexpr int board_size = 8;
    static constexpr std::size_t piece_count = 32;

    static constexpr std::size_t target_marker_count = 32;

    [[nodiscard]] static auto on_board(Square square) noexcept -> bool;

    [[nodiscard]] static auto to_chess_square(Square square) noexcept -> chess::Square;
    [[nodiscard]] static auto from_chess_square(chess::Square square) noexcept -> Square;

    [[nodiscard]] static auto square_name(chess::Square square) -> std::string;

    [[nodiscard]] static auto world_position(Square square) noexcept -> glm::vec3;

    [[nodiscard]] static auto model_path(chess::Piece piece) -> std::string;
    [[nodiscard]] static auto model_index(chess::Piece piece) noexcept -> std::size_t;
    [[nodiscard]] static auto piece_side(chess::Piece piece) noexcept -> chess::Side;
    [[nodiscard]] static auto piece_is_black(chess::Piece piece) noexcept -> bool;

    auto bind_to(Scene &scene) -> void;

    auto restart(Scene &scene) -> void;

    auto sync_pieces(Scene &scene) -> void;

    auto activate_cursor(Scene &scene) -> void;

    auto execute_move(Scene &scene, chess::Move move) -> void;

    auto clear_selection() -> void;
    auto rebuild_target_squares() -> void;

    auto place_markers(Scene &scene) -> void;

    [[nodiscard]] auto pick_square(CursorPositionEvent const &event, float aspect_ratio) const
            -> std::optional<Square>;

    auto update_camera(float delta_time) -> void;

    float camera_angle_ = 0.0F;
    float camera_target_angle_ = 0.0F;
    chess::Side camera_side_ = chess::Side::white;

    chess::ChessEngine chess_engine_;

    Scene const *bound_scene_ = nullptr;

    std::array<entt::entity, piece_count> piece_entities_{};

    std::array<ModelHandle, 12> piece_models_{};

    entt::entity cursor_entity_{entt::null};
    entt::entity selection_entity_{entt::null};

    std::array<entt::entity, target_marker_count> target_entities_{};

    entt::entity outlined_entity_{entt::null};

    Square cursor_{4, 1};
    std::optional<Square> hovered_;

    std::optional<chess::Square> selected_;
    std::vector<chess::Move> selected_moves_;

    std::vector<chess::Square> target_squares_;

    std::optional<chess::Square> pending_promotion_target_;
    std::optional<chess::PieceType> promotion_requested_;

    std::string status_;

    GameHost host_;

    Screen screen_ = Screen::playing;

    float game_over_timer_ = 0.0F;

    std::size_t loading_peak_ = 0;
    std::uint32_t loading_frames_ = 0;

    glm::ivec2 cursor_step_{0};

    bool activate_requested_ = false;
    bool click_requested_ = false;
    bool cancel_requested_ = false;
    bool restart_requested_ = false;

    mutable float aspect_ratio_ = 16.0F / 9.0F;

    std::uint32_t frames_since_update_ = 0;
};
