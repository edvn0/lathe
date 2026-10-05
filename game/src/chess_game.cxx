#include "chess_game.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <optional>
#include <ranges>
#include <string_view>
#include <utility>

#include <GLFW/glfw3.h>
#include <glm/common.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec4.hpp>
#include <imgui.h>

#include "assets/material_storage.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/imgui_widget.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace {

    // The board model: 8 squares of 3.6 voxels at the pack's 0.3 import scale, centred on the origin, 0.3 m thick.
    constexpr float square_size = 1.08F;
    constexpr float board_top_y = 0.3F;

    constexpr glm::vec3 world_up{0.0F, 1.0F, 0.0F};

    // Where a marker or captured piece goes when it isn't wanted.
    constexpr glm::vec3 stowed{0.0F, -50.0F, 0.0F};

    // From white's side, high enough to read the ranks.
    constexpr glm::vec3 camera_eye{0.0F, 10.5F, -6.5F};
    constexpr glm::vec3 camera_target{0.0F, 0.0F, 0.4F};
    constexpr float camera_fov_degrees = 40.0F;
    constexpr float camera_near_clip = 0.1F;
    constexpr float camera_far_clip = 200.0F;

    [[nodiscard]] auto view_matrix() -> glm::mat4 { return glm::lookAtLH(camera_eye, camera_target, world_up); }

    [[nodiscard]] auto projection_matrix(float aspect_ratio) -> glm::mat4 {
        return glm::perspectiveLH_ZO(glm::radians(camera_fov_degrees), aspect_ratio, camera_near_clip,
                                     camera_far_clip);
    }

    constexpr std::string_view board_model_path = "assets/models/chess/chess_board.glb";

    [[nodiscard]] auto piece_rotation(bool black) -> glm::quat {
        // Black faces white across the board.
        return black ? glm::angleAxis(glm::radians(180.0F), world_up) : glm::quat{1.0F, 0.0F, 0.0F, 0.0F};
    }

} // namespace

auto ChessGame::on_board(Square square) noexcept -> bool {
    return square.x >= 0 && square.x < board_size && square.y >= 0 && square.y < board_size;
}

auto ChessGame::index_of(Square square) noexcept -> std::size_t {
    return static_cast<std::size_t>((square.y * board_size) + square.x);
}

auto ChessGame::square_name(Square square) -> std::string {
    return std::format("{}{}", static_cast<char>('a' + square.x), square.y + 1);
}

auto ChessGame::world_position(Square square) noexcept -> glm::vec3 {
    return glm::vec3{(static_cast<float>(square.x) - 3.5F) * square_size, board_top_y,
                     (static_cast<float>(square.y) - 3.5F) * square_size};
}

auto ChessGame::model_path(Side side, Kind kind) -> std::string {
    constexpr std::array<std::string_view, 6> kinds{"pawn", "rook", "knight", "bishop", "queen", "king"};

    return std::format("assets/models/chess/{}_{}.glb", side == Side::white ? "white" : "black",
                       kinds[static_cast<std::size_t>(kind)]);
}

auto ChessGame::starting_layout() -> std::vector<std::pair<Piece, Square>> {
    constexpr std::array<Kind, board_size> back_rank{Kind::rook, Kind::knight, Kind::bishop, Kind::queen,
                                                     Kind::king, Kind::bishop, Kind::knight, Kind::rook};

    std::vector<std::pair<Piece, Square>> layout;
    layout.reserve(32);

    for (auto const side: {Side::white, Side::black}) {
        auto const back_y = side == Side::white ? 0 : 7;
        auto const pawn_y = side == Side::white ? 1 : 6;

        for (int file = 0; file < board_size; ++file) {
            layout.emplace_back(Piece{.side = side, .kind = back_rank[static_cast<std::size_t>(file)]},
                                Square{file, back_y});
        }
        for (int file = 0; file < board_size; ++file) {
            layout.emplace_back(Piece{.side = side, .kind = Kind::pawn}, Square{file, pawn_y});
        }
    }

    return layout;
}

auto ChessGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    scene.get_registry().clear();

    scene.environment = new_scene_environment();
    scene.environment.sun.elevation_degrees = 55.0F;
    scene.environment.sun.azimuth_degrees = 200.0F;

    bound_scene_ = nullptr; // rebind on the next on_update()

    // The board.
    {
        auto const board = Entity{&scene, "board"};
        board.emplace<Components::Transform>(Components::Transform{});

        auto model = renderer.load_model(std::string{board_model_path});

        if (model) {
            board.emplace<Components::Model>(Components::Model{.model = *model});
        } else {
            error("[ChessGame] Could not load '{}': {}; run the game with the chess pack in assets/models/chess/",
                  board_model_path, describe(model.error()));
        }
    }

    // The pieces, each model loaded once however many pawns share it.
    std::array<ModelHandle, 12> models{};

    auto const model_for = [&](Piece const &piece) -> ModelHandle {
        auto &slot = models[(static_cast<std::size_t>(piece.side) * 6) + static_cast<std::size_t>(piece.kind)];

        if (!slot.valid()) {
            auto loaded = renderer.load_model(model_path(piece.side, piece.kind));

            if (loaded) {
                slot = *loaded;
            } else {
                error("[ChessGame] Could not load '{}': {}; using a cube", model_path(piece.side, piece.kind),
                      describe(loaded.error()));
                slot = engine_models.cube;
            }
        }

        return slot;
    };

    for (auto &&[index, placement]: starting_layout() | std::views::enumerate) {
        auto const &[piece, square] = placement;

        auto const entity = GeneratedEntity{&scene, "piece_{}", static_cast<std::uint32_t>(index)};
        entity.emplace<Components::Transform>(Components::Transform{
                .position = world_position(square),
                .rotation = piece_rotation(piece.side == Side::black),
        });
        entity.emplace<Components::Model>(Components::Model{.model = model_for(piece)});
    }

    // Markers: flat slabs on the board, moved around (or stowed under it) rather than created and destroyed.
    auto &images = renderer.image_storage();
    auto &samplers = renderer.sampler_storage();

    std::vector<MaterialHandle> created_materials;

    auto const flat_material = [&](std::string name, glm::vec3 const &colour) -> MaterialHandle {
        auto material = renderer.create_material(
                MaterialCreateInfo{
                        .base_colour_factor = glm::vec4{colour, 1.0F},
                        .metallic_factor = 0.0F,
                        .roughness_factor = 0.6F,
                        .base_colour_texture = images.white(),
                        .normal_texture = images.flat_normal(),
                        .metallic_roughness_texture = images.metallic_roughness(),
                        .occlusion_texture = images.occlusion(),
                        .emissive_texture = images.emissive(),
                        .sampler = samplers.linear_repeat(),
                },
                std::move(name));

        if (!material) {
            error("[ChessGame] Could not create a material: {}", describe(material.error()));
            return MaterialHandle{};
        }

        created_materials.push_back(*material);
        return *material;
    };

    auto const cursor_look = flat_material("chess.cursor", glm::vec3{0.95F, 0.8F, 0.1F});
    auto const selection_look = flat_material("chess.selection", glm::vec3{0.15F, 0.45F, 0.9F});
    auto const target_look = flat_material("chess.target", glm::vec3{0.2F, 0.75F, 0.3F});

    auto const add_marker = [&](std::string_view name, MaterialHandle material, float footprint) {
        auto const entity = Entity{&scene, name};
        entity.emplace<Components::Transform>(Components::Transform{
                .position = stowed,
                .scale = glm::vec3{square_size * footprint, 0.01F, square_size * footprint},
        });
        entity.emplace<Components::Model>(Components::Model{.model = engine_models.cube});

        if (material.valid()) {
            entity.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = material});
        }
    };

    add_marker("cursor", cursor_look, 0.9F);
    add_marker("selection", selection_look, 0.96F);

    // A pawn has at most four places to go.
    for (int index = 0; index < 4; ++index) {
        auto const entity = GeneratedEntity{&scene, "target_{}", static_cast<std::uint32_t>(index)};
        entity.emplace<Components::Transform>(Components::Transform{
                .position = stowed,
                .scale = glm::vec3{square_size * 0.7F, 0.01F, square_size * 0.7F},
        });
        entity.emplace<Components::Model>(Components::Model{.model = engine_models.cube});

        if (target_look.valid()) {
            entity.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = target_look});
        }
    }

    for (auto const material: created_materials) {
        renderer.release_material(material);
    }

    info("[ChessGame] Set up the board");
}

auto ChessGame::bind_to(Scene &scene) -> void {
    bound_scene_ = &scene;

    pieces_.clear();
    outlined_entity_ = entt::null;

    for (auto &&[index, placement]: starting_layout() | std::views::enumerate) {
        auto piece = placement.first;
        piece.entity = scene.find_entity(std::format("piece_{}", index));
        pieces_.push_back(piece);
    }

    cursor_entity_ = scene.find_entity("cursor");
    selection_entity_ = scene.find_entity("selection");

    target_entities_.clear();

    for (int index = 0; index < 4; ++index) {
        target_entities_.push_back(scene.find_entity(std::format("target_{}", index)));
    }

    restart_requested_ = true;
}

auto ChessGame::restart(Scene &scene) -> void {
    auto &registry = scene.get_registry();

    board_.fill(-1);

    for (auto &&[index, placement]: starting_layout() | std::views::enumerate) {
        auto const &square = placement.second;
        board_[index_of(square)] = static_cast<int>(index);

        if (auto const entity = pieces_[static_cast<std::size_t>(index)].entity;
            registry.valid(entity) && registry.all_of<Components::Transform>(entity)) {
            registry.get<Components::Transform>(entity).position = world_position(square);
        }
    }

    cursor_ = Square{4, 1};
    hovered_.reset();
    selected_.reset();
    targets_.clear();
    to_move_ = Side::white;
    status_.clear();
}

auto ChessGame::piece_at(Square square) const noexcept -> Piece const * {
    if (!on_board(square)) {
        return nullptr;
    }

    auto const index = board_[index_of(square)];
    return index < 0 ? nullptr : &pieces_[static_cast<std::size_t>(index)];
}

auto ChessGame::pawn_targets(Square from) const -> std::vector<Square> {
    std::vector<Square> targets;

    auto const *pawn = piece_at(from);

    if (pawn == nullptr || pawn->kind != Kind::pawn) {
        return targets;
    }

    auto const forward = pawn->side == Side::white ? 1 : -1;
    auto const start_y = pawn->side == Side::white ? 1 : 6;

    auto const one = Square{from.x, from.y + forward};

    if (on_board(one) && piece_at(one) == nullptr) {
        targets.push_back(one);

        auto const two = Square{from.x, from.y + (2 * forward)};

        if (from.y == start_y && piece_at(two) == nullptr) {
            targets.push_back(two);
        }
    }

    for (auto const dx: {-1, 1}) {
        auto const diagonal = Square{from.x + dx, from.y + forward};
        auto const *other = piece_at(diagonal);

        if (other != nullptr && other->side != pawn->side) {
            targets.push_back(diagonal);
        }
    }

    return targets;
}

auto ChessGame::move_piece(Scene &scene, Square from, Square to) -> void {
    auto &registry = scene.get_registry();

    auto const moving = board_[index_of(from)];
    auto const captured = board_[index_of(to)];

    // A captured piece is stowed under the board rather than destroyed, so restart() can bring it back.
    if (captured >= 0) {
        if (auto const entity = pieces_[static_cast<std::size_t>(captured)].entity;
            registry.valid(entity) && registry.all_of<Components::Transform>(entity)) {
            registry.get<Components::Transform>(entity).position = stowed;
        }
    }

    board_[index_of(from)] = -1;
    board_[index_of(to)] = moving;

    if (auto const entity = pieces_[static_cast<std::size_t>(moving)].entity;
        registry.valid(entity) && registry.all_of<Components::Transform>(entity)) {
        registry.get<Components::Transform>(entity).position = world_position(to);
    }

    status_ = std::format("{}{}{}", square_name(from), captured >= 0 ? 'x' : '-', square_name(to));
    to_move_ = to_move_ == Side::white ? Side::black : Side::white;
}

auto ChessGame::activate_cursor(Scene &scene) -> void {
    if (selected_) {
        if (std::ranges::find(targets_, cursor_) != targets_.end()) {
            move_piece(scene, *selected_, cursor_);
        } else if (cursor_ != *selected_) {
            status_ = "A pawn can't go there.";
            return;
        }

        selected_.reset();
        targets_.clear();
        return;
    }

    auto const *piece = piece_at(cursor_);

    if (piece == nullptr) {
        status_ = "Nothing to pick up there.";
    } else if (piece->side != to_move_) {
        status_ = std::format("It is {}'s move.", to_move_ == Side::white ? "white" : "black");
    } else if (piece->kind != Kind::pawn) {
        status_ = "Only pawns can move so far.";
    } else if (auto targets = pawn_targets(cursor_); targets.empty()) {
        status_ = "That pawn is blocked.";
    } else {
        selected_ = cursor_;
        targets_ = std::move(targets);
        status_.clear();
    }
}

auto ChessGame::pick_square(CursorPositionEvent const &event, float aspect_ratio) -> std::optional<Square> {
    if (!event.inside) {
        return std::nullopt;
    }

    // The cursor's ray: the near and far plane points under it, unprojected. Depth runs 0 to 1 (perspectiveLH_ZO).
    auto const inverse = glm::inverse(projection_matrix(aspect_ratio) * view_matrix());
    auto const unproject = [&](float depth) {
        auto const point = inverse * glm::vec4{static_cast<float>(event.ndc_x), static_cast<float>(event.ndc_y), depth, 1.0F};
        return glm::vec3{point} / point.w;
    };

    auto const near_point = unproject(0.0F);
    auto const direction = unproject(1.0F) - near_point;

    if (std::abs(direction.y) < 1e-6F) {
        return std::nullopt;
    }

    auto const distance = (board_top_y - near_point.y) / direction.y;

    if (distance < 0.0F) {
        return std::nullopt;
    }

    auto const hit = near_point + (direction * distance);
    auto const square = Square{static_cast<int>(std::floor((hit.x / square_size) + 4.0F)),
                               static_cast<int>(std::floor((hit.z / square_size) + 4.0F))};

    return on_board(square) ? std::optional{square} : std::nullopt;
}

auto ChessGame::on_cursor_position(Scene & /*scene*/, CursorPositionEvent const &event) -> void {
    hovered_ = pick_square(event, aspect_ratio_);

    // The keyboard cursor follows the mouse while it is over the board.
    if (hovered_) {
        cursor_ = *hovered_;
    }
}

auto ChessGame::on_mouse_button_pressed(Scene & /*scene*/, MouseButtonPressedEvent const &event) -> void {
    if (event.button == GLFW_MOUSE_BUTTON_LEFT && hovered_) {
        click_requested_ = true;
    } else if (event.button == GLFW_MOUSE_BUTTON_RIGHT) {
        cancel_requested_ = true;
    }
}

auto ChessGame::place_markers(Scene &scene) -> void {
    auto &registry = scene.get_registry();

    // The picked-up pawn is outlined.
    auto const *selected_piece = selected_ ? piece_at(*selected_) : nullptr;
    auto const wanted_outline = selected_piece != nullptr ? selected_piece->entity : entt::entity{entt::null};

    if (wanted_outline != outlined_entity_) {
        if (registry.valid(outlined_entity_)) {
            registry.remove<Components::Outlined>(outlined_entity_);
        }
        if (registry.valid(wanted_outline)) {
            registry.emplace_or_replace<Components::Outlined>(wanted_outline);
        }

        outlined_entity_ = wanted_outline;
    }

    auto const put = [&registry](entt::entity entity, std::optional<Square> square, float lift) {
        if (registry.valid(entity) && registry.all_of<Components::Transform>(entity)) {
            registry.get<Components::Transform>(entity).position =
                    square ? world_position(*square) + glm::vec3{0.0F, lift, 0.0F} : stowed;
        }
    };

    put(cursor_entity_, cursor_, 0.015F);
    put(selection_entity_, selected_, 0.005F);

    for (auto &&[index, entity]: target_entities_ | std::views::enumerate) {
        auto const slot = static_cast<std::size_t>(index);
        put(entity, slot < targets_.size() ? std::optional{targets_[slot]} : std::nullopt, 0.01F);
    }
}

auto ChessGame::on_update(Scene &scene, float /*delta_time*/) -> void {
    frames_since_update_ = 0;

    if (bound_scene_ != &scene) {
        bind_to(scene);
    }

    if (std::exchange(restart_requested_, false)) {
        restart(scene);
    }

    auto const step = std::exchange(cursor_step_, glm::ivec2{0});
    cursor_ = glm::clamp(cursor_ + step, Square{0}, Square{board_size - 1});

    if (std::exchange(cancel_requested_, false)) {
        selected_.reset();
        targets_.clear();
    }

    // A click acts on the square under the mouse, as of the latest cursor position.
    if (std::exchange(click_requested_, false) && hovered_) {
        cursor_ = *hovered_;
        activate_requested_ = true;
    }

    if (std::exchange(activate_requested_, false)) {
        activate_cursor(scene);
    }

    place_markers(scene);
}

auto ChessGame::on_key_pressed(Scene & /*scene*/, KeyPressedEvent const &event) -> void {
    switch (event.key) {
        case GLFW_KEY_UP:
        case GLFW_KEY_W:
            ++cursor_step_.y;
            break;
        case GLFW_KEY_DOWN:
        case GLFW_KEY_S:
            --cursor_step_.y;
            break;
        case GLFW_KEY_LEFT:
        case GLFW_KEY_A:
            --cursor_step_.x;
            break;
        case GLFW_KEY_RIGHT:
        case GLFW_KEY_D:
            ++cursor_step_.x;
            break;
        case GLFW_KEY_ENTER:
        case GLFW_KEY_SPACE:
            activate_requested_ = true;
            break;
        case GLFW_KEY_BACKSPACE:
            cancel_requested_ = true;
            break;
        case GLFW_KEY_R:
            // Plain R only: Ctrl+R is the editor's repopulate.
            restart_requested_ = restart_requested_ || event.modifiers == 0;
            break;
        default:
            break;
    }
}

auto ChessGame::on_ui(Scene & /*scene*/, Renderer & /*renderer*/) -> void {
    // See frames_since_update_: on_ui() runs in the editor too, where there is no game.
    if (frames_since_update_ > 2) {
        return;
    }

    ++frames_since_update_;

    gui::widget("Chess", [&] {
        ImGui::Text("%s to move", to_move_ == Side::white ? "White" : "Black");
        ImGui::Text("Cursor  %s", square_name(cursor_).c_str());

        if (!status_.empty()) {
            ImGui::TextUnformatted(status_.c_str());
        }

        ImGui::Separator();
        ImGui::TextWrapped("Click a pawn to pick it up and a green square to put it down; right click or Backspace "
                           "puts it back. The arrows or WASD and Enter or Space work too. R starts over. Only pawns "
                           "move so far.");

        if (ImGui::Button("Reset")) {
            restart_requested_ = true;
        }
    });
}

auto ChessGame::camera(Scene const & /*scene*/, float aspect_ratio) const -> CameraParams {
    aspect_ratio_ = aspect_ratio;

    return CameraParams{
            .view = view_matrix(),
            .projection = projection_matrix(aspect_ratio),
            .near_clip = camera_near_clip,
            .far_clip = camera_far_clip,
            .vertical_fov_radians = glm::radians(camera_fov_degrees),
    };
}

auto ChessGame::benchmark_camera_path() const -> std::vector<CameraKeyframe> {
    return {
            {.position = {0.0F, 8.5F, -8.5F}, .target = {0.0F, 0.0F, 0.4F}},
            {.position = {8.5F, 6.0F, 0.0F}, .target = {0.0F, 0.0F, 0.0F}},
            {.position = {0.0F, 8.5F, 8.5F}, .target = {0.0F, 0.0F, -0.4F}},
            {.position = {-8.5F, 6.0F, 0.0F}, .target = {0.0F, 0.0F, 0.0F}},
    };
}
