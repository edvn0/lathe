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
#include <glm/gtc/quaternion.hpp>
#include <glm/matrix.hpp>
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

    constexpr float loading_warmup_frames = 30.0F;
    constexpr float game_over_delay_seconds = 1.6F;
    constexpr float menu_orbit_radians_per_second = 0.12F;

    constexpr ImGuiWindowFlags overlay_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                               ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                                               ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                                               ImGuiWindowFlags_AlwaysAutoResize;

    auto begin_centred(char const *id, float background_alpha) -> bool {
        auto const *viewport = ImGui::GetMainViewport();

        ImGui::SetNextWindowPos(viewport->GetCenter(), ImGuiCond_Always, ImVec2{0.5F, 0.5F});
        ImGui::SetNextWindowBgAlpha(background_alpha);

        return ImGui::Begin(id, nullptr, overlay_flags);
    }

    auto centred_text(std::string const &text, float scale) -> void {
        ImGui::SetWindowFontScale(scale);

        auto const width = ImGui::CalcTextSize(text.c_str()).x;

        ImGui::SetCursorPosX(std::max(0.0F, (ImGui::GetWindowSize().x - width) * 0.5F));
        ImGui::TextUnformatted(text.c_str());
        ImGui::SetWindowFontScale(1.0F);
    }

    auto menu_button(char const *label) -> bool {
        ImGui::SetWindowFontScale(1.6F);

        auto const size = ImVec2{300.0F, 52.0F};

        ImGui::SetCursorPosX(std::max(0.0F, (ImGui::GetWindowSize().x - size.x) * 0.5F));

        auto const pressed = ImGui::Button(label, size);

        ImGui::SetWindowFontScale(1.0F);

        return pressed;
    }

    // An arc that chases its own tail, drawn into the current window.
    auto spinner(ImVec2 centre, float radius, float thickness) -> void {
        constexpr int segments = 32;
        constexpr float sweep = 4.8F;

        auto *draw_list = ImGui::GetWindowDrawList();
        auto const start = std::fmod(static_cast<float>(ImGui::GetTime()) * 4.0F, 6.2831853F);

        draw_list->PathClear();

        for (int index = 0; index <= segments; ++index) {
            auto const angle = start + (static_cast<float>(index) / segments) * sweep;

            draw_list->PathLineTo(ImVec2{centre.x + std::cos(angle) * radius, centre.y + std::sin(angle) * radius});
        }

        draw_list->PathStroke(ImGui::GetColorU32(ImGuiCol_PlotHistogram), ImDrawFlags_None, thickness);
    }

    struct ResultText {
        std::string title;
        std::string reason;
    };

    auto describe_result(chess::GameState state, chess::Side side_to_move) -> ResultText {
        switch (state) {
            case chess::GameState::checkmate:
                // The side to move is the one that has been mated.
                return {"Checkmate", side_to_move == chess::Side::white ? "Black wins" : "White wins"};

            case chess::GameState::stalemate:
                return {"Draw", "Stalemate"};

            case chess::GameState::draw_fifty_move:
                return {"Draw", "Fifty-move rule"};

            case chess::GameState::draw_insufficient_material:
                return {"Draw", "Insufficient material"};

            case chess::GameState::draw_repetition:
                return {"Draw", "Threefold repetition"};

            case chess::GameState::playing:
                break;
        }

        return {};
    }

    constexpr float square_size = 1.08F;
    constexpr float board_top_y = 0.3F;

    constexpr glm::vec3 world_up{0.0F, 1.0F, 0.0F};

    constexpr glm::vec3 stowed{0.0F, -50.0F, 0.0F};

    constexpr glm::vec3 camera_eye{0.0F, 10.5F, -6.5F};
    constexpr glm::vec3 camera_target{0.0F, 0.0F, 0.4F};

    constexpr float camera_fov_degrees = 40.0F;
    constexpr float camera_near_clip = 0.1F;
    constexpr float camera_far_clip = 200.0F;

    constexpr std::string_view board_model_path = "chess/chess_board.glb";

    auto model_asset(std::string_view relative) -> DataPath {
        return data_path(std::format("assets/models/{}", relative));
    }

    // The camera orbits the board centre by `angle` radians around the up axis
    // (0 looks from white's side, pi from black's) and rises while it swings.
    constexpr float camera_flip_lift = 3.0F;
    constexpr float camera_flip_rate = 4.0F;
    constexpr float camera_flip_epsilon = 1e-3F;

    [[nodiscard]] auto view_matrix(float angle) -> glm::mat4 {
        auto const spin = glm::angleAxis(angle, world_up);

        auto const lift = glm::vec3{0.0F, camera_flip_lift * std::abs(std::sin(angle)), 0.0F};

        return glm::lookAtLH((spin * camera_eye) + lift, spin * camera_target, world_up);
    }

    [[nodiscard]] auto projection_matrix(float aspect_ratio) -> glm::mat4 {
        return glm::perspectiveLH_ZO(glm::radians(camera_fov_degrees), aspect_ratio, camera_near_clip, camera_far_clip);
    }

    [[nodiscard]] auto piece_rotation(bool black) -> glm::quat {
        return black ? glm::angleAxis(glm::radians(180.0F), world_up) : glm::quat{1.0F, 0.0F, 0.0F, 0.0F};
    }

    [[nodiscard]] auto is_promotion(chess::Move const &move) noexcept -> bool {
        return move.flag == chess::MoveFlag::promotion || move.flag == chess::MoveFlag::promotion_capture;
    }

    auto run_chess_self_test(Badge<ChessGame> b) -> bool {
        chess::ChessEngine engine;

        struct PerftCase {
            int depth;
            std::uint64_t expected;
        };

        constexpr std::array cases{
                PerftCase{
                        .depth = 1,
                        .expected = 20,
                },
                PerftCase{
                        .depth = 2,
                        .expected = 400,
                },
                PerftCase{
                        .depth = 3,
                        .expected = 8'902,
                },
                PerftCase{
                        .depth = 4,
                        .expected = 197'281,
                },
                PerftCase{
                        .depth = 5,
                        .expected = 4'865'609,
                },
        };

        auto string_stream = std::ostringstream{};
        // Time taken
        auto const start_time = std::chrono::high_resolution_clock::now();

        for (auto const &[depth, expected]: cases) {
            auto const actual = engine.perft(b, depth);
            auto const end_time = std::chrono::high_resolution_clock::now();
            auto const duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

            if (actual != expected) {
                string_stream << std::format("FAIL: perft({}) expected {}, got {}\n", depth, expected, actual);
                return false;
            }

            string_stream << std::format("PASS: perft({}) expected {}, got {} ({} ms)\n", depth, expected, actual,
                                         duration);
        }

        auto total = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                           start_time)
                             .count();

        info("[Chess] Self-test passed in {} ms\n{}", total, string_stream.str());
        return true;
    }

} // namespace

auto ChessGame::on_board(Square square) noexcept -> bool {
    return square.x >= 0 && square.x < board_size && square.y >= 0 && square.y < board_size;
}

auto ChessGame::to_chess_square(Square square) noexcept -> chess::Square {
    return static_cast<chess::Square>((square.y * board_size) + square.x);
}

auto ChessGame::from_chess_square(chess::Square square) noexcept -> Square {
    auto const index = static_cast<int>(square);

    return Square{
            index & 7,
            index >> 3,
    };
}

auto ChessGame::square_name(chess::Square square) -> std::string {
    auto const board_square = from_chess_square(square);

    return std::format("{}{}", static_cast<char>('a' + board_square.x), board_square.y + 1);
}

auto ChessGame::world_position(Square square) noexcept -> glm::vec3 {
    return glm::vec3{
            (static_cast<float>(square.x) - 3.5F) * square_size,
            board_top_y,
            (static_cast<float>(square.y) - 3.5F) * square_size,
    };
}

auto ChessGame::model_index(chess::Piece piece) noexcept -> std::size_t {
    if (piece == chess::Piece::none) {
        return 0;
    }

    // Piece is ordered:
    //
    // none,
    // white pawn, knight, bishop, rook, queen, king,
    // black pawn, knight, bishop, rook, queen, king.
    return static_cast<std::size_t>(piece) - 1;
}

auto ChessGame::piece_side(chess::Piece piece) noexcept -> chess::Side {
    switch (piece) {
        case chess::Piece::white_pawn:
        case chess::Piece::white_knight:
        case chess::Piece::white_bishop:
        case chess::Piece::white_rook:
        case chess::Piece::white_queen:
        case chess::Piece::white_king:
            return chess::Side::white;

        default:
            return chess::Side::black;
    }
}

auto ChessGame::piece_is_black(chess::Piece piece) noexcept -> bool {
    return piece != chess::Piece::none && piece_side(piece) == chess::Side::black;
}

auto ChessGame::model_path(chess::Piece piece) -> std::string {
    constexpr std::array<std::string_view, 6> names{
            "pawn", "knight", "bishop", "rook", "queen", "king",
    };

    auto const index = model_index(piece);

    auto const side = index >= 6 ? "black" : "white";
    auto const kind = names[index % 6];

    return std::format("chess/{}_{}.glb", side, kind);
}

auto ChessGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    scene.get_registry().clear();

    scene.environment = new_scene_environment();
    scene.environment.sun.elevation_degrees = 55.0F;
    scene.environment.sun.azimuth_degrees = 200.0F;

    // The perft self-test takes seconds; it belongs to development, not to an installed game.
    if (!host_.player_mode) {
        auto badge = Badge<ChessGame>{};
        thread_pool().detach_task([badge = badge] { run_chess_self_test(badge); });
    }

    screen_ = host_.player_mode ? Screen::loading : Screen::playing;
    loading_peak_ = 0;
    loading_frames_ = 0;
    game_over_timer_ = 0.0F;

    bound_scene_ = nullptr;

    chess_engine_.reset();

    piece_models_.fill(ModelHandle{});

    auto const acquire_model = [&](std::string_view relative) -> ModelHandle {
        if (host_.player_mode) {
            return renderer.model_streamer().request(renderer, model_asset(relative), engine_models.cube,
                                                     FlyString{std::string{relative}});
        }

        auto loaded = renderer.load_model(model_asset(relative));

        if (!loaded) {
            error("[ChessGame] Could not load '{}': {}; run the game with the chess pack in assets/models/chess/",
                  relative, describe(loaded.error()));

            return ModelHandle{};
        }

        return *loaded;
    };

    // Board.
    {
        auto const board = Entity{&scene, "board"};

        board.emplace<Components::Transform>(Components::Transform{});

        // An installed game streams its models so the loading screen can show progress; the editor loads them up front.
        auto const model = acquire_model(board_model_path);

        if (model.valid()) {
            board.emplace<Components::Model>(Components::Model{
                    .model = model,
            });
        }
    }

    auto const model_for = [&](chess::Piece piece) -> ModelHandle {
        auto const index = model_index(piece);
        auto &slot = piece_models_[index];

        if (!slot.valid()) {
            slot = acquire_model(model_path(piece));

            if (!slot.valid()) {
                slot = engine_models.cube;
            }
        }

        return slot;
    };

    // ChessEngine owns piece placement and gives every physical piece a stable
    // ID. Build one scene entity for each of those IDs.
    auto const state = chess_engine_.render_state();

    for (std::size_t index = 0; index < state.piece_count; ++index) {
        auto const &piece = state.pieces[index];

        auto const entity = GeneratedEntity{
                &scene,
                "piece_{}",
                static_cast<std::uint32_t>(piece.id),
        };

        entity.emplace<Components::Transform>(Components::Transform{
                .position = world_position(from_chess_square(piece.square)),
                .rotation = piece_rotation(piece_is_black(piece.piece)),
        });

        entity.emplace<Components::Model>(Components::Model{
                .model = model_for(piece.piece),
        });
    }

    // Ensure every promotion model has been loaded even if the exact model
    // wasn't needed by the loop above for some future custom starting state.
    constexpr std::array all_piece_models{
            chess::Piece::white_pawn, chess::Piece::white_knight, chess::Piece::white_bishop,
            chess::Piece::white_rook, chess::Piece::white_queen,  chess::Piece::white_king,
            chess::Piece::black_pawn, chess::Piece::black_knight, chess::Piece::black_bishop,
            chess::Piece::black_rook, chess::Piece::black_queen,  chess::Piece::black_king,
    };

    for (auto const piece: all_piece_models) {
        (void) model_for(piece);
    }

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
                .scale =
                        glm::vec3{
                                square_size * footprint,
                                0.01F,
                                square_size * footprint,
                        },
        });

        entity.emplace<Components::Model>(Components::Model{
                .model = engine_models.cube,
        });

        if (material.valid()) {
            entity.emplace<Components::MaterialOverride>(Components::MaterialOverride{
                    .material = material,
            });
        }
    };

    add_marker("cursor", cursor_look, 0.9F);

    add_marker("selection", selection_look, 0.96F);

    for (std::size_t index = 0; index < target_marker_count; ++index) {
        auto const entity = GeneratedEntity{
                &scene,
                "target_{}",
                static_cast<std::uint32_t>(index),
        };

        entity.emplace<Components::Transform>(Components::Transform{
                .position = stowed,
                .scale =
                        glm::vec3{
                                square_size * 0.7F,
                                0.01F,
                                square_size * 0.7F,
                        },
        });

        entity.emplace<Components::Model>(Components::Model{
                .model = engine_models.cube,
        });

        if (target_look.valid()) {
            entity.emplace<Components::MaterialOverride>(Components::MaterialOverride{
                    .material = target_look,
            });
        }
    }

    for (auto const material: created_materials) {
        renderer.release_material(material);
    }

    info("[ChessGame] Set up the board");
}

auto ChessGame::bind_to(Scene &scene) -> void {
    bound_scene_ = &scene;

    piece_entities_.fill(entt::null);
    target_entities_.fill(entt::null);

    outlined_entity_ = entt::null;

    for (std::size_t id = 0; id < piece_entities_.size(); ++id) {
        piece_entities_[id] = scene.find_entity(std::format("piece_{}", id));
    }

    cursor_entity_ = scene.find_entity("cursor");

    selection_entity_ = scene.find_entity("selection");

    for (std::size_t index = 0; index < target_entities_.size(); ++index) {
        target_entities_[index] = scene.find_entity(std::format("target_{}", index));
    }

    restart_requested_ = true;
}

auto ChessGame::clear_selection() -> void {
    selected_.reset();
    selected_moves_.clear();
    target_squares_.clear();

    pending_promotion_target_.reset();
    promotion_requested_.reset();
}

auto ChessGame::restart(Scene &scene) -> void {
    chess_engine_.reset();

    camera_side_ = chess_engine_.side_to_move();
    camera_angle_ = 0.0F;
    camera_target_angle_ = 0.0F;

    cursor_ = Square{4, 1};
    hovered_.reset();

    clear_selection();

    status_.clear();

    sync_pieces(scene);
}

auto ChessGame::sync_pieces(Scene &scene) -> void {
    auto &registry = scene.get_registry();

    auto const state = chess_engine_.render_state();

    std::array<bool, piece_count> visible{};

    for (std::size_t index = 0; index < state.piece_count; ++index) {
        auto const &piece = state.pieces[index];

        if (piece.id >= piece_entities_.size()) {
            continue;
        }

        visible[piece.id] = true;

        auto const entity = piece_entities_[piece.id];

        if (!registry.valid(entity)) {
            continue;
        }

        if (registry.all_of<Components::Transform>(entity)) {
            auto &transform = registry.get<Components::Transform>(entity);

            transform.position = world_position(from_chess_square(piece.square));

            transform.rotation = piece_rotation(piece_is_black(piece.piece));
        }

        if (registry.all_of<Components::Model>(entity)) {
            registry.get<Components::Model>(entity).model = piece_models_[model_index(piece.piece)];
        }
    }

    // Anything omitted by RenderState has been captured.
    for (std::size_t id = 0; id < piece_entities_.size(); ++id) {
        if (visible[id]) {
            continue;
        }

        auto const entity = piece_entities_[id];

        if (!registry.valid(entity) || !registry.all_of<Components::Transform>(entity)) {
            continue;
        }

        registry.get<Components::Transform>(entity).position = stowed;
    }
}

auto ChessGame::rebuild_target_squares() -> void {
    target_squares_.clear();

    for (auto const &move: selected_moves_) {
        if (std::ranges::find(target_squares_, move.to) != target_squares_.end()) {
            continue;
        }

        target_squares_.push_back(move.to);
    }
}

auto ChessGame::execute_move(Scene &scene, chess::Move move) -> void {
    auto const from_name = square_name(move.from);

    auto const to_name = square_name(move.to);

    auto const result = chess_engine_.update(move);

    if (!result.moved) {
        status_ = "That move is not legal.";
        return;
    }

    status_ = std::format("{}{}{}", from_name, result.capture ? "x" : "-", to_name);

    switch (result.game_state) {
        using enum chess::GameState;
        case checkmate:
            status_ += " checkmate.";
            break;

        case stalemate:
            status_ += " stalemate.";
            break;

        case draw_fifty_move:
            status_ += " draw (fifty-move rule).";
            break;

        case draw_insufficient_material:
            status_ += " draw (insufficient material).";
            break;

        case draw_repetition:
            status_ += " draw (threefold repetition).";
            break;

        case playing:
            if (result.check) {
                status_ += " check.";
            }

            break;
    }

    clear_selection();

    sync_pieces(scene);

    if (host_.player_mode && chess::is_game_over(result.game_state)) {
        game_over_timer_ = game_over_delay_seconds;
    }
}

auto ChessGame::activate_cursor(Scene &scene) -> void {
    // Promotion must be resolved before another board action is accepted.
    if (pending_promotion_target_) {
        status_ = "Choose a promotion piece.";
        return;
    }

    auto const square = to_chess_square(cursor_);

    if (selected_) {
        // Clicking the selected piece again puts it back.
        if (square == *selected_) {
            clear_selection();
            status_.clear();

            return;
        }

        std::array<chess::Move, 4> matches{};
        std::size_t match_count = 0;

        for (auto const &move: selected_moves_) {
            if (move.to != square) {
                continue;
            }

            if (match_count < matches.size()) {
                matches[match_count++] = move;
            }
        }

        if (match_count == 0) {
            status_ = "That piece cannot go there.";
            return;
        }

        if (match_count == 1 && !is_promotion(matches[0])) {
            execute_move(scene, matches[0]);

            return;
        }

        // Promotions have four legal moves with identical from/to.
        pending_promotion_target_ = square;
        status_ = "Choose promotion: queen, rook, bishop or knight.";

        return;
    }

    auto const piece = chess_engine_.piece_at(square);

    if (piece == chess::Piece::none) {
        status_ = "Nothing to pick up there.";
        return;
    }

    if (piece_side(piece) != chess_engine_.side_to_move()) {
        status_ =
                std::format("It is {}'s move.", chess_engine_.side_to_move() == chess::Side::white ? "white" : "black");

        return;
    }

    auto moves = chess_engine_.legal_moves(square);

    if (moves.empty()) {
        status_ = "That piece has no legal moves.";
        return;
    }

    selected_ = square;
    selected_moves_ = std::move(moves);

    rebuild_target_squares();

    status_.clear();
}

auto ChessGame::pick_square(CursorPositionEvent const &event, float aspect_ratio) const -> std::optional<Square> {
    if (!event.inside) {
        return std::nullopt;
    }

    auto const inverse = glm::inverse(projection_matrix(aspect_ratio) * view_matrix(camera_angle_));

    auto const unproject = [&](float depth) {
        auto const point = inverse * glm::vec4{
                                             static_cast<float>(event.ndc_x),
                                             static_cast<float>(event.ndc_y),
                                             depth,
                                             1.0F,
                                     };

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

    auto const square = Square{
            static_cast<int>(std::floor((hit.x / square_size) + 4.0F)),
            static_cast<int>(std::floor((hit.z / square_size) + 4.0F)),
    };

    return on_board(square) ? std::optional{square} : std::nullopt;
}

auto ChessGame::on_cursor_position(Scene & /*scene*/, CursorPositionEvent const &event) -> void {
    hovered_ = pick_square(event, aspect_ratio_);

    if (hovered_) {
        cursor_ = *hovered_;
    }
}

auto ChessGame::on_mouse_button_pressed(Scene & /*scene*/, MouseButtonPressedEvent const &event) -> void {
    if (host_.player_mode && screen_ != Screen::playing) {
        return;
    }

    if (event.button == GLFW_MOUSE_BUTTON_LEFT && hovered_) {
        click_requested_ = true;
    } else if (event.button == GLFW_MOUSE_BUTTON_RIGHT) {
        cancel_requested_ = true;
    }
}

auto ChessGame::place_markers(Scene &scene) -> void {
    auto &registry = scene.get_registry();

    entt::entity wanted_outline = entt::null;

    auto const state = chess_engine_.render_state();

    for (std::size_t index = 0; index < state.piece_count; ++index) {
        auto const &piece = state.pieces[index];

        if (selected_ && piece.square == *selected_) {
            wanted_outline = piece_entities_[piece.id];
            break;
        }
    }

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
        if (!registry.valid(entity) || !registry.all_of<Components::Transform>(entity)) {
            return;
        }

        registry.get<Components::Transform>(entity).position = square ? world_position(*square) +
                                                                                glm::vec3{
                                                                                        0.0F,
                                                                                        lift,
                                                                                        0.0F,
                                                                                }
                                                                      : stowed;
    };

    put(cursor_entity_, cursor_, 0.015F);

    put(selection_entity_, selected_ ? std::optional{from_chess_square(*selected_)} : std::nullopt, 0.005F);

    for (std::size_t index = 0; index < target_entities_.size(); ++index) {
        auto const square = index < target_squares_.size() ? std::optional{from_chess_square(target_squares_[index])}
                                                           : std::nullopt;

        put(target_entities_[index], square, 0.01F);
    }
}

auto ChessGame::update_camera(float delta_time) -> void {
    auto const side = chess_engine_.side_to_move();

    if (side != camera_side_) {
        camera_side_ = side;
        camera_target_angle_ += glm::radians(180.0F);
    }

    // Exponential approach: fast at first, settling smoothly.
    auto const remaining = camera_target_angle_ - camera_angle_;

    if (std::abs(remaining) < camera_flip_epsilon) {
        camera_angle_ = camera_target_angle_;
        return;
    }

    camera_angle_ += remaining * (1.0F - std::exp(-camera_flip_rate * std::min(delta_time, 0.1F)));
}

auto ChessGame::on_update(Scene &scene, float delta_time) -> void {
    frames_since_update_ = 0;

    if (bound_scene_ != &scene) {
        bind_to(scene);
    }

    if (host_.player_mode && screen_ != Screen::playing) {
        // Behind the menus the board idles: it turns slowly on the menu and holds still elsewhere.
        if (screen_ == Screen::loading || screen_ == Screen::menu) {
            camera_angle_ += menu_orbit_radians_per_second * std::min(delta_time, 0.1F);
            camera_target_angle_ = camera_angle_;
        }

        return;
    }

    if (std::exchange(restart_requested_, false)) {
        restart(scene);
    }

    if (game_over_timer_ > 0.0F) {
        game_over_timer_ -= delta_time;

        if (game_over_timer_ <= 0.0F) {
            game_over_timer_ = 0.0F;
            screen_ = Screen::game_over;
            clear_selection();
        }
    }

    update_camera(delta_time);

    auto step = std::exchange(cursor_step_, glm::ivec2{0});

    // Keys move the cursor as seen on screen, so mirror them from black's side.
    if (std::cos(camera_target_angle_) < 0.0F) {
        step = -step;
    }

    cursor_ = glm::clamp(cursor_ + step, Square{0}, Square{board_size - 1});

    if (std::exchange(cancel_requested_, false)) {
        clear_selection();
        status_.clear();
    }

    if (std::exchange(click_requested_, false) && hovered_) {
        cursor_ = *hovered_;
        activate_requested_ = true;
    }

    if (std::exchange(activate_requested_, false)) {
        activate_cursor(scene);
    }

    if (promotion_requested_ && pending_promotion_target_ && selected_) {
        auto const promotion = std::exchange(promotion_requested_, std::nullopt);

        auto const target = *pending_promotion_target_;

        auto const it = std::ranges::find_if(selected_moves_, [&](chess::Move const &move) {
            return move.to == target && is_promotion(move) && move.promotion == *promotion;
        });

        if (it != selected_moves_.end()) {
            auto const move = *it;

            execute_move(scene, move);
        } else {
            pending_promotion_target_.reset();
            status_ = "Could not promote to that piece.";
        }
    }

    place_markers(scene);
}

auto ChessGame::on_key_pressed(Scene & /*scene*/, KeyPressedEvent const &event) -> void {
    if (host_.player_mode) {
        switch (screen_) {
            case Screen::loading:
                return;

            case Screen::menu:
            case Screen::game_over:
                if (event.key == GLFW_KEY_ENTER || event.key == GLFW_KEY_SPACE) {
                    start_new_game();
                }

                return;

            case Screen::paused:
                if (event.key == GLFW_KEY_ESCAPE) {
                    screen_ = Screen::playing;
                }

                return;

            case Screen::playing:
                if (event.key == GLFW_KEY_ESCAPE) {
                    screen_ = Screen::paused;
                    return;
                }

                break;
        }
    }

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

        case GLFW_KEY_Q:
            if (pending_promotion_target_) {
                promotion_requested_ = chess::PieceType::queen;
            }
            break;

        case GLFW_KEY_T:
            if (pending_promotion_target_) {
                promotion_requested_ = chess::PieceType::rook;
            }
            break;

        case GLFW_KEY_B:
            if (pending_promotion_target_) {
                promotion_requested_ = chess::PieceType::bishop;
            }
            break;

        case GLFW_KEY_N:
            if (pending_promotion_target_) {
                promotion_requested_ = chess::PieceType::knight;
            }
            break;

        case GLFW_KEY_R:
            // Plain R only: Ctrl+R is the editor's repopulate.
            restart_requested_ = restart_requested_ || event.modifiers == 0;
            break;

        default:
            break;
    }
}

auto ChessGame::on_ui(Scene & /*scene*/, Renderer &renderer) -> void {
    if (frames_since_update_ > 2) {
        return;
    }

    ++frames_since_update_;

    if (host_.player_mode) {
        draw_player_ui(renderer);
        return;
    }

    gui::widget("Chess", [&] -> void {
        auto const side = chess_engine_.side_to_move();

        ImGui::Text("%s to move", side == chess::Side::white ? "White" : "Black");

        ImGui::Text("Cursor  %s", square_name(to_chess_square(cursor_)).c_str());

        if (!status_.empty()) {
            ImGui::TextUnformatted(status_.c_str());
        }

        draw_promotion_choice();

        ImGui::Separator();

        switch (chess_engine_.game_state()) {
            case chess::GameState::checkmate:
                ImGui::TextUnformatted("Game over: checkmate.");
                break;

            case chess::GameState::stalemate:
                ImGui::TextUnformatted("Game over: stalemate.");
                break;

            case chess::GameState::draw_fifty_move:
            case chess::GameState::draw_insufficient_material:
            case chess::GameState::draw_repetition:
                ImGui::Text("Game over: draw (%s).",
                            describe_result(chess_engine_.game_state(), side).reason.c_str());
                break;

            case chess::GameState::playing:
                ImGui::TextWrapped("Click a piece to select it and a green square "
                                   "to move. Right click or Backspace cancels the "
                                   "selection. Arrows or WASD move the cursor; "
                                   "Enter or Space selects or moves. R starts over.");
                break;
        }

        if (ImGui::Button("Reset")) {
            restart_requested_ = true;
        }
    });
}

auto ChessGame::camera(Scene const & /*scene*/, float aspect_ratio) const -> CameraParams {
    aspect_ratio_ = aspect_ratio;

    return CameraParams{
            .view = view_matrix(camera_angle_),
            .projection = projection_matrix(aspect_ratio),
            .near_clip = camera_near_clip,
            .far_clip = camera_far_clip,
            .vertical_fov_radians = glm::radians(camera_fov_degrees),
    };
}

auto ChessGame::benchmark_camera_path() const -> std::vector<CameraKeyframe> {
    return {
            {
                    .position = {0.0F, 8.5F, -8.5F},
                    .target = {0.0F, 0.0F, 0.4F},
            },
            {
                    .position = {8.5F, 6.0F, 0.0F},
                    .target = {0.0F, 0.0F, 0.0F},
            },
            {
                    .position = {0.0F, 8.5F, 8.5F},
                    .target = {0.0F, 0.0F, -0.4F},
            },
            {
                    .position = {-8.5F, 6.0F, 0.0F},
                    .target = {0.0F, 0.0F, 0.0F},
            },
    };
}

auto ChessGame::start_new_game() -> void {
    restart_requested_ = true;
    game_over_timer_ = 0.0F;
    screen_ = Screen::playing;
}

auto ChessGame::draw_promotion_choice() -> void {
    if (!pending_promotion_target_) {
        return;
    }

    ImGui::Separator();

    ImGui::TextUnformatted("Promote pawn to:");

    if (ImGui::Button("Queen")) {
        promotion_requested_ = chess::PieceType::queen;
    }

    ImGui::SameLine();

    if (ImGui::Button("Rook")) {
        promotion_requested_ = chess::PieceType::rook;
    }

    ImGui::SameLine();

    if (ImGui::Button("Bishop")) {
        promotion_requested_ = chess::PieceType::bishop;
    }

    ImGui::SameLine();

    if (ImGui::Button("Knight")) {
        promotion_requested_ = chess::PieceType::knight;
    }
}

auto ChessGame::draw_player_ui(Renderer &renderer) -> void {
    switch (screen_) {
        case Screen::loading:
            draw_loading(renderer);
            break;

        case Screen::menu:
            draw_menu();
            break;

        case Screen::paused:
            draw_pause();
            break;

        case Screen::game_over:
            draw_game_over();
            break;

        case Screen::playing:
            draw_hud();
            break;
    }
}

auto ChessGame::draw_loading(Renderer &renderer) -> void {
    auto const pending = renderer.model_streamer().pending_count() + renderer.texture_streamer().pending_count();

    loading_peak_ = std::max(loading_peak_, pending);
    ++loading_frames_;

    auto const streaming_progress =
            loading_peak_ == 0 ? 0.0F : 1.0F - static_cast<float>(pending) / static_cast<float>(loading_peak_);
    auto const warmup_progress = std::min(1.0F, static_cast<float>(loading_frames_) / loading_warmup_frames);

    // Models first; the last frames are the pipelines and the driver settling on the finished board.
    auto const progress = pending == 0 ? 0.5F + 0.5F * warmup_progress : 0.5F * streaming_progress;

    if (pending == 0 && warmup_progress >= 1.0F) {
        screen_ = Screen::menu;
        return;
    }

    auto const *viewport = ImGui::GetMainViewport();

    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowBgAlpha(1.0F);

    ImGui::Begin("##loading", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                         ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs);

    auto const centre = viewport->GetCenter();

    spinner(ImVec2{centre.x, centre.y - 24.0F}, 28.0F, 5.0F);

    ImGui::SetCursorScreenPos(ImVec2{centre.x - 150.0F, centre.y + 44.0F});
    ImGui::ProgressBar(progress, ImVec2{300.0F, 6.0F}, "");

    ImGui::SetCursorScreenPos(ImVec2{centre.x - 40.0F, centre.y + 62.0F});
    ImGui::SetWindowFontScale(1.4F);
    ImGui::TextUnformatted("Loading...");
    ImGui::SetWindowFontScale(1.0F);

    ImGui::End();
}

auto ChessGame::draw_menu() -> void {
    begin_centred("##chess_menu", 0.8F);

    ImGui::Dummy(ImVec2{340.0F, 8.0F});
    centred_text("Chess", 4.0F);
    ImGui::Dummy(ImVec2{0.0F, 18.0F});

    if (menu_button("Play")) {
        start_new_game();
    }

    ImGui::Dummy(ImVec2{0.0F, 4.0F});

    if (menu_button("Quit") && host_.request_exit) {
        host_.request_exit();
    }

    ImGui::Dummy(ImVec2{0.0F, 8.0F});

    ImGui::End();
}

auto ChessGame::draw_pause() -> void {
    begin_centred("##chess_pause", 0.85F);

    ImGui::Dummy(ImVec2{340.0F, 8.0F});
    centred_text("Paused", 3.0F);
    ImGui::Dummy(ImVec2{0.0F, 18.0F});

    if (menu_button("Resume")) {
        screen_ = Screen::playing;
    }

    ImGui::Dummy(ImVec2{0.0F, 4.0F});

    if (menu_button("Restart")) {
        start_new_game();
    }

    ImGui::Dummy(ImVec2{0.0F, 4.0F});

    if (menu_button("Main menu")) {
        screen_ = Screen::menu;
    }

    ImGui::Dummy(ImVec2{0.0F, 4.0F});

    if (menu_button("Quit") && host_.request_exit) {
        host_.request_exit();
    }

    ImGui::Dummy(ImVec2{0.0F, 8.0F});

    ImGui::End();
}

auto ChessGame::draw_game_over() -> void {
    auto const result = describe_result(chess_engine_.game_state(), chess_engine_.side_to_move());

    begin_centred("##chess_game_over", 0.88F);

    ImGui::Dummy(ImVec2{340.0F, 8.0F});
    centred_text(result.title, 3.5F);
    centred_text(result.reason, 2.0F);
    ImGui::Dummy(ImVec2{0.0F, 18.0F});

    if (menu_button("Play again")) {
        start_new_game();
    }

    ImGui::Dummy(ImVec2{0.0F, 4.0F});

    if (menu_button("Main menu")) {
        screen_ = Screen::menu;
    }

    ImGui::Dummy(ImVec2{0.0F, 4.0F});

    if (menu_button("Quit") && host_.request_exit) {
        host_.request_exit();
    }

    ImGui::Dummy(ImVec2{0.0F, 8.0F});

    ImGui::End();
}

auto ChessGame::draw_hud() -> void {
    auto const *viewport = ImGui::GetMainViewport();

    ImGui::SetNextWindowPos(ImVec2{viewport->WorkPos.x + 16.0F, viewport->WorkPos.y + 16.0F});
    ImGui::SetNextWindowBgAlpha(0.45F);

    ImGui::Begin("##chess_hud", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                         ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize |
                         ImGuiWindowFlags_NoFocusOnAppearing);

    ImGui::SetWindowFontScale(1.6F);
    ImGui::TextUnformatted(chess_engine_.side_to_move() == chess::Side::white ? "White to move" : "Black to move");

    if (!status_.empty()) {
        ImGui::TextUnformatted(status_.c_str());
    }

    ImGui::SetWindowFontScale(1.0F);

    draw_promotion_choice();

    ImGui::TextDisabled("Esc: menu");

    ImGui::End();

    draw_state_banner();
}

auto ChessGame::draw_state_banner() -> void {
    auto const state = chess_engine_.game_state();
    auto const in_check = chess_engine_.render_state().in_check;

    if (state == chess::GameState::playing && !in_check) {
        return;
    }

    auto const result = describe_result(state, chess_engine_.side_to_move());
    auto const is_mate = state == chess::GameState::checkmate;

    std::string text;
    ImVec4 colour{1.0F, 0.75F, 0.2F, 1.0F};

    if (is_mate) {
        text = std::format("CHECKMATE - {}", result.reason);
        colour = ImVec4{1.0F, 0.3F, 0.3F, 1.0F};
    } else if (state != chess::GameState::playing) {
        text = std::format("DRAW - {}", result.reason);
    } else {
        text = std::format("{} IN CHECK", chess_engine_.side_to_move() == chess::Side::white ? "WHITE" : "BLACK");
    }

    auto const *viewport = ImGui::GetMainViewport();

    ImGui::SetNextWindowPos(ImVec2{viewport->WorkPos.x + viewport->WorkSize.x * 0.5F, viewport->WorkPos.y + 16.0F},
                            ImGuiCond_Always, ImVec2{0.5F, 0.0F});
    ImGui::SetNextWindowBgAlpha(0.7F);

    ImGui::Begin("##chess_state_banner", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                         ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize |
                         ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoInputs);

    ImGui::SetWindowFontScale(2.4F);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::SetWindowFontScale(1.0F);

    ImGui::End();
}
