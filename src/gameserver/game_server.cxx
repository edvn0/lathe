#include "gameserver/game_server.hxx"

#include <algorithm>
#include <array>
#include <format>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace gameserver {

    namespace {
        auto make_token() -> std::string {
            std::random_device device;

            auto const draw = [&device] { return (static_cast<std::uint64_t>(device()) << 32U) | device(); };

            return std::format("{:016x}{:016x}", draw(), draw());
        }
    }

    namespace {
        // Runs on a worker: reads the game, so it must only be called from the room's strand.
        auto render_states(IServerGame const &game, RoomId room_id, std::span<PlayerId const> players)
                -> std::vector<std::pair<PlayerId, std::string>> {
            std::vector<std::pair<PlayerId, std::string>> states;

            states.reserve(players.size());

            for (auto const player : players) {
                JsonWriter writer;

                writer.begin_object({}, true);
                writer.value("type", "state");
                writer.value("room", room_id);
                writer.begin_object("state", true);
                game.write_state(player, writer);
                writer.end_object();
                writer.end_object();

                states.emplace_back(player, writer.str());
            }

            return states;
        }
    }

    GameServer::GameServer(SendFunction send, Clock::duration reconnect_grace, std::size_t worker_threads)
        : send_(std::move(send)), reconnect_grace_(reconnect_grace) {
        auto const count = worker_threads != 0 ? worker_threads : std::max(1U, std::thread::hardware_concurrency());

        for (std::size_t index = 0; index < count; ++index) {
            workers_.emplace_back([this](std::stop_token const &stop) { worker_loop(stop); });
        }
    }

    GameServer::~GameServer() = default;

    auto GameServer::register_game(std::string name, GameFactory factory) -> void {
        std::scoped_lock const lock{mutex_};

        games_.insert_or_assign(std::move(name), std::move(factory));
    }

    auto GameServer::on_connect(ConnectionId connection) -> void {
        std::scoped_lock const lock{mutex_};

        if (connections_.contains(connection)) {
            return;
        }

        auto const id = next_player_++;

        auto &player = players_[id];

        player.id = id;
        player.token = make_token();
        player.connection = connection;

        tokens_[player.token] = id;
        connections_[connection] = id;

        send_welcome(player);
    }

    auto GameServer::on_disconnect(ConnectionId connection, Clock::time_point now) -> void {
        std::scoped_lock const lock{mutex_};

        auto const entry = connections_.find(connection);

        if (entry == connections_.end()) {
            return;
        }

        auto const id = entry->second;

        connections_.erase(entry);

        auto &player = players_.at(id);

        player.connection.reset();

        if (player.room) {
            auto const room_id = *player.room;

            auto const &room = *rooms_.at(room_id);

            if (room.started) {
                player.disconnected_at = now;

                notify_others(room, room_id, id, "opponent_disconnected");
                return;
            }
        }

        leave_room(player, "player_left");

        forget(id);
    }

    auto GameServer::on_message(ConnectionId connection, std::string_view text) -> void {
        std::scoped_lock const lock{mutex_};

        auto const entry = connections_.find(connection);

        if (entry == connections_.end()) {
            return;
        }

        auto const parsed = parse_json(text);

        if (!parsed || !parsed->is_object()) {
            send_error(connection, "bad_json", "message is not a JSON object");
            return;
        }

        auto const type = (*parsed)["type"].as_string();

        if (type == "resume") {
            handle_resume(connection, *parsed);
            return;
        }

        auto &player = players_.at(entry->second);

        if (type == "create_room") {
            handle_create_room(player, *parsed);
        } else if (type == "join_room") {
            handle_join_room(player, *parsed);
        } else if (type == "action") {
            handle_action(player, *parsed);
        } else if (type == "leave_room") {
            if (!player.room) {
                send_error(connection, "not_in_room", "you are not in a room");
                return;
            }

            auto const room_id = *player.room;

            leave_room(player, "player_left");

            JsonWriter writer;

            writer.begin_object({}, true);
            writer.value("type", "left");
            writer.value("room", room_id);
            writer.end_object();

            send_to(player.id, writer.str());
        } else {
            send_error(connection, "unknown_type", "unknown message type");
        }
    }

    auto GameServer::wait_idle() -> void {
        std::unique_lock lock{mutex_};

        idle_.wait(lock, [this] { return pending_jobs_ == 0; });
    }

    auto GameServer::on_tick(Clock::time_point now) -> void {
        std::scoped_lock const lock{mutex_};

        std::vector<PlayerId> expired;

        for (auto const &[id, player] : players_) {
            if (player.disconnected_at && now - *player.disconnected_at >= reconnect_grace_) {
                expired.push_back(id);
            }
        }

        for (auto const id : expired) {
            auto const player = players_.find(id);

            // A room closed for an earlier expiry already forgot the other disconnected seats.
            if (player == players_.end()) {
                continue;
            }

            if (player->second.room) {
                close_room(*player->second.room, "player_timeout");
            } else {
                forget(id);
            }
        }
    }

    auto GameServer::handle_resume(ConnectionId connection, JsonValue const &message) -> void {
        auto const target = tokens_.find(message["token"].as_string());

        if (target == tokens_.end()) {
            send_error(connection, "bad_token", "unknown or expired session token");
            return;
        }

        auto const fresh_id = connections_.at(connection);

        auto const old_id = target->second;

        if (old_id == fresh_id) {
            send_error(connection, "bad_request", "this connection already holds that session");
            return;
        }

        auto &old_player = players_.at(old_id);

        if (old_player.connection) {
            send_error(connection, "session_in_use", "that session is connected elsewhere");
            return;
        }

        if (players_.at(fresh_id).room) {
            send_error(connection, "already_in_room", "leave the current room first");
            return;
        }

        forget(fresh_id);

        connections_[connection] = old_id;

        old_player.connection = connection;
        old_player.disconnected_at.reset();

        send_welcome(old_player);

        if (!old_player.room) {
            return;
        }

        auto const room_id = *old_player.room;

        auto const room = rooms_.at(room_id);

        send_joined(old_player);

        if (room->started) {
            render_for(room_id, room, {old_id});
        }

        notify_others(*room, room_id, old_id, "opponent_reconnected");
    }

    auto GameServer::handle_create_room(Player &player, JsonValue const &message) -> void {
        auto const connection = *player.connection;

        if (player.room) {
            send_error(connection, "already_in_room", "leave the current room first");
            return;
        }

        auto const game_name = message["game"].as_string();

        auto const factory = games_.find(game_name);

        if (factory == games_.end()) {
            send_error(connection, "unknown_game", "no such game is registered");
            return;
        }

        auto const room_id = next_room_++;

        auto room = std::make_shared<Room>();

        room->game_name = game_name;
        room->game = factory->second();

        rooms_.emplace(room_id, std::move(room));

        seat_player(player, room_id);
    }

    auto GameServer::handle_join_room(Player &player, JsonValue const &message) -> void {
        auto const connection = *player.connection;

        if (player.room) {
            send_error(connection, "already_in_room", "leave the current room first");
            return;
        }

        auto const &room_value = message["room"];

        if (!room_value.is_number()) {
            send_error(connection, "bad_request", "join_room needs a numeric room");
            return;
        }

        auto const room_id = static_cast<RoomId>(room_value.as_number());

        auto const room = rooms_.find(room_id);

        if (room == rooms_.end()) {
            send_error(connection, "no_such_room", "room does not exist");
            return;
        }

        if (room->second->started) {
            send_error(connection, "room_full", "game already started");
            return;
        }

        seat_player(player, room_id);
    }

    auto GameServer::handle_action(Player &player, JsonValue const &message) -> void {
        auto const connection = *player.connection;

        if (!player.room) {
            send_error(connection, "not_in_room", "join a room first");
            return;
        }

        auto const room_id = *player.room;

        auto const room = rooms_.at(room_id);

        if (!room->started) {
            send_error(connection, "not_started", "waiting for more players");
            return;
        }

        auto const result = std::make_shared<ActionResult>();

        submit(
                room,
                [game = room->game, player_id = player.id, action = message["action"], seats = room->seats, room_id,
                 result] {
                    auto const applied = game->apply_action(player_id, action);

                    if (!applied) {
                        result->error = applied.error();
                        return;
                    }

                    result->states = render_states(*game, room_id, seats);
                    result->outcome = game->outcome();
                },
                [this, room, room_id, player_id = player.id, result] {
                    finish_action(room_id, *room, player_id, *result);
                });
    }

    auto GameServer::finish_action(RoomId room_id, Room const &room, PlayerId player, ActionResult const &result)
            -> void {
        if (result.error) {
            reject(player, result.error->code, result.error->message);
            return;
        }

        deliver(result.states);

        if (!result.outcome) {
            return;
        }

        JsonWriter writer;

        writer.begin_object({}, true);
        writer.value("type", "game_over");
        writer.value("room", room_id);

        if (result.outcome->winner) {
            writer.value("winner", static_cast<std::uint64_t>(*result.outcome->winner));
        } else {
            writer.null("winner");
        }

        writer.value("reason", result.outcome->reason);
        writer.end_object();

        for (auto const seat : room.seats) {
            send_to(seat, writer.str());
        }

        close_room(room_id, {});
    }

    auto GameServer::seat_player(Player &player, RoomId room_id) -> void {
        auto const room = rooms_.at(room_id);

        player.room = room_id;

        room->seats.push_back(player.id);

        send_joined(player);

        if (room->seats.size() < room->game->player_count()) {
            return;
        }

        room->started = true;

        auto const states = std::make_shared<RenderedStates>();

        submit(
                room,
                [game = room->game, seats = room->seats, room_id, states] {
                    game->start(seats);

                    *states = render_states(*game, room_id, seats);
                },
                [this, states] { deliver(*states); });
    }

    auto GameServer::send_welcome(Player const &player) -> void {
        JsonWriter writer;

        writer.begin_object({}, true);
        writer.value("type", "welcome");
        writer.value("player", static_cast<std::uint64_t>(player.id));
        writer.value("token", player.token);
        writer.end_object();

        send_to(player.id, writer.str());
    }

    auto GameServer::send_joined(Player const &player) -> void {
        auto const &room = *rooms_.at(*player.room);

        auto const seat = std::ranges::find(room.seats, player.id);

        JsonWriter writer;

        writer.begin_object({}, true);
        writer.value("type", "joined");
        writer.value("room", *player.room);
        writer.value("game", room.game_name);
        writer.value("player", static_cast<std::uint64_t>(player.id));
        writer.value("seat", static_cast<std::uint64_t>(seat - room.seats.begin()));
        writer.end_object();

        send_to(player.id, writer.str());
    }

    // Takes `player` out of their room. A started room cannot go on without them, so it closes for everyone else.
    auto GameServer::leave_room(Player &player, std::string_view reason) -> void {
        if (!player.room) {
            return;
        }

        auto const room_id = *player.room;

        auto &room = *rooms_.at(room_id);

        std::erase(room.seats, player.id);

        player.room.reset();

        if (room.started) {
            close_room(room_id, reason);
        } else if (room.seats.empty()) {
            room.closed = true;

            rooms_.erase(room_id);
        }
    }

    // An empty reason closes the room without telling the seats (the caller already did).
    auto GameServer::close_room(RoomId room_id, std::string_view reason) -> void {
        auto const room = rooms_.find(room_id);

        if (room == rooms_.end()) {
            return;
        }

        // Queued jobs for this room are dropped by its strand; the strand keeps the room object alive until then.
        room->second->closed = true;

        auto const seats = room->second->seats;

        rooms_.erase(room);

        for (auto const seat : seats) {
            auto const player = players_.find(seat);

            if (player == players_.end()) {
                continue;
            }

            player->second.room.reset();

            // A seat that was only being held for a reconnect has nothing left to come back to.
            if (!player->second.connection) {
                forget(seat);
                continue;
            }

            if (reason.empty()) {
                continue;
            }

            JsonWriter writer;

            writer.begin_object({}, true);
            writer.value("type", "room_closed");
            writer.value("room", room_id);
            writer.value("reason", reason);
            writer.end_object();

            send_to(seat, writer.str());
        }
    }

    auto GameServer::forget(PlayerId player) -> void {
        auto const entry = players_.find(player);

        if (entry == players_.end()) {
            return;
        }

        tokens_.erase(entry->second.token);
        players_.erase(entry);
    }

    // Queues a render of `players`' views on the room's strand, so it sees the game between actions, never mid-action.
    auto GameServer::render_for(RoomId room_id, std::shared_ptr<Room> const &room, std::vector<PlayerId> players)
            -> void {
        auto const states = std::make_shared<RenderedStates>();

        submit(
                room,
                [game = room->game, room_id, players = std::move(players), states] {
                    *states = render_states(*game, room_id, players);
                },
                [this, states] { deliver(*states); });
    }

    auto GameServer::deliver(RenderedStates const &states) -> void {
        for (auto const &[player, text] : states) {
            send_to(player, text);
        }
    }

    auto GameServer::submit(std::shared_ptr<Room> const &room, std::function<void()> work, std::function<void()> done)
            -> void {
        ++pending_jobs_;

        room->jobs.push_back(Job{.work = std::move(work), .done = std::move(done)});

        if (room->running) {
            return;
        }

        room->running = true;

        enqueue([this, room] { run_strand(room); });
    }

    // Drains one room's jobs in order. Only one strand per room is ever queued or running.
    auto GameServer::run_strand(std::shared_ptr<Room> const &room) -> void {
        for (;;) {
            Job job;

            {
                std::scoped_lock const lock{mutex_};

                if (room->jobs.empty()) {
                    room->running = false;
                    return;
                }

                job = std::move(room->jobs.front());

                room->jobs.pop_front();

                if (room->closed) {
                    finish_job();
                    continue;
                }
            }

            job.work();

            std::scoped_lock const lock{mutex_};

            if (!room->closed) {
                job.done();
            }

            finish_job();
        }
    }

    auto GameServer::finish_job() -> void {
        if (--pending_jobs_ == 0) {
            idle_.notify_all();
        }
    }

    auto GameServer::enqueue(std::function<void()> task) -> void {
        {
            std::scoped_lock const lock{queue_mutex_};

            queue_.push_back(std::move(task));
        }

        queue_ready_.notify_one();
    }

    auto GameServer::worker_loop(std::stop_token const &stop) -> void {
        for (;;) {
            std::function<void()> task;

            {
                std::unique_lock lock{queue_mutex_};

                if (!queue_ready_.wait(lock, stop, [this] { return !queue_.empty(); })) {
                    return;
                }

                task = std::move(queue_.front());

                queue_.pop_front();
            }

            task();
        }
    }

    auto GameServer::notify_others(Room const &room, RoomId room_id, PlayerId except, std::string_view type) -> void {
        JsonWriter writer;

        writer.begin_object({}, true);
        writer.value("type", type);
        writer.value("room", room_id);
        writer.value("player", static_cast<std::uint64_t>(except));
        writer.end_object();

        for (auto const seat : room.seats) {
            if (seat != except) {
                send_to(seat, writer.str());
            }
        }
    }

    auto GameServer::send_to(PlayerId player, std::string text) -> void {
        auto const entry = players_.find(player);

        if (entry != players_.end() && entry->second.connection) {
            send_(*entry->second.connection, std::move(text));
        }
    }

    auto GameServer::reject(PlayerId player, std::string_view code, std::string_view message) -> void {
        auto const entry = players_.find(player);

        if (entry != players_.end() && entry->second.connection) {
            send_error(*entry->second.connection, code, message);
        }
    }

    auto GameServer::send_error(ConnectionId connection, std::string_view code, std::string_view message) -> void {
        JsonWriter writer;

        writer.begin_object({}, true);
        writer.value("type", "error");
        writer.value("code", code);
        writer.value("message", message);
        writer.end_object();

        send_(connection, writer.str());
    }

}
