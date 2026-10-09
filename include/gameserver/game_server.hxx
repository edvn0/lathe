#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gameserver/server_game.hxx"

namespace gameserver {

    using ConnectionId = std::uint64_t;
    using RoomId = std::uint32_t;

    using Clock = std::chrono::steady_clock;

    using GameFactory = std::function<std::unique_ptr<IServerGame>()>;
    using SendFunction = std::function<void(ConnectionId, std::string)>;

    // Transport-agnostic room and session logic. A transport (WebSocket, TCP, an in-memory test pipe) feeds it
    // connect / disconnect / message events and supplies a send function. Every entry point is thread-safe, so a
    // transport may call in from its own I/O threads. Messages are JSON text:
    //
    //   server -> client  {"type":"welcome","player":7,"token":"..."}   sent on connect; keep the token to resume
    //   client -> server  {"type":"resume","token":"..."}               rebind a fresh connection to an old player
    //                     {"type":"create_room","game":"chess"}         creates a room and joins it
    //                     {"type":"join_room","room":1}
    //                     {"type":"action","action":{...}}              game-defined payload
    //                     {"type":"leave_room"}
    //   server -> client  {"type":"joined","room":1,"game":"chess","player":7,"seat":0}
    //                     {"type":"state","room":1,"state":{...}}       per-player view, on start and after each action
    //                     {"type":"left","room":1}                      acknowledges leave_room
    //                     {"type":"opponent_disconnected","room":1,"player":8}   seat held until the grace period ends
    //                     {"type":"opponent_reconnected","room":1,"player":8}
    //                     {"type":"game_over","room":1,"winner":7,"reason":"..."}
    //                     {"type":"room_closed","room":1,"reason":"..."}
    //                     {"type":"error","code":"...","message":"..."}
    //
    // A player in a started room who disconnects keeps their seat for `reconnect_grace`; on_tick() closes the room
    // once that runs out. Anyone else is forgotten on disconnect.
    //
    // Threading: session bookkeeping (players, rooms, seats) is guarded by one short-held mutex, but IServerGame
    // calls run on a worker pool. Each room is a strand: its game is only ever touched by one worker at a time and its
    // jobs run in submission order, while different rooms run in parallel. Results are delivered back under the
    // mutex, so a slow game never blocks the lobby or other games. Games therefore need no locking of their own.
    class GameServer {
    public:
        // worker_threads == 0 uses one per hardware thread.
        explicit GameServer(SendFunction send, Clock::duration reconnect_grace = std::chrono::seconds{60},
                            std::size_t worker_threads = 0);

        // Joins the workers; the transport must already be stopped.
        ~GameServer();

        GameServer(GameServer const &) = delete;
        auto operator=(GameServer const &) -> GameServer & = delete;
        GameServer(GameServer &&) = delete;
        auto operator=(GameServer &&) -> GameServer & = delete;

        auto register_game(std::string name, GameFactory factory) -> void;

        auto on_connect(ConnectionId connection) -> void;
        auto on_disconnect(ConnectionId connection, Clock::time_point now = Clock::now()) -> void;
        auto on_message(ConnectionId connection, std::string_view text) -> void;

        // Expires seats whose reconnect grace has run out. Call periodically.
        auto on_tick(Clock::time_point now = Clock::now()) -> void;

        // Blocks until every submitted game job has run and delivered its result. For tests and orderly shutdown; do not
        // call from a send callback.
        auto wait_idle() -> void;

    private:
        struct Player {
            PlayerId id = 0;

            std::string token;

            std::optional<ConnectionId> connection;
            std::optional<RoomId> room;
            std::optional<Clock::time_point> disconnected_at;
        };

        using RenderedStates = std::vector<std::pair<PlayerId, std::string>>;

        struct ActionResult {
            std::optional<ActionError> error;

            RenderedStates states;

            std::optional<GameOutcome> outcome;
        };

        struct Job {
            // Runs on a worker, off the mutex, and may touch the game.
            std::function<void()> work;

            // Runs on the same worker under the mutex, unless the room closed in the meantime.
            std::function<void()> done;
        };

        struct Room {
            std::string game_name;

            // Only the room's strand touches the game once the room exists.
            std::shared_ptr<IServerGame> game;

            std::vector<PlayerId> seats;

            bool started = false;
            bool closed = false;

            std::deque<Job> jobs;
            bool running = false;
        };

        auto handle_resume(ConnectionId connection, JsonValue const &message) -> void;
        auto handle_create_room(Player &player, JsonValue const &message) -> void;
        auto handle_join_room(Player &player, JsonValue const &message) -> void;
        auto handle_action(Player &player, JsonValue const &message) -> void;

        auto seat_player(Player &player, RoomId room_id) -> void;
        auto send_welcome(Player const &player) -> void;
        auto send_joined(Player const &player) -> void;
        auto leave_room(Player &player, std::string_view reason) -> void;
        auto close_room(RoomId room_id, std::string_view reason) -> void;
        auto forget(PlayerId player) -> void;

        auto render_for(RoomId room_id, std::shared_ptr<Room> const &room, std::vector<PlayerId> players) -> void;
        auto finish_action(RoomId room_id, Room const &room, PlayerId player, ActionResult const &result) -> void;
        auto deliver(RenderedStates const &states) -> void;

        auto submit(std::shared_ptr<Room> const &room, std::function<void()> work, std::function<void()> done) -> void;
        auto run_strand(std::shared_ptr<Room> const &room) -> void;
        auto finish_job() -> void;
        auto enqueue(std::function<void()> task) -> void;
        auto worker_loop(std::stop_token const &stop) -> void;

        auto notify_others(Room const &room, RoomId room_id, PlayerId except, std::string_view type) -> void;

        auto send_to(PlayerId player, std::string text) -> void;
        auto send_error(ConnectionId connection, std::string_view code, std::string_view message) -> void;
        auto reject(PlayerId player, std::string_view code, std::string_view message) -> void;

        SendFunction send_;

        Clock::duration reconnect_grace_;

        std::mutex mutex_;

        std::unordered_map<std::string, GameFactory> games_;
        std::unordered_map<PlayerId, Player> players_;
        std::unordered_map<ConnectionId, PlayerId> connections_;
        std::unordered_map<std::string, PlayerId> tokens_;
        std::unordered_map<RoomId, std::shared_ptr<Room>> rooms_;

        PlayerId next_player_ = 1;
        RoomId next_room_ = 1;

        std::size_t pending_jobs_ = 0;
        std::condition_variable idle_;

        std::mutex queue_mutex_;
        std::condition_variable_any queue_ready_;
        std::deque<std::function<void()>> queue_;

        // Declared last so the workers are joined before anything they use is destroyed.
        std::vector<std::jthread> workers_;
    };

}
