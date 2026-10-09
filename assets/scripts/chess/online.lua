-- An online session with a lathe-server: connecting, the lobby, waiting for an opponent and playing, with automatic
-- reconnects. It owns the network (native.net) and knows the server's protocol; main.lua only reads `phase` and
-- reacts to the events update() returns, so the board code never touches a socket.
--
-- phase: "connecting", "lobby", "waiting" (in a room, opponent not yet seated), "playing", "reconnecting", "failed"
--
-- update(dt) returns a list of events:
--   { kind = "state", state = <the server's chess state: board, history, your_side, ...> }
--   { kind = "game_over", winner = <player id or nil>, reason = "checkmate" }
--   { kind = "room_closed", reason = "player_left" }
--   { kind = "notice", text = "..." }

local net = require("native.net")

local GAME = "chess"
local RETRY_SECONDS = 2.0
local RETRY_LIMIT = 40
local RESUME_WAIT = 1.0

local Session = {}
Session.__index = Session

local M = {}

function M.connect(url)
    local self = setmetatable({}, Session)

    self.client = net.new()
    self.url = url
    self.phase = "connecting"
    self.status = "Connecting to " .. url
    self.error = nil
    self.player = nil
    self.room = nil
    self.side = nil
    self.retry_timer = 0.0
    self.retries = 0
    self.resuming = false

    self.client:connect(url)

    return self
end

function Session:send(message)
    if not self.client:send(message) then
        self.error = "Not connected"
    end
end

function Session:create_room()
    self.error = nil
    self:send({ type = "create_room", game = GAME })
end

function Session:join_room(room)
    self.error = nil
    self:send({ type = "join_room", room = room })
end

-- Leaves the room (if any) and returns to the lobby.
function Session:leave()
    if self.room then
        self:send({ type = "leave_room" })
    end

    self.room = nil
    self.side = nil

    if self.phase ~= "failed" and self.phase ~= "connecting" then
        self.phase = "lobby"
    end
end

function Session:move(from_name, to_name, promotion)
    local action = { from = from_name, to = to_name }

    if promotion then
        action.promotion = promotion:sub(1, 1) == "k" and "n" or promotion:sub(1, 1)
    end

    self:send({ type = "action", action = action })
end

function Session:close()
    self.client:close()
end

function Session:retry()
    self.phase = "connecting"
    self.status = "Connecting to " .. self.url
    self.error = nil
    self.resuming = false
    self.client:connect(self.url)
end

-- A dropped connection: give a game the server's grace period to come back, fail anything else.
function Session:on_closed(reason)
    if self.phase == "playing" or self.phase == "waiting" or self.phase == "reconnecting" then
        if self.retries >= RETRY_LIMIT then
            self.phase = "failed"
            self.status = "Lost the connection to the server"
            return
        end

        self.phase = "reconnecting"
        self.status = "Connection lost, reconnecting..."
        self.retry_timer = RETRY_SECONDS
        return
    end

    self.phase = "failed"
    self.status = (reason and reason ~= "") and ("Could not connect: " .. reason) or "Could not connect"
end

function Session:on_message(message, events)
    local kind = message.type

    if kind == "welcome" then
        if not self.resuming then
            self.player = message.player
            self.phase = "lobby"
        elseif message.player == self.player then
            -- The server took our resume. If we were in a room, joined and state follow straight away.
            self.resume_wait = RESUME_WAIT
        end
        -- Otherwise this is the throwaway session every new connection starts with; the resume answer is next.
    elseif kind == "joined" then
        self.resuming = false
        self.retries = 0
        self.room = message.room
        self.side = message.seat == 0 and "white" or "black"

        if self.phase ~= "playing" then
            self.phase = "waiting"
        end
    elseif kind == "state" then
        self.resuming = false
        self.retries = 0
        self.phase = "playing"
        self.side = message.state.your_side or self.side

        table.insert(events, { kind = "state", state = message.state })
    elseif kind == "game_over" then
        self.room = nil
        self.phase = "lobby"

        table.insert(events, { kind = "game_over", winner = message.winner, reason = message.reason })
    elseif kind == "room_closed" then
        self.room = nil
        self.side = nil
        self.phase = "lobby"

        table.insert(events, { kind = "room_closed", reason = message.reason })
    elseif kind == "left" then
        self.room = nil
        self.side = nil
        self.phase = "lobby"
    elseif kind == "opponent_disconnected" then
        table.insert(events, { kind = "notice", text = "Opponent disconnected, holding their seat" })
    elseif kind == "opponent_reconnected" then
        table.insert(events, { kind = "notice", text = "Opponent is back" })
    elseif kind == "error" then
        if message.code == "bad_token" then
            -- The server forgot us (the grace period ran out): carry on as a new player.
            self.client:forget_session()
            self.resuming = false
            self.room = nil
            self.side = nil
            self.phase = "lobby"

            table.insert(events, { kind = "room_closed", reason = "session_expired" })
        elseif message.code == "illegal_move" or message.code == "not_your_turn" then
            table.insert(events, { kind = "notice", text = message.message })
        else
            self.error = message.message
        end
    end
end

function Session:update(dt)
    local events = {}

    for _, event in ipairs(self.client:poll()) do
        if event.kind == "open" then
            if self.resuming then
                self.client:resume()
            end
        elseif event.kind == "message" then
            if event.message then
                self:on_message(event.message, events)
            end
        elseif event.kind == "close" then
            self:on_closed(event.reason)
        end
    end

    -- Resumed, but the server had no room for us any more (it ended while we were away).
    if self.resume_wait then
        self.resume_wait = self.resume_wait - dt

        if self.resume_wait <= 0.0 then
            self.resume_wait = nil

            if self.resuming then
                self.resuming = false
                self.retries = 0
                self.room = nil
                self.side = nil
                self.phase = "lobby"

                table.insert(events, { kind = "room_closed", reason = "game_ended" })
            end
        end
    end

    if self.phase == "reconnecting" then
        self.retry_timer = self.retry_timer - dt

        if self.retry_timer <= 0.0 then
            self.retries = self.retries + 1
            self.retry_timer = RETRY_SECONDS
            self.resuming = true
            self.client:reconnect()
        end
    end

    return events
end

return M
