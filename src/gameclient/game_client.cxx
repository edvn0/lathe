#include "gameclient/game_client.hxx"

#include <utility>

#include "core/json.hxx"

namespace gameclient {

    GameClient::GameClient(TransportFactory factory) : factory_(std::move(factory)) {}

    GameClient::~GameClient() { close(); }

    auto GameClient::connect(std::string url) -> void {
        {
            std::scoped_lock const lock{mutex_};

            url_ = std::move(url);
            token_.clear();
        }

        dial();
    }

    auto GameClient::reconnect() -> void {
        {
            std::scoped_lock const lock{mutex_};

            if (url_.empty()) {
                return;
            }
        }

        dial();
    }

    auto GameClient::dial() -> void {
        std::shared_ptr<IClientTransport> previous;
        std::string url;
        std::uint64_t generation = 0;

        {
            std::scoped_lock const lock{mutex_};

            previous = std::move(transport_);
            url = url_;
            generation = ++generation_;
            open_ = false;
        }

        // Destroyed outside the lock: closing joins the I/O thread, which may be waiting for it in a callback.
        previous.reset();

        std::shared_ptr<IClientTransport> transport = factory_();

        auto *const raw = transport.get();

        {
            std::scoped_lock const lock{mutex_};

            transport_ = std::move(transport);
        }

        raw->connect(
                url,
                ClientTransportEvents{
                        .on_open =
                                [this, generation] {
                                    std::scoped_lock const lock{mutex_};

                                    if (generation != generation_) {
                                        return;
                                    }

                                    open_ = true;
                                    events_.push_back({ClientEvent::Kind::opened, {}});
                                },
                        .on_message =
                                [this, generation](std::string text) {
                                    std::scoped_lock const lock{mutex_};

                                    if (generation != generation_) {
                                        return;
                                    }

                                    // The first welcome carries the session to resume. Later ones, from a fresh
                                    // connection that has not resumed yet, belong to a throwaway session.
                                    if (token_.empty()) {
                                        if (auto const parsed = parse_json(text);
                                            parsed && (*parsed)["type"].as_string() == "welcome") {
                                            token_ = (*parsed)["token"].as_string();
                                        }
                                    }

                                    events_.push_back({ClientEvent::Kind::message, std::move(text)});
                                },
                        .on_close =
                                [this, generation](std::string reason) {
                                    std::scoped_lock const lock{mutex_};

                                    if (generation != generation_) {
                                        return;
                                    }

                                    open_ = false;
                                    events_.push_back({ClientEvent::Kind::closed, std::move(reason)});
                                },
                });
    }

    auto GameClient::resume() -> bool {
        std::string token;

        {
            std::scoped_lock const lock{mutex_};

            token = token_;
        }

        if (token.empty()) {
            return false;
        }

        JsonWriter writer;

        writer.begin_object({}, true);
        writer.value("type", "resume");
        writer.value("token", token);
        writer.end_object();

        return send(writer.str());
    }

    auto GameClient::send(std::string text) -> bool {
        std::shared_ptr<IClientTransport> transport;

        {
            std::scoped_lock const lock{mutex_};

            if (!open_) {
                return false;
            }

            transport = transport_;
        }

        // Outside the lock: a transport may deliver the reply on this very thread.
        return transport && transport->send(std::move(text));
    }

    auto GameClient::forget_session() -> void {
        std::scoped_lock const lock{mutex_};

        token_.clear();
    }

    auto GameClient::close() -> void {
        std::shared_ptr<IClientTransport> previous;

        {
            std::scoped_lock const lock{mutex_};

            previous = std::move(transport_);
            ++generation_;
            open_ = false;
        }

        previous.reset();
    }

    auto GameClient::drain() -> std::vector<ClientEvent> {
        std::scoped_lock const lock{mutex_};

        return std::exchange(events_, {});
    }

    auto GameClient::is_open() const -> bool {
        std::scoped_lock const lock{mutex_};

        return open_;
    }

    auto GameClient::token() const -> std::string {
        std::scoped_lock const lock{mutex_};

        return token_;
    }

}
