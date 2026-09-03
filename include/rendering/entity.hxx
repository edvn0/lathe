#pragma once

#include "core/fly_string.hxx"
#include "rendering/scene.hxx"

namespace detail {
    template<typename NameType = FlyString>
    struct Meta {
        NameType name;
    };
} // namespace detail

namespace Components {
    using Meta = detail::Meta<FlyString>;
    using GeneratedMeta = detail::Meta<std::string>;
} // namespace Components

namespace detail {
    template<typename NameType = FlyString>
    class Entity {
    public:
        Entity(Scene *s, entt::entity e) noexcept :
            scene(s), entity(e), name(scene->registry.get_or_emplace<detail::Meta<NameType>>(entity).name) {}

        Entity(Scene *s, std::string_view n) noexcept :
            scene(s), entity(s->registry.create()),
            name(scene->registry.emplace<detail::Meta<NameType>>(entity, detail::Meta<NameType>{.name = NameType(n)})
                         .name) {}

        template<typename... Args>
        Entity(Scene *s, std::format_string<Args...> fmt, Args &&...args) noexcept :
            Entity(s, std::format(fmt, std::forward<Args>(args)...)) {}

        ~Entity() = default;

        template<typename T, typename... Args>
        auto emplace(Args &&...args) const -> decltype(auto) {
            return scene->registry.emplace<T>(entity, std::forward<Args>(args)...);
        }

        template<typename T>
        auto has() const -> bool {
            return scene->registry.all_of<T>(entity);
        }

        template<typename T>
        auto get() const -> T & {
            return scene->registry.get<T>(entity);
        }

        template<typename T>
        auto get() const -> const T & {
            return scene->registry.get<T>(entity);
        }

        operator entt::entity() const noexcept { return entity; }

    private:
        mutable Scene *scene;
        entt::entity entity;
        NameType &name;

        friend class Scene;
    };

    class ReadOnlyEntity {
    public:
        ReadOnlyEntity(Scene *s, entt::entity e) noexcept : scene(s), entity(e) {}
        ~ReadOnlyEntity() = default;

        template<typename T>
        auto has() const -> bool {
            return scene->registry.all_of<T>(entity);
        }

        template<typename T>
        auto get() const -> const T & {
            return scene->registry.get<T>(entity);
        }

    private:
        Scene *scene;
        entt::entity entity;

        friend class Scene;
    };

    // Handed to IScript::on_update, which may run concurrently (once per
    // entity referencing a parallelizable() script) across thread_pool() --
    // see script.hxx. Deliberately exposes neither Entity's lazy
    // get_or_emplace<Meta> name resolution (a registry-structural mutation)
    // nor emplace<T>(): no path here can add/remove components or
    // create/destroy entities, which entt::registry only allows safely from
    // the main thread. get<T>() in-place mutation of a component the entity
    // already has is fine to do concurrently across *different* entities.
    class ScriptEntity {
    public:
        ScriptEntity(Scene *s, entt::entity e) noexcept : scene(s), entity(e) {}
        ~ScriptEntity() = default;

        template<typename T>
        auto has() const -> bool {
            return scene->registry.all_of<T>(entity);
        }

        template<typename T>
        auto get() const -> T & {
            return scene->registry.get<T>(entity);
        }

        operator entt::entity() const noexcept { return entity; }

    private:
        Scene *scene;
        entt::entity entity;

        friend class Scene;
    };

} // namespace detail


// For UUID, bullet_i etc, we use std::string.
using GeneratedEntity = detail::Entity<std::string>;

// Else, for named entities, we use FlyString.
using Entity = detail::Entity<FlyString>;

using ReadOnlyEntity = detail::ReadOnlyEntity;
using ScriptEntity = detail::ScriptEntity;
