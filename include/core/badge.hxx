#pragma once

template<typename T>
class Badge {
    friend T;

    Badge() = default;

public:
    Badge(Badge const &) = default;
};
