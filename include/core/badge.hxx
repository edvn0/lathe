#pragma once

// Should disallow construction of Badge by anyone other than the friend class.

template<typename T>
class Badge {
    friend T;

    Badge() = default;

public:
    Badge(Badge const &) = default;
};
