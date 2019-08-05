#pragma once

#include <optional>

template <class DefaultRethrowT>
class ThrowWrap {
public:
    template <class T, class RethrowT = DefaultRethrowT>
    static T& Wrap(std::optional<T>& val) {
	if (val) {
	    return *val;
	}
        throw RethrowT();
    }

    template <class T, class RethrowT = DefaultRethrowT>
    static const T& Wrap(const std::optional<T>& val) {
        if (val) {
            return *val;
        }
        throw RethrowT();
    }
};
