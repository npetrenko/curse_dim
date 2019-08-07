#pragma once

#include <cstdint>
#include <utility>

template <class ValueType>
class alignas(alignof(ValueType) >= 64 ? alignof(ValueType) : 64) AlignToCacheline {
public:
    template <class T>
    AlignToCacheline(T&& val) noexcept(noexcept(ValueType(std::forward<T>(val))))
        : data(std::forward<T>(val)) {
    }

    ValueType data;
};
