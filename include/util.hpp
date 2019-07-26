#pragma once

#include <iostream>
#include <type_traits>
#include <tuple>

template <bool is_const>
class MemoryViewTemplate;

class ParticleStorage;
template <class Cont, class = std::enable_if_t<std::is_same_v<Cont, ParticleStorage> ||
                                               std::is_same_v<Cont, MemoryViewTemplate<true>> ||
                                               std::is_same_v<Cont, MemoryViewTemplate<false>>>>
inline std::ostream& operator<<(std::ostream& stream, const Cont& container) {
    stream << "{";
    for (const auto& elem : container) {
        stream << elem << ",";
    }
    stream << "}";
    return stream;
}

template <class Func, size_t lower_bound, class... T>
inline static void CallOnTupleIxHelper(Func&& cb, const std::tuple<T...>& tup,
                                       size_t index) noexcept {
    if constexpr (sizeof...(T) == lower_bound) {
        std::terminate();
    } else {
        if (lower_bound == index) {
            cb(std::get<lower_bound>(tup));
        } else if (lower_bound < index) {
            CallOnTupleIxHelper<Func, lower_bound + 1, T...>(std::forward<Func>(cb), tup, index);
        }
    }
}

template <class Func, class... T>
inline static void CallOnTupleIx(Func&& cb, const std::tuple<T...>& tup, size_t index) noexcept {
    CallOnTupleIxHelper<Func, 0, T...>(std::forward<Func>(cb), tup, index);
}
