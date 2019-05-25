#pragma once

#include <iostream>
#include <type_traits>
#include <tuple>

template <class Cont, class = decltype(std::declval<Cont>().begin())>
inline std::ostream& operator<<(std::ostream& stream, const Cont& container) {
    stream << "{";
    for (const auto& elem : container) {
        stream << elem << ",";
    }
    stream << "}";
    return stream;
}

template <class Func, size_t lower_bound = 0, class... T>
inline static void CallOnTupleIx(Func&& cb, std::tuple<T...>& tup, size_t index) {
    if constexpr (sizeof...(T) == lower_bound) {
        throw std::runtime_error("out of range");
    } else {
        if (lower_bound == index) {
            cb(std::get<lower_bound>(tup));
        } else if (lower_bound < index) {
            CallOnTupleIx<Func, lower_bound + 1, T...>(std::forward<Func>(cb), tup, index);
        }
    }
}

template <class DerivedT>
struct CRTPDerivedCaster {
    using DerivedClass = DerivedT;
    DerivedT* GetDerived() {
        return static_cast<DerivedT*>(this);
    }

    const DerivedT* GetDerived() const {
        return static_cast<const DerivedT*>(this);
    }
};
