#pragma once

#include <iostream>
#include <type_traits>
#include <tuple>

class ParticleStorage;
class MemoryView;
class ConstMemoryView;

template <class Cont, class = std::enable_if_t<std::is_same_v<Cont, ParticleStorage> ||
                                               std::is_same_v<Cont, MemoryView> ||
                                               std::is_same_v<Cont, ConstMemoryView>>>
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

template <class DerivedT>
class CRTPDerivedCaster {
protected:
    using DerivedClass = DerivedT;
    inline DerivedT* GetDerived() {
        return static_cast<DerivedT*>(this);
    }

    inline const DerivedT* GetDerived() const {
        return static_cast<const DerivedT*>(this);
    }
};

template <class T>
class HintableKernel;

template <class T>
struct IsHintable {
    template <class DerivedT>
    static void Helper(const HintableKernel<DerivedT>&);

    template <class TestT, class = decltype(Helper(std::declval<TestT>()))>
    static std::true_type Tester(TestT val);

    static std::false_type Tester(...);

    static const bool value = std::is_same_v<std::true_type, decltype(Tester(std::declval<T>()))>;
};

template <class T>
inline constexpr bool IsHintable_v = IsHintable<T>::value;
