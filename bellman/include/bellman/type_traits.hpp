#pragma once

#include <utility>
#include <type_traits>

struct NullType {};

template <class Derived>
class CRTPDerivedCaster {
protected:
    Derived* GetDerived() {
        return static_cast<Derived*>(this);
    }

    const Derived* GetDerived() const {
        return static_cast<const Derived*>(this);
    }
};

class HintableKernel;

template <class... T>
struct InheritFrom;

template <class... T>
struct InheritFromVirtual;

namespace type_traits {

// Find the first type that matches the predicate
template <template <class T> class Predicate, class... Args>
struct FindFirstMatching;

template <template <class T> class Predicate>
struct FindFirstMatching<Predicate> {
    using Type = NullType;
    static const bool has_match = false;
};

template <template <class T> class Predicate, class Arg, class... Args>
struct FindFirstMatching<Predicate, Arg, Args...> {
    using Type = std::conditional_t<Predicate<Arg>::value, Arg,
                                    typename FindFirstMatching<Predicate, Args...>::Type>;
    static const bool has_match = !std::is_same_v<NullType, Type>;
};

// Utility class to get the deepest class in CRTP inheritance chain
template <typename T>
struct GetDeepest {
    using Type = T;
};

template <template <class...> class DT, class... T>
struct GetDeepest<DT<T...>> {
    template <class CLS>
    struct Predicate {
        static const bool value = std::is_base_of_v<DT<T...>, CLS>;
    };

    static const bool HasCRTPDerived = FindFirstMatching<Predicate, T...>::has_match;
    using DerivedT = typename FindFirstMatching<Predicate, T...>::Type;

    using Type = std::conditional_t<HasCRTPDerived, typename GetDeepest<DerivedT>::Type, DT<T...>>;
};

template <class T>
using DeepestCRTPType = typename GetDeepest<T>::Type;

template <class T>
DeepestCRTPType<T> GetDeepestLevelCopy(const T& arg) {
    return static_cast<const DeepestCRTPType<T>&>(arg);
}

// Class to find if the class is public derived from HintableKernel<T>
template <class T>
struct IsHintable {
    static void Helper(const ::HintableKernel&);

    template <class TestT, class = decltype(Helper(std::declval<TestT&>()))>
    static std::true_type Tester(const TestT& val);

    static std::false_type Tester(...);

    static const bool value = std::is_same_v<std::true_type, decltype(Tester(std::declval<T&>()))>;
};

template <class T>
inline constexpr bool IsHintable_v = IsHintable<T>::value;

template <class T>
struct IsInheritFrom {
    static constexpr bool value = false;
};

template <class... T>
struct IsInheritFrom<InheritFrom<T...>> {
    static constexpr bool value = true;
};

template <class T>
struct IsInheritFromVirtual {
    static constexpr bool value = false;
};

template <class... T>
struct IsInheritFromVirtual<InheritFromVirtual<T...>> {
    static constexpr bool value = true;
};

template <class T>
inline constexpr bool IsInheritFrom_v = IsInheritFrom<T>::value || IsInheritFromVirtual<T>::value;

template <class T>
inline constexpr bool IsInheritFromVirtual_v = IsInheritFromVirtual<T>::value;

}  // namespace type_traits
