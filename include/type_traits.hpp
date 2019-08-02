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

template <class... T>
struct InheritFrom;

template <class... T>
struct InheritFromVirtual;

namespace type_traits {

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
