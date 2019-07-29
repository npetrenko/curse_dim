#pragma once

#include <memory>
#include <cassert>
#include <type_traits>
#include <typeinfo>

struct DynamicCastTag {};

struct StaticCastTag {};

class ICloneable {
public:
    virtual std::unique_ptr<ICloneable> IClone() const = 0;
    virtual ~ICloneable() = default;
};

template <class... inherit_from>
struct InheritFrom : public inherit_from... {};

template <class T>
struct InheritFrom<T> : public T {
    using T::T;
};

template <class... inherit_from>
struct InheritFromVirtual : public virtual inherit_from... {};

template <class T>
struct InheritFromVirtual<T> : public virtual T {
    using T::T;
};

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

template <class Derived, class AnotherBase, bool derived_is_abstract, class CastTag,
          bool base_is_cloneable = std::is_base_of_v<ICloneable, AnotherBase>,
          bool is_virtual_inheritance = IsInheritFromVirtual_v<AnotherBase>>
class _CloneableImpl;

#define __CloneImpl                                                           \
public:                                                                       \
    std::unique_ptr<Derived> Clone() const {                                  \
        auto ptr = this->IClone().release();                                  \
        return std::unique_ptr<Derived>{Caster(ptr, CastTag{})};              \
    }                                                                         \
                                                                              \
    std::unique_ptr<ICloneable> IClone() const override {                     \
        assert(typeid(*this) == typeid(Derived));                             \
        return std::make_unique<Derived>(static_cast<const Derived&>(*this)); \
    }                                                                         \
                                                                              \
private:                                                                      \
    Derived* Caster(ICloneable* ptr, StaticCastTag) const {                   \
        return static_cast<Derived*>(ptr);                                    \
    }                                                                         \
                                                                              \
    Derived* Caster(ICloneable* ptr, DynamicCastTag) const {                  \
        return dynamic_cast<Derived*>(ptr);                                   \
    }

#define __CloneImplAbstract                                      \
public:                                                          \
    std::unique_ptr<Derived> Clone() const {                     \
        auto ptr = this->IClone().release();                     \
        return std::unique_ptr<Derived>{Caster(ptr, CastTag{})}; \
    }                                                            \
                                                                 \
private:                                                         \
    Derived* Caster(ICloneable* ptr, StaticCastTag) const {      \
        return static_cast<Derived*>(ptr);                       \
    }                                                            \
                                                                 \
    Derived* Caster(ICloneable* ptr, DynamicCastTag) const {     \
        return dynamic_cast<Derived*>(ptr);                      \
    }

// three identical implementations, only the inheritance is different

#define IMPLEMENT(IsAbstract, ImplType)                                                        \
    /* "no base is defined" case*/                                                             \
    template <class Derived, class CastTag>                                                    \
    class _CloneableImpl<Derived, void, IsAbstract, CastTag, false, IsVirtInh>                 \
        : public ICloneable {                                                                  \
        ImplType                                                                               \
    };                                                                                         \
                                                                                               \
    /* Base is defined, and already provides ICloneable*/                                      \
    template <class Derived, class AnotherBase, class CastTag>                                 \
    class _CloneableImpl<Derived, AnotherBase, IsAbstract, CastTag, true, IsVirtInh>           \
        : InherType AnotherBase {                                                              \
        static_assert(IsInheritFrom_v<AnotherBase>,                                            \
                      "Inheritance in EnableClone can only be done through InheritFrom<...>"); \
                                                                                               \
    public:                                                                                    \
        using AnotherBase::AnotherBase;                                                        \
        ImplType                                                                               \
    };                                                                                         \
                                                                                               \
    /* Base is defined, but has no ICloneable*/                                                \
    template <class Derived, class AnotherBase, class CastTag>                                 \
    class _CloneableImpl<Derived, AnotherBase, IsAbstract, CastTag, false, IsVirtInh>          \
        : InherType AnotherBase, public ICloneable {                                           \
        static_assert(IsInheritFrom_v<AnotherBase>,                                            \
                      "Inheritance in EnableClone can only be done through InheritFrom<...>"); \
                                                                                               \
    public:                                                                                    \
        using AnotherBase::AnotherBase;                                                        \
        ImplType                                                                               \
    };

#define IsVirtInh false
#define InherType public

IMPLEMENT(false, __CloneImpl)
IMPLEMENT(true, __CloneImplAbstract)

#undef IsVirtInh
#undef InherType

#define IsVirtInh true
#define InherType \
public            \
    virtual

IMPLEMENT(false, __CloneImpl)
IMPLEMENT(true, __CloneImplAbstract)

#undef IsVirtInh
#undef InherType

#undef __CloneImpl
#undef __CloneImplAbstract
#undef IMPLEMENT

template <class Derived, class AnotherBase = void, class CastTag = StaticCastTag>
class EnableClone : public _CloneableImpl<Derived, AnotherBase, false, CastTag> {
    using _CloneableImpl<Derived, AnotherBase, false, CastTag>::_CloneableImpl;
};

template <class Derived, class AnotherBase = void, class CastTag = StaticCastTag>
class EnableCloneInterface : public _CloneableImpl<Derived, AnotherBase, true, CastTag> {
    using _CloneableImpl<Derived, AnotherBase, true, CastTag>::_CloneableImpl;
};
