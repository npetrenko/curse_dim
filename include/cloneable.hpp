#pragma once

#include "type_traits.hpp"

#include <memory>
#include <cassert>
#include <typeinfo>

class ICloneable {
public:
    inline std::unique_ptr<ICloneable> IClone() const {
        return std::unique_ptr<ICloneable>(ICloneImpl());
    }

    virtual ~ICloneable() = default;

protected:
    virtual ICloneable* ICloneImpl() const = 0;
};

template <class... inherit_from>
struct InheritFrom : public inherit_from... {};

template <class T>
struct InheritFrom<T> : public T {
    using T::T;
};

template <class... inherit_from>
struct InheritFromVirtual : public virtual inherit_from... {};

template <class Derived, class AnotherBase, bool derived_is_abstract,
          bool base_is_cloneable = std::is_base_of_v<ICloneable, AnotherBase>>
class _CloneableImpl;

#define __CloneImpl                                                        \
public:                                                                    \
    inline std::unique_ptr<Derived> Clone() const {                        \
        auto ptr = this->ICloneImpl();                                     \
        return std::unique_ptr<Derived>{static_cast<Derived*>(ptr)};       \
    }                                                                      \
                                                                           \
protected:                                                                 \
    inline _CloneableImpl* ICloneImpl() const override {                   \
        if (typeid(*this) != typeid(Derived)) {                            \
            throw 0;                                        \
        }                                                                  \
        auto ret = new Derived(static_cast<const Derived&>(*this));        \
        return static_cast<_CloneableImpl*>(ret); \
    }                                                                      \
                                                                           \
private:

#define __CloneImplAbstract                                           \
public:                                                               \
    inline std::unique_ptr<Derived> Clone() const {                   \
        auto ptr = this->ICloneImpl();                                \
        return std::unique_ptr<Derived>{dynamic_cast<Derived*>(ptr)}; \
    }                                                                 \
                                                                      \
private:

// three identical implementations, only the inheritance is different

#define Implement(IsAbstract, ImplType)                                                           \
    /* "no base is defined" case*/                                                                \
    template <class Derived>                                                                      \
    class _CloneableImpl<Derived, void, IsAbstract, false> : public virtual ICloneable {          \
        ImplType                                                                                  \
    };                                                                                            \
                                                                                                  \
    /* Base is defined, and already provides ICloneable*/                                         \
    template <class Derived, class AnotherBase>                                                   \
    class _CloneableImpl<Derived, AnotherBase, IsAbstract, true> : public AnotherBase {           \
        static_assert(type_traits::IsInheritFrom_v<AnotherBase>,                                  \
                      "Inheritance in EnableClone can only be done through InheritFrom...<...>"); \
                                                                                                  \
    public:                                                                                       \
        using AnotherBase::AnotherBase;                                                           \
        ImplType                                                                                  \
    };                                                                                            \
                                                                                                  \
    /* Base is defined, but has no ICloneable*/                                                   \
    template <class Derived, class AnotherBase>                                                   \
    class _CloneableImpl<Derived, AnotherBase, IsAbstract, false> : public AnotherBase,           \
                                                                    public virtual ICloneable {   \
        static_assert(type_traits::IsInheritFrom_v<AnotherBase>,                                  \
                      "Inheritance in EnableClone can only be done through InheritFrom...<...>"); \
                                                                                                  \
    public:                                                                                       \
        using AnotherBase::AnotherBase;                                                           \
        ImplType                                                                                  \
    };

Implement(false, __CloneImpl) Implement(true, __CloneImplAbstract)

#undef __CloneImpl
#undef __CloneImplAbstract
#undef Implement

    template <class Derived, class AnotherBase = void>
    class EnableClone : public _CloneableImpl<Derived, AnotherBase, false> {
    using _CloneableImpl<Derived, AnotherBase, false>::_CloneableImpl;
};

template <class Derived, class AnotherBase = void>
class EnableCloneInterface : public _CloneableImpl<Derived, AnotherBase, true> {
    using _CloneableImpl<Derived, AnotherBase, true>::_CloneableImpl;
};
