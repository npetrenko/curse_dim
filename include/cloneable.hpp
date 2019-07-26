#include <memory>
#include <cassert>
#include <type_traits>
#include <typeinfo>

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

template <class T>
struct IsInheritFrom{
    static constexpr bool value = false;
};

template <class... T>
struct IsInheritFrom<InheritFrom<T...>> {
    static constexpr bool value = true;
};

template <class T>
inline constexpr bool IsInheritFrom_v = IsInheritFrom<T>::value;

template <class Derived, class AnotherBase, bool derived_is_abstract,
          bool base_is_cloneable = std::is_base_of_v<ICloneable, AnotherBase>>
class _CloneableImpl;

#define __CloneImpl                                                           \
public:                                                                       \
    std::unique_ptr<Derived> Clone() const {                                  \
        auto ptr = this->IClone().release();                                  \
        return std::unique_ptr<Derived>{static_cast<Derived*>(ptr)};    \
    }                                                                         \
                                                                              \
    std::unique_ptr<ICloneable> IClone() const override {                     \
        assert(typeid(*this) == typeid(Derived));                             \
        return std::make_unique<Derived>(static_cast<const Derived&>(*this)); \
    }                                                                         \
                                                                              \
private:

#define __CloneImplAbstract                                                \
public:                                                                    \
    std::unique_ptr<Derived> Clone() const {                               \
        auto ptr = this->IClone().release();                               \
        return std::unique_ptr<Derived>{static_cast<Derived*>(ptr)}; \
    }                                                                      \
                                                                           \
private:

// three identical implementations, only the inheritance is different

#define IMPLEMENT(IsAbstract, ImplType)                                                        \
    /* "no base is defined" case*/                                                             \
    template <class Derived>                                                                   \
    class _CloneableImpl<Derived, void, IsAbstract, false> : public ICloneable {               \
        ImplType                                                                               \
    };                                                                                         \
                                                                                               \
    /* Base is defined, and already provides ICloneable*/                                      \
    template <class Derived, class AnotherBase>                                                \
    class _CloneableImpl<Derived, AnotherBase, IsAbstract, true> : public AnotherBase {        \
        static_assert(IsInheritFrom_v<AnotherBase>,                                            \
                      "Inheritance in EnableClone can only be done through InheritFrom<...>"); \
                                                                                               \
    public:                                                                                    \
        using AnotherBase::AnotherBase;                                                        \
        ImplType                                                                               \
    };                                                                                         \
                                                                                               \
    /* Base is defined, but has no ICloneable*/                                                \
    template <class Derived, class AnotherBase>                                                \
    class _CloneableImpl<Derived, AnotherBase, IsAbstract, false> : public AnotherBase,        \
                                                                    public ICloneable {        \
        static_assert(IsInheritFrom_v<AnotherBase>,                                            \
                      "Inheritance in EnableClone can only be done through InheritFrom<...>"); \
                                                                                               \
    public:                                                                                    \
        using AnotherBase::AnotherBase;                                                        \
        ImplType                                                                               \
    };

IMPLEMENT(false, __CloneImpl)
IMPLEMENT(true, __CloneImplAbstract)

#undef __CloneImpl
#undef __CloneImplAbstract
#undef IMPLEMENT

template <class Derived, class AnotherBase = void>
class EnableClone : public _CloneableImpl<Derived, AnotherBase, false> {
    using _CloneableImpl<Derived, AnotherBase, false>::_CloneableImpl;
};

template <class Derived, class AnotherBase = void>
class EnableCloneInterface : public _CloneableImpl<Derived, AnotherBase, true> {
    using _CloneableImpl<Derived, AnotherBase, true>::_CloneableImpl;
};