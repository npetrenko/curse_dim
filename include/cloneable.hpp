#pragma once

#include <type_traits>

class ICloneable {
public:
    virtual ~ICloneable() = default;
};

template <class T>
struct InheritFrom : public T {
    using T::T;
};

template <class Derived, class AnotherBase, bool derived_is_abstract,
          bool base_is_cloneable = std::is_base_of_v<ICloneable, AnotherBase>>
class _CloneableImpl;

// three identical implementations, only the inheritance is different

#define Implement(IsAbstract)                                                                     \
    /* "no base is defined" case*/                                                                \
    template <class Derived>                                                                      \
    class _CloneableImpl<Derived, void, IsAbstract, false> : public virtual ICloneable {          \
    };                                                                                            \
                                                                                                  \
    /* Base is defined, and already provides ICloneable*/                                         \
    template <class Derived, class AnotherBase>                                                   \
    class _CloneableImpl<Derived, AnotherBase, IsAbstract, true> : public AnotherBase {           \
    };                                                                                            \
                                                                                                  \
    /* Base is defined, but has no ICloneable*/                                                   \
    template <class Derived, class AnotherBase>                                                   \
    class _CloneableImpl<Derived, AnotherBase, IsAbstract, false> : public AnotherBase,           \
                                                                    public virtual ICloneable {   \
    };

Implement(false) 
Implement(true)

#undef Implement

template <class Derived, class AnotherBase = void>
class EnableClone : public _CloneableImpl<Derived, AnotherBase, false> {
};

template <class Derived, class AnotherBase = void>
class EnableCloneInterface : public _CloneableImpl<Derived, AnotherBase, true> {
};
