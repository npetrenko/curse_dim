#include <bellman/abstract_kernel.hpp>
#include <bellman/kernel.hpp>
#include <bellman/type_traits.hpp>

#include <string>

#include <gtest/gtest.h>

#define CheckType(LHS, RHS)                                                              \
    do {                                                                                 \
        if constexpr (!std::is_same_v<LHS, RHS>) {                                       \
            ASSERT_EQ(std::string(typeid(LHS).name()), std::string(typeid(RHS).name())); \
        }                                                                                \
    } while (false)

template <class DerivedT>
struct Base : public CRTPDerivedCaster<DerivedT> {};

struct Derived : public Base<Derived> {};

TEST(TypeTraits, Simple) {
    using Type = type_traits::DeepestCRTPType<Base<Derived>>;
    CheckType(Type, Derived);
}
