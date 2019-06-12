#include <src/abstract_kernel.hpp>
#include <src/kernel.hpp>
#include <src/type_traits.hpp>

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

template <int val>
class Kernel : public AbstractKernel<Kernel<val>, std::false_type> {};

TEST(TypeTraits, ActionConditionedKernel) {
    ActionConditionedKernel ac{Kernel<1>{}, Kernel<2>{}};
    using Type =
        type_traits::DeepestCRTPType<AbstractConditionedKernel<decltype(ac), std::false_type>>;
    CheckType(Type, decltype(ac));
}

class Policy : public AbstractAgentPolicy<Policy> {};

TEST(TypeTraits, IsHintable) {
    Policy policy;
    MDPKernel kernel{ActionConditionedKernel{Kernel<1>{}, Kernel<2>{}}, &policy};
    ASSERT_TRUE(type_traits::IsHintable_v<decltype(kernel)>);
}
