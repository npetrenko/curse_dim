#include <gtest/gtest.h>
#include <bellman/bellman_operators/uniform_operator.hpp>

TEST(UB, Constructs) {
    DiscreteQFuncEst est;
    static_assert(std::is_base_of_v<ICloneable, DiscreteQFuncEst>);
    ASSERT_TRUE(dynamic_cast<ICloneable*>(&est));
}
