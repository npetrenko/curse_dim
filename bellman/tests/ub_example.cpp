#include <gtest/gtest.h>
#include <bellman/bellman_operators/uniform_operator.hpp>

TEST(UB, Constructs) {
    DiscreteQFuncEst est;
    static_assert(std::is_convertible_v<DiscreteQFuncEst*, ICloneable*>);
    ASSERT_TRUE(dynamic_cast<ICloneable*>(&est));
}
