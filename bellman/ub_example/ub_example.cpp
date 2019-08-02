#include <bellman/bellman_operators/uniform_operator.hpp>

int main() {
    DiscreteQFuncEst est;
    static_assert(std::is_convertible_v<DiscreteQFuncEst*, ICloneable*>);
    bool should_be_true = dynamic_cast<ICloneable*>(&est) != 0;
    bool has_failed = !should_be_true;
    return has_failed;
}
