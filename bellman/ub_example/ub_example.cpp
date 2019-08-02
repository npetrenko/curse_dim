#include <bellman/bellman_operators/qfunc.hpp>

int main() {
    DiscreteQFuncEst est;
    static_assert(std::is_convertible_v<DiscreteQFuncEst*, ICloneable*>);
    bool should_be_true = dynamic_cast<ICloneable*>(&est) != 0; // UBSan complains here
    bool has_failed = !should_be_true;
    return has_failed;
}
