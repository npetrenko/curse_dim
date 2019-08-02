#pragma once

#include "../cloneable.hpp"
#include "../bellman.hpp"

class DiscreteQFuncEst final : public EnableClone<DiscreteQFuncEst, InheritFrom<IQFuncEstimate>> {
};
