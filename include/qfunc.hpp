#pragma once

#include "cloneable.hpp"

class IQFuncEstimate : public EnableCloneInterface<IQFuncEstimate> {
public:
    virtual ~IQFuncEstimate() = default;
};

class DiscreteQFuncEst final : public EnableClone<DiscreteQFuncEst, InheritFrom<IQFuncEstimate>> {
};
