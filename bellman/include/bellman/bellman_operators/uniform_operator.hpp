#pragma once

#include "abstract_bellman.hpp"
#include "../builder.hpp"
#include "../pimpl_ptr.hpp"

class UniformBellmanOperator;
using UniformBellmanOperatorPtr = std::unique_ptr<UniformBellmanOperator>;

class UniformBellmanOperator : public IBellmanOperator {
public:
    class Builder;
    void MakeIteration() override;
    const DiscreteQFuncEst& GetQFunc() const& override;
    DiscreteQFuncEst GetQFunc() && override;
    const ConstantWeightedParticleCluster& GetSamplingDistribution() const override;

    UniformBellmanOperator(UniformBellmanOperator&&) = default;
    ~UniformBellmanOperator() = default;

private:
    class Impl;
    UniformBellmanOperator(Builder&&);

    PimplPtr<Impl> impl_;
};

class UniformBellmanOperator::Builder
    : public AbstractBellmanOperator::Builder<UniformBellmanOperator::Builder> {
    friend class AbstractBellmanOperator::Builder<UniformBellmanOperator::Builder>;
    friend class Impl;

public:
    Builder() = default;

    inline Builder& SetInitRadius(FloatT init_radius) {
        init_radius_ = init_radius;
        return *this;
    }

private:
    std::unique_ptr<UniformBellmanOperator> BuildImpl() &&;
    BuilderOption<FloatT> init_radius_{"init_radius_"};
};
