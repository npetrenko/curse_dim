#pragma once

#include "abstract_bellman.hpp"
#include <optional>

class StationaryBellmanOperator;
using StationaryBellmanOperatorPtr = std::unique_ptr<StationaryBellmanOperator>;

class StationaryBellmanOperator : public IBellmanOperator {
public:
    class Builder;

    void MakeIteration() override;
    const DiscreteQFuncEst& GetQFunc() const& override;
    DiscreteQFuncEst GetQFunc() && override;
    const VectorWeightedParticleCluster& GetSamplingDistribution() const override;

    ~StationaryBellmanOperator();

private:
    StationaryBellmanOperator(Builder&&);
    class Impl;

    std::unique_ptr<Impl> impl_;
};

class StationaryBellmanOperator::Builder : public AbstractBellmanOperator::Builder<Builder> {
    friend class AbstractBellmanOperator::Builder<Builder>;
    friend class Impl;

public:
    Builder() = default;

    inline Builder& SetInitRadius(FloatT init_radius) {
        init_radius_ = init_radius;
        return *this;
    }

    inline Builder& SetNumBurninIter(size_t num_iter) {
        num_burnin_ = num_iter;
        return *this;
    }

    inline Builder& SetInvariantDensityThreshold(FloatT val) {
        invariant_density_threshold_ = val;
        return *this;
    }

    inline Builder& SetDensityRatioThreshold(FloatT val) {
        density_ratio_threshold_ = val;
        return *this;
    }

private:
    StationaryBellmanOperatorPtr BuildImpl();

    BuilderOption<FloatT> init_radius_{"init_radius_"};
    BuilderOption<FloatT> invariant_density_threshold_{"invariant_density_threshold_"};
    BuilderOption<FloatT> density_ratio_threshold_{"density_ratio_threshold_"};
    BuilderOption<size_t> num_burnin_{"num_burnin_"};
};
