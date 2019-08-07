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

    std::optional<FloatT> init_radius_;
    std::optional<FloatT> invariant_density_threshold_;
    std::optional<FloatT> density_ratio_threshold_;
    std::optional<size_t> num_burnin_;
};
