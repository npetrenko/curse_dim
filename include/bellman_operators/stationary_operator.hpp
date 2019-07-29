#pragma once

#include <include/bellman.hpp>
#include <include/bellman_operators/abstract_bellman.hpp>
#include <include/matrix.hpp>

#include <random>
#include <optional>

#include <glog/logging.h>

#ifndef NDEBUG
#include <fenv.h>
#endif

#include <thread_pool/include/for_loop.hpp>

class StationaryBellmanOperator final : public AbstractBellmanOperator {
public:
    class Builder;

    void MakeIteration() override;

    inline const DiscreteQFuncEst& GetQFunc() const override {
        return qfunc_primary_;
    }

    const VectorWeightedParticleCluster& GetSamplingDistribution() const override {
        if (!prev_sampling_distribution_.get()) {
            throw std::runtime_error("Sampling distribution has not been initialized");
        }
        return *prev_sampling_distribution_;
    }

private:
    struct Params {
        FloatT invariant_density_threshold;
        FloatT density_ratio_threshold;
        FloatT init_radius;
    };

    const Params kParams;

    StationaryBellmanOperator(AbstractBellmanOperator::Params&&, Params&& params);
    void UpdateParticleCluster(size_t num_iterations);

    void RecomputeWeights();

    DiscreteQFuncEst qfunc_primary_, qfunc_secondary_;
    std::unique_ptr<StationaryDensityEstimator> density_estimator_{nullptr};
    Matrix<std::vector<FloatT>> additional_weights_;
    std::unique_ptr<VectorWeightedParticleCluster> prev_sampling_distribution_{nullptr};
};

class StationaryBellmanOperator::Builder : public AbstractBellmanOperator::Builder<Builder> {
    friend class AbstractBellmanOperator::Builder<Builder>;

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
    std::unique_ptr<StationaryBellmanOperator> BuildImpl(AbstractBellmanOperator::Params&& params);

    std::optional<FloatT> init_radius_;
    std::optional<FloatT> invariant_density_threshold_;
    std::optional<FloatT> density_ratio_threshold_;
    std::optional<size_t> num_burnin_;
};
