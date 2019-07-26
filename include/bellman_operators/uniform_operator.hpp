#pragma once

#include "../bellman.hpp"
#include "../kernel.hpp"
#include "../particle.hpp"
#include "qfunc.hpp"
#include "environment.hpp"
#include "abstract_bellman.hpp"
#include "../density_estimators/stationary_estimator.hpp"

#include <random>
#include <cassert>
#include <optional>

#include <glog/logging.h>

#ifndef NDEBUG
#include <fenv.h>
#endif

#include "../../thread_pool/include/for_loop.hpp"
#include "../matrix.hpp"

class UniformBellmanOperator : public AbstractBellmanOperator {
public:
    class Builder;
    void MakeIteration() override;

    inline const DiscreteQFuncEst& GetQFunc() const override {
        return qfunc_primary_;
    }
    inline const ConstantWeightedParticleCluster& GetSamplingDistribution() const override {
        return *sampling_distribution_;
    }

private:
    UniformBellmanOperator() = default;
    void NormalizeWeights();

    EnvParams env_params_;
    FloatT radius_;
    std::mt19937* random_device_;
    Matrix<std::vector<FloatT>> additional_weights_;

    DiscreteQFuncEst qfunc_primary_, qfunc_secondary_;
    std::unique_ptr<ConstantWeightedParticleCluster> sampling_distribution_;
};

class UniformBellmanOperator::Builder
    : public AbstractBellmanOperator::Builder<UniformBellmanOperator::Builder> {
public:
    Builder() = default;

    inline Builder& SetInitRadius(FloatT init_radius) {
        init_radius_ = init_radius;
	return *this;
    }

    UniformBellmanOperator Build() &&;

private:
    FloatT init_radius_;
};
