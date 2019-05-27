#pragma once

#include <src/bellman.hpp>
#include <src/kernel.hpp>
#include <src/particle.hpp>
#include <src/bellman_operators/qfunc.hpp>

#include <src/bellman_operators/environment.hpp>

#include <random>
#include <cassert>
#include <optional>

#ifndef NDEBUG
#include <fenv.h>
#endif

template <class RandomDeviceT, class RewardFuncT, class... T>
class UniformBellmanOperator {
public:
    UniformBellmanOperator(EnvParams<RewardFuncT, T...> env_params, size_t num_particles,
                           FloatT density_ratio_threshold, FloatT init_radius,
                           RandomDeviceT* random_device)
        : env_params_{std::move(env_params)},
          density_ratio_threshold_{density_ratio_threshold},
          init_radius_{init_radius},
          random_device_{random_device},
          qfunc_primary_{num_particles, env_params_.ac_kernel.GetDim()},
          qfunc_secondary_{num_particles, env_params_.ac_kernel.GetDim()} {

        std::uniform_real_distribution<FloatT> distr{-init_radius, init_radius};
        RandomVectorizingInitializer<MemoryView, decltype(distr), RandomDeviceT> initializer{
            env_params_.ac_kernel.GetSpaceDim(), random_device, distr};

        qfunc_primary_.SetParticleCluster(ParticleCluster{num_particles, initializer});

        NormalizeWeights();
    }

    void MakeIteration() {
#ifndef NDEBUG
        feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif
    }

    DiscreteQFuncEst& GetQFunc() {
        return qfunc_primary_;
    }

    const DiscreteQFuncEst& GetQFunc() const {
        return qfunc_primary_;
    }

private:
    void NormalizeWeights() {
    }

    EnvParams<RewardFuncT, T...> env_params_;
    FloatT density_ratio_threshold_, init_radius_;
    RandomDeviceT* random_device_;
    DiscreteQFuncEst qfunc_primary_, qfunc_secondary_;
};
