#pragma once

#include <src/bellman.hpp>
#include <src/kernel.hpp>
#include <src/particle.hpp>
#include <src/bellman_operators/qfunc.hpp>
#include <src/bellman_operators/environment.hpp>
#include <src/density_estimators/stationary_estimator.hpp>

#include <random>
#include <cassert>
#include <optional>

#ifndef NDEBUG
#include <fenv.h>
#endif

struct StationaryBellmanOperatorParams {
    size_t num_particles;
    FloatT density_ratio_threshold, init_radius;
    FloatT uniform_sampling_ratio;
    size_t num_burnin_iterations = 10;
};

template <class RandomDeviceT, class RewardFuncT, class EstimatorKernelT, class... T>
class StationaryBellmanOperator {
public:
    StationaryBellmanOperator(EnvParams<RewardFuncT, T...> env_params,
                              const StationaryBellmanOperatorParams& operator_params,
                              RandomDeviceT* random_device)
        : env_params_{std::move(env_params)},
          operator_params_{operator_params},
          random_device_{random_device},
          qfunc_primary_{operator_params_.num_particles, env_params_.ac_kernel.GetDim()},
          qfunc_secondary_{operator_params_.num_particles, env_params_.ac_kernel.GetDim()},
          density_estimator_{env_params_.ac_kernel,
                             EmptyInitializer<MemoryView>{env_params_.ac_kernel.GetSpaceDim()},
                             operator_params_.num_particles} {

        std::uniform_real_distribution<FloatT> distr{-operator_params_.init_radius,
                                                     operator_params_.init_radius};

        RandomVectorizingInitializer<MemoryView, decltype(distr), RandomDeviceT> initializer{
            env_params_.ac_kernel.GetSpaceDim(), random_device, distr};

        qfunc_primary_.SetParticleCluster(density_estimator_.GetCluster());

        density_estimator_ = std::make_unique<StationaryDensityEstimator<EstimatorKernelT>>(
            MDPKernel{env_params_.ac_kernel, nullptr}, initializer, operator_params_.num_particles);

        UpdateWeights(operator_params_.num_burnin_iterations);

        NormalizeWeights();
    }

    void MakeIteration() {
#ifndef NDEBUG
        feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif
        GreedyPolicy policy{qfunc_primary_};

        UpdateWeights(4);
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

    void UpdateWeights(size_t num_iterations) {
        GreedyPolicy policy{qfunc_primary_};
        auto& mdp_kernel = static_cast<decltype(MDPKernel{env_params_.ac_kernel, &policy})&>(
            density_estimator_->GetKernel());

        mdp_kernel.SetPolicy(&policy);
        density_estimator_->MakeIteration(num_iterations);
        mdp_kernel.SetPolicy(nullptr);
    }

    EnvParams<RewardFuncT, T...> env_params_;
    StationaryBellmanOperatorParams operator_params_;
    RandomDeviceT* random_device_;
    DiscreteQFuncEst qfunc_primary_, qfunc_secondary_;
    std::unique_ptr<StationaryDensityEstimator<EstimatorKernelT>> density_estimator_;
};
