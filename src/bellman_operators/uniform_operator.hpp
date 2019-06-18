#pragma once

#include <src/bellman.hpp>
#include <src/kernel.hpp>
#include <src/particle.hpp>
#include <src/bellman_operators/qfunc.hpp>
#include <src/bellman_operators/environment.hpp>
#include <src/bellman_operators/abstract_bellman.hpp>
#include <src/density_estimators/stationary_estimator.hpp>

#include <random>
#include <cassert>
#include <optional>

#include <glog/logging.h>

#ifndef NDEBUG
#include <fenv.h>
#endif

#include <thread_pool/include/for_loop.hpp>

template <class RandomDeviceT, class RewardFuncT, class... Kernels>
class UniformBellmanOperator : public AbstractBellmanOperator {
public:
    UniformBellmanOperator(EnvParams<RewardFuncT, Kernels...> env_params, size_t num_particles,
                           FloatT radius, RandomDeviceT* random_device);

    void MakeIteration() override;

    const DiscreteQFuncEst& GetQFunc() const override {
        return qfunc_primary_;
    }

    const ConstantWeightedParticleCluster& GetSamplingDistribution() const override {
        return *sampling_distribution_;
    }

private:
    void NormalizeWeights();

    EnvParams<RewardFuncT, Kernels...> env_params_;
    FloatT radius_;
    RandomDeviceT* random_device_;
    std::vector<std::array<FloatT, sizeof...(Kernels)>> additional_weights_;
    DiscreteQFuncEst qfunc_primary_, qfunc_secondary_;
    std::unique_ptr<ConstantWeightedParticleCluster> sampling_distribution_;
};

template <class RandomDeviceT, class RewardFuncT, class... T>
UniformBellmanOperator<RandomDeviceT, RewardFuncT, T...>::UniformBellmanOperator(
    EnvParams<RewardFuncT, T...> env_params, size_t num_particles, FloatT radius,
    RandomDeviceT* random_device)
    : env_params_{std::move(env_params)},
      radius_{radius},
      random_device_{random_device},
      additional_weights_(num_particles),
      qfunc_primary_{num_particles, env_params_.ac_kernel.GetDim()},
      qfunc_secondary_{num_particles, env_params_.ac_kernel.GetDim()} {
    std::uniform_real_distribution<FloatT> distr{-radius, radius};
    RandomVectorizingInitializer<MemoryView, decltype(distr), RandomDeviceT> initializer{
        env_params_.ac_kernel.GetSpaceDim(), random_device, distr};

    qfunc_primary_.SetParticleCluster(ParticleCluster{num_particles, initializer});
    {
        std::uniform_real_distribution<FloatT> q_init{-0.01, 0.01};
        qfunc_primary_.SetRandom(random_device, q_init);
        qfunc_secondary_.SetRandom(random_device, q_init);
    }

    NormalizeWeights();
    {
        FloatT weight = pow(1 / (2 * radius_), env_params.ac_kernel.GetSpaceDim());
        sampling_distribution_ = std::make_unique<ConstantWeightedParticleCluster>(
            qfunc_primary_.GetParticleCluster(), weight);
    }
}

template <class RandomDeviceT, class RewardFuncT, class... T>
void UniformBellmanOperator<RandomDeviceT, RewardFuncT, T...>::MakeIteration() {
#ifndef NDEBUG
    feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif
    auto& cluster = qfunc_primary_.GetParticleCluster();
    assert(additional_weights_.size() == cluster.size());
    assert(additional_weights_[0].size() == env_params_.ac_kernel.GetDim());

    GreedyPolicy policy{qfunc_primary_};

    auto& ac_kernel = env_params_.ac_kernel;

    ParallelFor{0, cluster.size(), 32}([&](size_t i) {
        for (size_t action_number = 0; action_number < ac_kernel.GetDim(); ++action_number) {
            qfunc_secondary_.ValueAtIndex(i)[action_number] =
                env_params_.reward_function(cluster[i], action_number);
        }
        for (size_t j = 0; j < cluster.size(); ++j) {
            for (size_t action_number = 0; action_number < ac_kernel.GetDim(); ++action_number) {
                size_t reaction = policy.React(j);
                FloatT density =
                    ac_kernel.GetTransDensityConditionally(cluster[i], cluster[j], action_number);
                qfunc_secondary_.ValueAtIndex(i)[action_number] +=
                    env_params_.kGamma * density * qfunc_primary_.ValueAtIndex(j)[reaction] *
                    additional_weights_[i][action_number] / cluster.size();
            }
        }
    });

    std::swap(qfunc_primary_, qfunc_secondary_);
    qfunc_primary_.SetParticleCluster(std::move(qfunc_secondary_.GetParticleCluster()));
}

template <class RandomDeviceT, class RewardFuncT, class... T>
void UniformBellmanOperator<RandomDeviceT, RewardFuncT, T...>::NormalizeWeights() {
    auto& cluster = qfunc_primary_.GetParticleCluster();
    for (size_t action_number = 0; action_number < env_params_.ac_kernel.GetDim();
         ++action_number) {
        ParallelFor{0, cluster.size(), 1}([&](size_t i) {
            FloatT sum = 0;
            for (size_t j = 0; j < cluster.size(); ++j) {
                sum += env_params_.ac_kernel.GetTransDensityConditionally(
                           /*from*/ cluster[i], cluster[j], action_number) /
                       cluster.size();
            }
            // Importance sampling correction is also included into sum, so it may be not
            // close to 1
            additional_weights_[i][action_number] = 1 / sum;
        });
    }
}
