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
                           FloatT radius, RandomDeviceT* random_device)
        : env_params_{std::move(env_params)},
          radius_{radius},
          random_device_{random_device},
          qfunc_primary_{num_particles, env_params_.ac_kernel.GetDim()},
          qfunc_secondary_{num_particles, env_params_.ac_kernel.GetDim()} {

        additional_weights_.resize(num_particles);
        for (auto& elem : additional_weights_) {
            for (auto& point : elem) {
                point = 1.;
            }
        }

        std::uniform_real_distribution<FloatT> distr{-radius, radius};
        RandomVectorizingInitializer<MemoryView, decltype(distr), RandomDeviceT> initializer{
            env_params_.ac_kernel.GetSpaceDim(), random_device, distr};

        qfunc_primary_.SetParticleCluster(ParticleCluster{num_particles, initializer});

        NormalizeWeights();
    }

    void MakeIteration() {
#ifndef NDEBUG
        feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif
        auto& cluster = qfunc_primary_.GetParticleCluster();
        assert(additional_weights_.size() == cluster.size());
        assert(additional_weights_[0].size() == env_params_.ac_kernel.GetDim());

        GreedyPolicy policy{qfunc_primary_};

        auto& ac_kernel = env_params_.ac_kernel;

        for (size_t i = 0; i < cluster.size(); ++i) {
            for (size_t action_number = 0; action_number < ac_kernel.GetDim(); ++action_number) {
                qfunc_secondary_.ValueAtIndex(i, action_number) =
                    env_params_.reward_function(cluster[i], action_number);
            }
            for (size_t j = 0; j < cluster.size(); ++j) {
                for (size_t action_number = 0; action_number < ac_kernel.GetDim();
                     ++action_number) {
                    size_t reaction = policy.React(j);
                    FloatT density = ac_kernel.GetTransDensityConditionally(cluster[i], cluster[j],
                                                                            action_number);
                    qfunc_secondary_.ValueAtIndex(i, action_number) +=
                        env_params_.kGamma * density * qfunc_primary_.ValueAtIndex(j, reaction) *
                        additional_weights_[i][action_number];
                }
            }
        }

        std::swap(qfunc_primary_, qfunc_secondary_);
        qfunc_primary_.SetParticleCluster(std::move(qfunc_secondary_.GetParticleCluster()));
    }

    DiscreteQFuncEst& GetQFunc() {
        return qfunc_primary_;
    }

    const DiscreteQFuncEst& GetQFunc() const {
        return qfunc_primary_;
    }

private:
    void NormalizeWeights() {
        auto& cluster = qfunc_primary_.GetParticleCluster();
        for (size_t action_number = 0; action_number < env_params_.ac_kernel.GetDim();
             ++action_number) {
            for (size_t i = 0; i < cluster.size(); ++i) {
                FloatT sum = 0;
                for (size_t j = 0; j < cluster.size(); ++j) {
                    sum += env_params_.ac_kernel.GetTransDensityConditionally(
                        /*from*/ cluster[i], cluster[j], action_number);
                }
                additional_weights_[i][action_number] /= sum;
            }
        }
    }

    EnvParams<RewardFuncT, T...> env_params_;
    FloatT radius_;
    RandomDeviceT* random_device_;
    std::vector<std::array<FloatT, sizeof...(T)>> additional_weights_;
    DiscreteQFuncEst qfunc_primary_, qfunc_secondary_;
};
