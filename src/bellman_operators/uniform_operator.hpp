#pragma once

#include <src/bellman.hpp>
#include <src/kernel.hpp>
#include <src/particle.hpp>
#include <src/bellman_operators/qfunc.hpp>

#include <random>
#include <cassert>
#include <optional>

#include <fenv.h>
feenableexcept(FE_INVALID | FE_OVERFLOW);

template <class RandomDeviceT, class RewardFuncT, class... T>
class UniformBellmanOperator {
public:
    UniformBellmanOperator(ActionConditionedKernel<T...> ac_kernel, RewardFuncT reward_func,
                           size_t num_particles, FloatT radius, RandomDeviceT* random_device)
        : ac_kernel_{std::move(ac_kernel)},
          radius_{radius},
          random_device_{random_device},
          qfunc_primary_{num_particles, ac_kernel_.GetDim()},
          qfunc_secondary_{num_particles, ac_kernel_.GetDim()},
          reward_func_{std::move(reward_func)} {


        additional_weights_.resize(num_particles);
        for (auto& elem : additional_weights_) {
            for (auto& point : elem) {
                point = 1.;
            }
        }

        std::uniform_real_distribution<FloatT> distr{-radius, radius};
        RandomVectorizingInitializer<MemoryView, decltype(distr), RandomDeviceT> initializer{
            ac_kernel_.GetSpaceDim(), random_device, distr};

        qfunc_primary_.SetParticleCluster(ParticleCluster{num_particles, initializer});

        NormalizeWeights();
    }

    void MakeIteration() {
	auto& cluster = qfunc_primary_.GetParticleCluster();
	assert(additional_weights_.size() == cluster.size());
	assert(additional_weights_[0].size() == ac_kernel_.GetDim());
	
        GreedyPolicy policy{qfunc_primary_};

        for (size_t i = 0; i < cluster.size(); ++i) {
            for (size_t action_number = 0; action_number < ac_kernel_.GetDim(); ++action_number) {
                qfunc_secondary_.ValueAtIndex(i, action_number) =
                    reward_func_(cluster[i], action_number);
            }
            for (size_t j = 0; j < cluster.size(); ++j) {
                for (size_t action_number = 0; action_number < ac_kernel_.GetDim();
                     ++action_number) {
                    size_t reaction = policy.React(j);
                    FloatT density = ac_kernel_.GetTransDensityConditionally(
                        cluster[i], cluster[j], action_number);
                    qfunc_secondary_.ValueAtIndex(i, action_number) +=
                        gamma_ * density * qfunc_primary_.ValueAtIndex(j, reaction) *
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
        for (size_t action_number = 0; action_number < ac_kernel_.GetDim(); ++action_number) {
            for (size_t i = 0; i < cluster.size(); ++i) {
                FloatT sum = 0;
                for (size_t j = 0; j < cluster.size(); ++j) {
                    sum += ac_kernel_.GetTransDensityConditionally(/*from*/ cluster[i],
                                                                   cluster[j], action_number);
                }
                additional_weights_[i][action_number] /= sum;
            }
        }
    }

    ActionConditionedKernel<T...> ac_kernel_;
    FloatT radius_;
    RandomDeviceT* random_device_;
    std::vector<std::array<FloatT, sizeof...(T)>> additional_weights_;
    DiscreteQFuncEst qfunc_primary_, qfunc_secondary_;
    RewardFuncT reward_func_;
    FloatT gamma_ = 0.99;
};
