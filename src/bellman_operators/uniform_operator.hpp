#pragma once

#include <src/bellman.hpp>
#include <src/kernel.hpp>
#include <src/particle.hpp>

#include <random>
#include <cassert>

class QFuncEst : public AbstractQFuncEstimate<QFuncEst> {
public:
    QFuncEst(size_t num_particles, size_t dim) : values_(num_particles * dim, 0), dim_(dim) {
    }

    FloatT& ValueAtIndex(size_t state_ix, size_t action_number) {
        return values_[state_ix * dim_ + action_number];
    }

    FloatT ValueAtIndex(size_t state_ix, size_t action_number) const {
        return values_[state_ix * dim_ + action_number];
    }

    void SetZero() {
        for (auto& val : values_) {
            val = 0;
        }
    }

private:
    std::vector<FloatT> values_;
    size_t dim_;
};

template <class RandomDeviceT, class RewardFuncT, class... T>
class UniformBellmanOperator {
public:
    template <class Func>
    UniformBellmanOperator(ActionConditionedKernel<T...> ac_kernel, RewardFuncT reward_func,
                           size_t num_particles, FloatT diameter, RandomDeviceT* random_device)
        : ac_kernel_{std::move(ac_kernel)},
          diameter_{diameter},
          random_device_{random_device},
          cluster_{num_particles, EmptyInitializer<MemoryView>{ac_kernel_.GetDim()}},
          additional_weights_(num_particles, 1.),
          qfunc_primary_{num_particles, ac_kernel_.GetDim()},
          qfunc_secondary_{num_particles, ac_kernel_.GetDim()},
          reward_func_{std::move(reward_func)} {

        std::uniform_real_distribution<FloatT> distr{-diameter, diameter};
        RandomVectorizingInitializer<MemoryView, decltype(distr), RandomDeviceT> initializer{
            ac_kernel_.GetDim(), random_device, distr};

        for (auto& particle : cluster_) {
            initializer.Initialize(particle);
        }

        NormalizeWeights();
    }

    void MakeIteration() {
        GreedyPolicy policy{qfunc_primary_};

        for (size_t i = 0; i < cluster_.size(); ++i) {
            for (size_t action_number = 0; action_number < ac_kernel_.GetDim(); ++action_number) {
                qfunc_secondary_.ValueAtIndex(i, action_number) =
                    reward_func_(cluster_[i], action_number);
            }
            for (size_t j = 0; j < cluster_.size(); ++j) {
                for (size_t action_number = 0; action_number < ac_kernel_.GetDim();
                     ++action_number) {
                    size_t reaction = policy.React(j);
                    FloatT density =
                        ac_kernel_.GetTransDensityConditionally(cluster_[i], cluster_[j], action_number);
                    qfunc_secondary_.ValueAtIndex(i, action_number) +=
                        gamma_ * density * qfunc_primary_.ValueAtIndex(j, reaction) *
                        additional_weights_[i];
                }
            }
        }

	std::swap(qfunc_primary_, qfunc_secondary_);
    }

    const QFuncEst& GetQFunc() const {
	return qfunc_primary_;
    }

private:
    void NormalizeWeights() {
        for (size_t action_number = 0; action_number < ac_kernel_.GetDim(); ++action_number) {
            for (size_t i = 0; i < cluster_.size(); ++i) {
                FloatT sum = 0;
                for (size_t j = 0; j < cluster_.size(); ++j) {
                    sum += ac_kernel_.GetTransDensityConditionally(/*from*/ cluster_[i],
                                                                   cluster_[j], action_number);
                }
                additional_weights_[i][action_number] /= sum;
            }
        }
    }

    ActionConditionedKernel<T...> ac_kernel_;
    FloatT diameter_;
    RandomDeviceT* random_device_;
    ParticleCluster cluster_;
    std::vector<std::array<FloatT, sizeof...(T)>> additional_weights_;
    QFuncEst qfunc_primary_, qfunc_secondary_;
    RewardFuncT reward_func_;
    FloatT gamma_ = 0.99;
};
