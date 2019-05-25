#pragma once

#include <src/bellman.hpp>
#include <src/kernel.hpp>
#include <src/particle.hpp>

#include <random>

template <size_t dim>
class QFuncEst : public AbstractQFuncEstimate<QFuncEst<dim>> {
public:
    QFuncEst(size_t num_particles) : values_(num_particles, 0) {
    }

    auto& operator[](size_t i) {
        return values_[i];
    }

    const auto& operator[](size_t i) const {
        return values_[i];
    }

    void EnumerateCluster(ParticleCluster* cluster) const {
	size_t particle_index = 0;
        for (auto& particle : *cluster) {
	    assert(particle_index < values_.size());
            particle.SetStorageLocalIndex(++particle_index);
        }
	assert(particle_index == values_.size());
    }

    FloatT& ValueAtPoint() {
    }

private:
    std::vector<std::array<FloatT, dim>> values_;
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
          qfunc_primary_{num_particles},
          qfunc_secondary_{num_particles},
          reward_func_{std::move(reward_func)} {

        std::uniform_real_distribution<FloatT> distr{-diameter, diameter};
        RandomVectorizingInitializer<MemoryView, decltype(distr), RandomDeviceT> initializer{
            ac_kernel_.GetDim(), random_device, distr};

        for (auto& particle : cluster_) {
            initializer.Initialize(particle);
        }
	qfunc_primary_.EnumerateCluster(&cluster_);

        NormalizeWeights();
    }

    void MakeIteration() {
        for (size_t j = 0; j < cluster_.size(); ++j) {
            for (size_t i = 0; i < cluster_.size(); ++i) {
                for (size_t action_number = 0; action_number < ac_kernel_; ++action_number) {
		    qfunc_secondary_[i][action_number];
                }
            }
        }
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
    QFuncEst<sizeof...(T)> qfunc_primary_, qfunc_secondary_;
    RewardFuncT reward_func_;
};
