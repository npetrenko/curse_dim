#pragma once

#include <src/bellman.hpp>
#include <src/kernel.hpp>
#include <src/particle.hpp>

#include <random>

template <size_t dim>
class UniformQFuncEst : public AbstractQFuncEstimate<UniformQFuncEst<dim>> {
public:
    UniformQFuncEst() {
    }

private:
    std::vector<std::array<FloatT, dim>> values;
};

template <class RandomDeviceT, class... T>
class UniformBellmanOperator {
public:
    UniformBellmanOperator(ActionConditionedKernel<T...>&& ac_kernel, size_t num_particles,
                           FloatT diameter, RandomDeviceT* random_device)
        : ac_kernel_{std::move(ac_kernel)},
          diameter_{diameter},
          random_device_{random_device},
          cluster_{num_particles, EmptyInitializer<MemoryView>{ac_kernel_.GetDim()}} {
	      std::uniform_real_distribution<FloatT> distr{-diameter, diameter};
	      RandomVectorizingInitializer<MemoryView, decltype(distr), RandomDeviceT> initializer{ac_kernel_.GetDim(), random_device, distr};
	      for (auto& particle : cluster_) {
		  initializer.Initialize(particle);
	      }
    }

private:
    ActionConditionedKernel<T...> ac_kernel_;
    FloatT diameter_;
    RandomDeviceT* random_device_;
    ParticleCluster cluster_;
};
