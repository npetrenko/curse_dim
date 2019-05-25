#pragma once

#include <src/bellman.hpp>
#include <src/kernel.hpp>

template <size_t dim>
class UniformQFuncEst: public AbstractQFuncEstimate<UniformQFuncEst<dim>> {
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
          storage_{num_particles * ac_kernel.GetDim()} {
        particles_.reserve(num_particles);
        for (size_t i = 0; i < num_particles; ++i) {
            particles_.emplace_back({ZeroInitializer(ac_kernel.GetDim(), &storage_)});
        }
    }

private:
    ActionConditionedKernel<T...> ac_kernel_;
    FloatT diameter_;
    RandomDeviceT* random_device_;
    ParticleStorage storage_;
    std::vector<Particle<MemoryView>> particles_;
};
