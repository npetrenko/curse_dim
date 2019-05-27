#pragma once

#include <src/types.hpp>
#include <src/kernel.hpp>
#include <src/particle.hpp>
#include <src/particle_storage.hpp>

class WeightedParticleCluster : public ParticleCluster {
public:
    template <class S>
    WeightedParticleCluster(size_t size, const AbstractInitializer<S, MemoryView>& initializer)
        : ParticleCluster{size, initializer}, weights_(size) {
    }

    std::vector<FloatT>& GetWeights() {
	return weights_;
    }

    const std::vector<FloatT>& GetWeights() const {
	return weights_;
    }

private:
    std::vector<FloatT> weights_;
};

template <class T>
class StationaryDensityEstimator {
public:
    template <class S>
    StationaryDensityEstimator(const AbstractKernel<T>& kernel,
                               const AbstractInitializer<S, MemoryView>& initializer,
                               size_t cluster_size)
        : kernel_{kernel},
          cluster_{cluster_size, initializer},
          secondary_cluster_{cluster_size, EmptyInitializer<MemoryView>{initializer.GetDim()}} {
    }

    void MakeIteration(size_t num) {
	if (num == 0) {
	    return;
	}

        for (size_t i = 0; i < num; ++i) {
            MakeIterationHelper();
        }

	MakeWeighing();
    }

    const WeightedParticleCluster& GetCluster() const {
	return cluster_;
    }

    WeightedParticleCluster& GetCluster() {
	return cluster_;
    }

    AbstractKernel<T>& GetKernel() {
	return kernel_;
    }

    const AbstractKernel<T>& GetKernel() const {
	return kernel_;
    }

private:
    void MakeIterationHelper() {
        for (size_t i = 0; i < cluster_.size(); ++i) {
            kernel_.Evolve(cluster_[i], &secondary_cluster_[i]);
        }
        std::swap(static_cast<ParticleCluster&>(cluster_), secondary_cluster_);
    }

    void MakeWeighing() {
	for (size_t i = 0; i < cluster_.size(); ++i) {
	    auto& particle = cluster_[i];
	    FloatT& particle_weight = cluster_.GetWeights()[i];
	    particle_weight = 0;
	    for (const auto& from_particle : secondary_cluster_) {
		particle_weight += kernel_.GetTransDensity(from_particle, particle);
	    }
	}
    }

    const AbstractKernel<T>& kernel_;
    WeightedParticleCluster cluster_;
    ParticleCluster secondary_cluster_;
};
