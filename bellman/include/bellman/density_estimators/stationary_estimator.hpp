#pragma once

#include "../types.hpp"
#include "../kernel.hpp"
#include "../particle.hpp"
#include "../particle_storage.hpp"
#include "../util.hpp"
#include "../type_traits.hpp"
#include "../builder.hpp"
#include "weighted_particle_clusters.hpp"

class StationaryDensityEstimator {
public:
    class Builder;
    void MakeIteration(size_t num_iterations, std::mt19937* local_rd_initializer);

    inline const VectorWeightedParticleCluster& GetCluster() const {
        return *cluster_;
    }

    inline const IKernel& GetKernel() const {
        return *kernel_;
    }

    inline void ResetKernel(IKernel* new_kernel) {
        kernel_ = new_kernel;
    }

    inline std::shared_ptr<VectorWeightedParticleCluster> GetClusterPtr() const {
        return cluster_;
    }

private:
    StationaryDensityEstimator() = default;
    void MakeWeighing();

    IKernel* kernel_;
    std::shared_ptr<VectorWeightedParticleCluster> cluster_;
    ParticleCluster secondary_cluster_;
};

class StationaryDensityEstimator::Builder {
public:
    inline Builder& SetClusterSize(size_t size) {
        cluster_size_ = NumParticles{size};
        return *this;
    }

    template <class S>
    Builder& SetInitializer(const AbstractInitializer<S, MemoryView>& initializer) {
        auto prim_init = [init = type_traits::GetDeepestLevelCopy(initializer), this] {
            return std::make_shared<VectorWeightedParticleCluster>(cluster_size_.Value(), init);
        };
        auto sec_init = [this, dim = initializer.GetDim()] {
            return ParticleCluster(cluster_size_.Value(),
                                   EmptyInitializer(ParticleDim(dim), ClusterInitializationTag()));
        };

        primary_cluster_builder_ = std::move(prim_init);
        secondary_cluster_builder_ = std::move(sec_init);
        return *this;
    }

    inline std::shared_ptr<VectorWeightedParticleCluster> GetParticleClusterPtr() {
        MaybeInitPrimary();
        return primary_cluster_.value();
    }

    inline Builder& SetKernel(IKernel* kernel) {
        kernel_ = kernel;
        return *this;
    }

    std::unique_ptr<StationaryDensityEstimator> Build() &&;

private:
    void MaybeInitPrimary();
    BuilderOption<NumParticles> cluster_size_{"cluster_size_"};
    BuilderOption<std::function<std::shared_ptr<VectorWeightedParticleCluster>()>>
        primary_cluster_builder_{"initializer"};
    BuilderOption<std::function<ParticleCluster()>> secondary_cluster_builder_{"initializer"};
    BuilderOption<IKernel*> kernel_{"kernel_"};

    std::optional<std::shared_ptr<VectorWeightedParticleCluster>> primary_cluster_;
};
