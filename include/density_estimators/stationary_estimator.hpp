#pragma once

#include "../types.hpp"
#include "../kernel.hpp"
#include "../particle.hpp"
#include "../particle_storage.hpp"
#include "../util.hpp"
#include "../type_traits.hpp"
#include "weighted_particle_clusters.hpp"

#include <thread_pool/include/for_loop.hpp>

#include <glog/logging.h>

class StationaryDensityEstimator {
public:
    template <class S>
    StationaryDensityEstimator(RNGKernel* kernel,
                               const AbstractInitializer<S, MemoryView>& initializer,
                               size_t cluster_size)
        : kernel_{kernel},
          cluster_{cluster_size, initializer},
          secondary_cluster_{cluster_size,
                             EmptyInitializer<MemoryView>{ParticleDim{initializer.GetDim()}}} {
    }

    void MakeIteration(size_t num_iterations, std::mt19937* local_rd_initializer);

    inline const VectorWeightedParticleCluster& GetCluster() const {
        return cluster_;
    }

    inline VectorWeightedParticleCluster& GetCluster() {
        return cluster_;
    }

    inline const IKernel& GetKernel() const {
        return *kernel_;
    }

    inline void ResetKernel(IKernel* new_kernel) {
        kernel_ = new_kernel;
    }

private:
    void MakeWeighing();

    IKernel* kernel_;
    VectorWeightedParticleCluster cluster_;
    ParticleCluster secondary_cluster_;
};
