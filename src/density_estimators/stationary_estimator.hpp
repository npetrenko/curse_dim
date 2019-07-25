#pragma once

#include "../types.hpp"
#include "../kernel.hpp"
#include "../particle.hpp"
#include "../particle_storage.hpp"
#include "../util.hpp"
#include "../type_traits.hpp"
#include "weighted_particle_clusters.hpp"

#include "../thread_pool/include/for_loop.hpp"

#include <glog/logging.h>

class StationaryDensityEstimator {
public:
    template <class S>
    StationaryDensityEstimator(RNGKernel* kernel,
                               const AbstractInitializer<S, MemoryView>& initializer,
                               size_t cluster_size)
        : kernel_{kernel},
          cluster_{cluster_size, initializer},
          secondary_cluster_{cluster_size, EmptyInitializer<MemoryView>{initializer.GetDim()}} {
    }

    template <bool parallel = false, class RandomDeviceT = std::mt19937>
    void MakeIteration(size_t num_iterations, RandomDeviceT* local_rd_initializer = nullptr) {
        if (num_iterations == 0) {
            return;
        }

        if constexpr (!parallel) {
            (void)local_rd_initializer;
            LOG(INFO) << "Entering sequential stationary loop";
            for (size_t iter_num = 0; iter_num < num_iterations; ++iter_num) {
                for (size_t i = 0; i < cluster_.size(); ++i) {
                    kernel_->Evolve(cluster_[i], &secondary_cluster_[i]);
                }
                std::swap(static_cast<ParticleCluster&>(cluster_), secondary_cluster_);
            }
            LOG(INFO) << "Finished sequential stationary loop";
        } else {
            std::vector<RandomDeviceT> rds;
            rds.reserve(cluster_.size());
            assert(local_rd_initializer);
            for (size_t i = 0; i < cluster_.size(); ++i) {
                rds.emplace_back((*local_rd_initializer)());
            }

            LOG(INFO) << "Entering parallel stationary loop";
            ParallelFor{0, cluster_.size(), 1}([&](size_t i) {
                for (size_t iter_num = 0; iter_num < num_iterations; ++iter_num) {
                    Particle<MemoryView>*from, *to;
                    if (iter_num % 2) {
                        from = &secondary_cluster_[i];
                        to = &cluster_[i];
                    } else {
                        from = &cluster_[i];
                        to = &secondary_cluster_[i];
                    }
                    kernel_->Evolve(*from, to, &rds[i]);
                }
            });
            if (!(num_iterations % 2)) {
                std::swap(static_cast<ParticleCluster&>(cluster_), secondary_cluster_);
            }
            LOG(INFO) << "Finished stationary";
        }

        MakeWeighing();
    }

    const VectorWeightedParticleCluster& GetCluster() const {
        return cluster_;
    }

    VectorWeightedParticleCluster& GetCluster() {
        return cluster_;
    }

    const RNGKernel& GetKernel() const {
        return *kernel_;
    }

    void ResetKernel(RNGKernel* new_kernel) {
        kernel_ = new_kernel;
    }

private:
    void MakeWeighing() {
        if (auto hintable_ptr = dynamic_cast<HintableKernel*>(kernel_)) {
            MakeWeighingHintable(*hintable_ptr);
        } else {
            MakeWeighingUsual();
        }
    }

    void MakeWeighingUsual() {
        ParallelFor{0, cluster_.size(), 1}([&](size_t i) {
            const auto& particle = cluster_[i];
            FloatT& particle_weight = cluster_.GetMutableWeights()[i];
            particle_weight = 0;
            for (const auto& from_particle : cluster_) {
                particle_weight +=
                    kernel_->GetTransDensity(from_particle, particle) / cluster_.size();
            }
        });
    }

    void MakeWeighingHintable(const HintableKernel& hintable_kernel) {
        using HintT = decltype(hintable_kernel.CalculateHint(cluster_[0]));
        std::vector<HintT> hints(cluster_.size());
        ParallelFor{0, cluster_.size(),
                    1}([&](size_t i) { hints[i] = hintable_kernel.CalculateHint(cluster_[i]); });

        ParallelFor{0, cluster_.size(), 1}([&](size_t i) {
            const auto& particle = cluster_[i];
            FloatT& particle_weight = cluster_.GetMutableWeights()[i];
            particle_weight = 0;
            for (size_t from_ix = 0; from_ix < cluster_.size(); ++from_ix) {
                const auto& from_particle = cluster_[from_ix];
                HintT* hint = &hints[from_ix];
                particle_weight +=
                    hintable_kernel.GetTransDensityWithHint(from_particle, particle, hint) /
                    cluster_.size();
            }
        });
    }

    RNGKernel* kernel_;
    VectorWeightedParticleCluster cluster_;
    ParticleCluster secondary_cluster_;
};
