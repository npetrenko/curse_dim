#pragma once

#include <src/types.hpp>
#include <src/kernel.hpp>
#include <src/particle.hpp>
#include <src/particle_storage.hpp>
#include <src/util.hpp>
#include <src/type_traits.hpp>

#include <thread_pool/include/for_loop.hpp>

#include <glog/logging.h>

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

template <class T, bool HasRNG>
class StationaryDensityEstimator {
public:
    template <class S>
    StationaryDensityEstimator(AbstractKernel<T, HasRNG>* kernel,
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

    const WeightedParticleCluster& GetCluster() const {
        return cluster_;
    }

    WeightedParticleCluster& GetCluster() {
        return cluster_;
    }

    const AbstractKernel<T>& GetKernel() const {
        return *kernel_;
    }

    void ResetKernel(AbstractKernel<T, HasRNG>* new_kernel) {
        kernel_ = new_kernel;
    }

private:
    void MakeWeighing() {
        if constexpr (type_traits::IsHintable_v<T>) {
            MakeWeighingHintable(static_cast<T&>(*kernel_));
        } else {
            MakeWeighingUsual();
        }
    }

    void MakeWeighingUsual() {
        ParallelFor{0, cluster_.size(), 1}([&](size_t i) {
            const auto& particle = cluster_[i];
            FloatT& particle_weight = cluster_.GetWeights()[i];
            particle_weight = 0;
            for (const auto& from_particle : cluster_) {
                particle_weight +=
                    kernel_->GetTransDensity(from_particle, particle) / cluster_.size();
            }
        });
    }

    template <class DerivedT>
    void MakeWeighingHintable(const HintableKernel<DerivedT>& hintable_kernel) {
        using HintT = decltype(hintable_kernel.CalculateHint(cluster_[0]));
        std::vector<HintT> hints(cluster_.size());
        ParallelFor{0, cluster_.size(),
                    1}([&](size_t i) { hints[i] = hintable_kernel.CalculateHint(cluster_[i]); });

        ParallelFor{0, cluster_.size(), 1}([&](size_t i) {
            const auto& particle = cluster_[i];
            FloatT& particle_weight = cluster_.GetWeights()[i];
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

    AbstractKernel<T, HasRNG>* kernel_;
    WeightedParticleCluster cluster_;
    ParticleCluster secondary_cluster_;
};
