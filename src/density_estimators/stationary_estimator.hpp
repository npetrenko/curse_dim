#pragma once

#include <src/types.hpp>
#include <src/kernel.hpp>
#include <src/particle.hpp>
#include <src/particle_storage.hpp>
#include <src/util.hpp>
#include <src/type_traits.hpp>

#include <thread_pool/include/for_loop.hpp>

#include <glog/logging.h>

class AbstractWeightedParticleCluster : public ParticleCluster {
public:
    using ParticleCluster::ParticleCluster;
    AbstractWeightedParticleCluster(ParticleCluster cluster)
        : ParticleCluster{std::move(cluster)} {
    }

    virtual const std::vector<FloatT>& GetWeights() const = 0;
    virtual std::vector<FloatT>& GetMutableWeights() = 0;
    virtual ~AbstractWeightedParticleCluster() = default;
};

class VectorWeightedParticleCluster final : public AbstractWeightedParticleCluster {
public:
    template <class S>
    VectorWeightedParticleCluster(size_t size,
                                  const AbstractInitializer<S, MemoryView>& initializer)
        : AbstractWeightedParticleCluster{size, initializer}, weights_(size) {
    }

    inline const std::vector<FloatT>& GetWeights() const override {
        return weights_;
    }

    inline std::vector<FloatT>& GetMutableWeights() override {
        return weights_;
    }

private:
    std::vector<FloatT> weights_;
};

class ConstantWeightedParticleCluster final : public AbstractWeightedParticleCluster {
public:
    template <class S>
    ConstantWeightedParticleCluster(size_t size,
                                    const AbstractInitializer<S, MemoryView>& initializer,
                                    FloatT weighing_constant)
        : AbstractWeightedParticleCluster{size, initializer},
          weighing_constant_{weighing_constant} {
    }

    ConstantWeightedParticleCluster(ParticleCluster cluster, FloatT weighing_constant)
        : AbstractWeightedParticleCluster{std::move(cluster)},
          weighing_constant_{weighing_constant} {
    }

    inline std::vector<FloatT>& GetMutableWeights() override {
        throw std::runtime_error(
            "GetMutableWeights for ConstantWeightedParticleCluster makes no sense");
    }

    inline const std::vector<FloatT>& GetWeights() const override {
        MaybeInitialize();
        return weights_;
    }

private:
    inline void MaybeInitialize() const {
        if (!vector_is_ititialized_) {
            vector_is_ititialized_ = true;
            weights_.resize(this->size(), weighing_constant_);
        }
    }

    mutable std::vector<FloatT> weights_;
    const FloatT weighing_constant_;
    mutable bool vector_is_ititialized_{false};
};

template <class T, class HasRNGTag>
class StationaryDensityEstimator {
public:
    template <class S>
    StationaryDensityEstimator(AbstractKernel<T, HasRNGTag>* kernel,
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

    const AbstractKernel<T>& GetKernel() const {
        return *kernel_;
    }

    void ResetKernel(AbstractKernel<T, HasRNGTag>* new_kernel) {
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
            FloatT& particle_weight = cluster_.GetMutableWeights()[i];
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

    AbstractKernel<T, HasRNGTag>* kernel_;
    VectorWeightedParticleCluster cluster_;
    ParticleCluster secondary_cluster_;
};
