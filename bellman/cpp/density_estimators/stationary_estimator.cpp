#include <bellman/density_estimators/stationary_estimator.hpp>

#include <glog/logging.h>
#include <thread_pool/for_loop.hpp>

void StationaryDensityEstimator::MakeIteration(size_t num_iterations,
                                               std::mt19937* local_rd_initializer) {
    if (num_iterations == 0) {
        return;
    }

    std::vector<std::mt19937> rds;
    rds.reserve(cluster_->size());
    assert(local_rd_initializer);
    for (size_t i = 0; i < cluster_->size(); ++i) {
        rds.emplace_back((*local_rd_initializer)());
    }

    VLOG(4) << "Entering parallel stationary loop";
    ParallelFor{0, cluster_->size(), 1}([&](size_t i) {
        for (size_t iter_num = 0; iter_num < num_iterations; ++iter_num) {
            Particle<MemoryView>*from, *to;
            if (iter_num % 2) {
                from = &secondary_cluster_[i];
                to = &(*cluster_)[i];
            } else {
                from = &(*cluster_)[i];
                to = &secondary_cluster_[i];
            }
            kernel_->Evolve(*from, to, &rds[i]);
        }
    });
    if (!(num_iterations % 2)) {
        std::swap(static_cast<ParticleCluster&>(*cluster_), secondary_cluster_);
    }
    VLOG(4) << "Finished stationary";

    MakeWeighing();
}

namespace {
void MakeWeighingUsual(const IKernel& kernel, VectorWeightedParticleCluster* cluster) {
    VLOG(4) << "Computing usual weighing";
    ParallelFor{0, cluster->size(), 1}([&, weights = cluster->GetMutableWeights()](size_t i) {
        const auto& particle = (*cluster)[i];
        FloatT& particle_weight = weights[i];
        particle_weight = 0;
        for (const auto& from_particle : *cluster) {
            particle_weight += kernel.GetTransDensity(from_particle, particle) / cluster->size();
        }
    });
    VLOG(4) << "Finished";
}

void MakeWeighingHintable(const IHintableKernel& hintable_kernel,
                          VectorWeightedParticleCluster* cluster) {
    VLOG(4) << "Computing weighing for hintable kernel";
    using HintT = decltype(hintable_kernel.CalculateHint((*cluster)[0]));
    std::vector<HintT> hints(cluster->size());
    ParallelFor{0, cluster->size(),
                1}([&](size_t i) { hints[i] = hintable_kernel.CalculateHint((*cluster)[i]); });

    ParallelFor{0, cluster->size(), 1}([&, weights = cluster->GetMutableWeights()](size_t i) {
        const auto& particle = (*cluster)[i];
        FloatT& particle_weight = weights[i];
        particle_weight = 0;
        for (size_t from_ix = 0; from_ix < cluster->size(); ++from_ix) {
            const auto& from_particle = (*cluster)[from_ix];
            HintT* hint = &hints[from_ix];
            particle_weight +=
                hintable_kernel.GetTransDensityWithHint(from_particle, particle, hint) /
                cluster->size();
        }
    });
    VLOG(4) << "Finished weighing";
}
}  // namespace

void StationaryDensityEstimator::MakeWeighing() {
    if (auto hintable_ptr = dynamic_cast<IHintableKernel*>(kernel_)) {
        MakeWeighingHintable(*hintable_ptr, cluster_.get());
    } else {
        MakeWeighingUsual(*kernel_, cluster_.get());
    }
}

using Builder = StationaryDensityEstimator::Builder;
void Builder::MaybeInitPrimary() {
    if (!primary_cluster_) {
        try {
            primary_cluster_ = primary_cluster_builder_.value()();
        } catch (std::bad_optional_access&) {
            throw BuilderNotInitialized();
        }
    }
}

std::unique_ptr<StationaryDensityEstimator> Builder::Build() && {
    try {
        auto estimator =
            std::unique_ptr<StationaryDensityEstimator>(new StationaryDensityEstimator);
        estimator->kernel_ = kernel_.value();
        if (!primary_cluster_) {
            MaybeInitPrimary();
        }
        estimator->cluster_ = std::move(primary_cluster_.value());
        estimator->secondary_cluster_ = secondary_cluster_builder_.value()();
        return estimator;
    } catch (std::bad_optional_access&) {
        throw BuilderNotInitialized();
    }
}
