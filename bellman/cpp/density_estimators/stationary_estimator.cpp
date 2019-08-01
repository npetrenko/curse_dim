#include <bellman/density_estimators/stationary_estimator.hpp>

#include <thread_pool/for_loop.hpp>

void StationaryDensityEstimator::MakeIteration(size_t num_iterations,
                                               std::mt19937* local_rd_initializer) {
    if (num_iterations == 0) {
        return;
    }

    std::vector<std::mt19937> rds;
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

    MakeWeighing();
}

namespace {
void MakeWeighingUsual(const IKernel& kernel, VectorWeightedParticleCluster* cluster) {
    ParallelFor{0, cluster->size(), 1}([&, weights = cluster->GetMutableWeights()](size_t i) {
        const auto& particle = (*cluster)[i];
        FloatT& particle_weight = weights[i];
        particle_weight = 0;
        for (const auto& from_particle : *cluster) {
            particle_weight += kernel.GetTransDensity(from_particle, particle) / cluster->size();
        }
    });
}

void MakeWeighingHintable(const IHintableKernel& hintable_kernel,
                          VectorWeightedParticleCluster* cluster) {
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
}
}  // namespace

void StationaryDensityEstimator::MakeWeighing() {
    if (auto hintable_ptr = dynamic_cast<IHintableKernel*>(kernel_)) {
        MakeWeighingHintable(*hintable_ptr, &cluster_);
    } else {
        MakeWeighingUsual(*kernel_, &cluster_);
    }
}
