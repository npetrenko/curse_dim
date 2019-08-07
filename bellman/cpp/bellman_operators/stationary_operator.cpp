#include <bellman/bellman_operators/stationary_operator.hpp>
#include <thread_pool/for_loop.hpp>
#include <glog/logging.h>
#include <bellman/density_estimators/stationary_estimator.hpp>
#include <bellman/matrix.hpp>

#ifndef NDEBUG
#include <fenv.h>
#endif

class StationaryBellmanOperator::Impl final : public AbstractBellmanOperator {
    friend class StationaryBellmanOperator::Builder;

public:
    Impl(StationaryBellmanOperator::Builder&&);

    void MakeIteration() override;

    const DiscreteQFuncEst& GetQFunc() const& override {
        return qfunc_primary_;
    }

    DiscreteQFuncEst GetQFunc() && override {
        return std::move(qfunc_primary_);
    }

    const VectorWeightedParticleCluster& GetSamplingDistribution() const override;

private:
    struct Params {
        FloatT invariant_density_threshold;
        FloatT density_ratio_threshold;
        FloatT init_radius;
    };

    static Params ConstructParams(StationaryBellmanOperator::Builder&& builder) {
        return {builder.invariant_density_threshold_.value(),
                builder.density_ratio_threshold_.value(), builder.init_radius_.value()};
    }

    const Params kParams;

    void UpdateParticleCluster(size_t num_iterations);

    void RecomputeWeights();

    DiscreteQFuncEst qfunc_primary_, qfunc_secondary_;
    std::unique_ptr<StationaryDensityEstimator> density_estimator_{nullptr};
    Matrix<std::vector<FloatT>> additional_weights_;
    std::unique_ptr<VectorWeightedParticleCluster> prev_sampling_distribution_{nullptr};
};

StationaryBellmanOperator::Impl::Impl(StationaryBellmanOperator::Builder&& builder)
    : AbstractBellmanOperator(std::move(builder)), kParams(ConstructParams(std::move(builder))) {
}

StationaryBellmanOperator::StationaryBellmanOperator(Builder&& builder)
    : impl_(std::make_unique<Impl>(std::move(builder))) {
}

StationaryBellmanOperator::~StationaryBellmanOperator() = default;

const DiscreteQFuncEst& StationaryBellmanOperator::GetQFunc() const& {
    return impl_->GetQFunc();
}

DiscreteQFuncEst StationaryBellmanOperator::GetQFunc() && {
    return std::move(*impl_).GetQFunc();
}

void StationaryBellmanOperator::MakeIteration() {
    impl_->MakeIteration();
}

const VectorWeightedParticleCluster& StationaryBellmanOperator::GetSamplingDistribution() const {
    return impl_->GetSamplingDistribution();
}

std::unique_ptr<StationaryBellmanOperator> StationaryBellmanOperator::Builder::BuildImpl() {
    VLOG(4) << "Started building StationaryBellmanOperator";

    auto op =
        std::unique_ptr<StationaryBellmanOperator>(new StationaryBellmanOperator(std::move(*this)));
    auto* impl = op->impl_.get();

    {
        VLOG(4) << "Initializing QFunctions";
        auto initializer = [&] {
            return DiscreteQFuncEst{NumParticles{impl->GetNumParticles()},
                                    NumActions{impl->GetEnvParams().ac_kernel->GetNumActions()}};
        };
        impl->qfunc_primary_ = initializer();
        impl->qfunc_secondary_ = initializer();
    }
    impl->additional_weights_ = Matrix<std::vector<FloatT>>(
        {static_cast<MatrixDims::value_type>(impl->GetNumParticles()),
         static_cast<MatrixDims::value_type>(impl->GetEnvParams().ac_kernel->GetNumActions())});

    assert(init_radius_ > 0);
    {
        std::uniform_real_distribution<FloatT> distr{-init_radius_.value(), init_radius_.value()};
        RandomVectorizingInitializer initializer{
            ParticleDim{impl->GetEnvParams().ac_kernel->GetSpaceDim()}, impl->GetRD(), distr,
            ClusterInitializationTag{}};

        VLOG(4) << "Initializing density estimator";

        StationaryDensityEstimator::Builder builder;
        builder.SetClusterSize(impl->GetNumParticles())
            .SetKernel(nullptr)
            .SetInitializer(initializer);

        impl->qfunc_primary_.SetParticleCluster(builder.GetParticleClusterPtr());
        impl->qfunc_secondary_.SetParticleCluster(builder.GetParticleClusterPtr());

        impl->density_estimator_ = std::move(builder).Build();
    }
    {
        VLOG(4) << "Initializing QFunction values";
        std::uniform_real_distribution<FloatT> q_init{-0.01, 0.01};
        impl->qfunc_primary_.SetRandom(impl->GetRD(), q_init);
        impl->qfunc_secondary_.SetRandom(impl->GetRD(), q_init);
    }

    VLOG(4) << "Initializing particle cluster with " << num_burnin_.value() << " iterations";
    impl->UpdateParticleCluster(num_burnin_.value());
    VLOG(4) << "Cluster is initialized";

    return op;
}

const VectorWeightedParticleCluster& StationaryBellmanOperator::Impl::GetSamplingDistribution()
    const {
    if (!prev_sampling_distribution_.get()) {
        throw std::runtime_error("Sampling distribution has not been initialized");
    }
    return *prev_sampling_distribution_;
}

void StationaryBellmanOperator::Impl::MakeIteration() {
#ifndef NDEBUG
    feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif
    UpdateParticleCluster(1);

    const ParticleCluster& cluster = qfunc_primary_.GetParticleCluster();
    GreedyPolicy policy{qfunc_primary_};

    const auto* ac_kernel = GetEnvParams().ac_kernel.get();

    ParallelFor{0, GetNumParticles(), 255}([&, num_actions = ac_kernel->GetNumActions(),
                                            num_particles = GetNumParticles()](size_t from_ix) {
        if (density_estimator_->GetCluster().GetWeights()[from_ix] <
            kParams.invariant_density_threshold) {
            return;
        }
        MemoryView update_view = qfunc_secondary_.ValueAtIndex(from_ix);
        for (size_t action_number = 0; action_number < num_actions; ++action_number) {
            update_view[action_number] =
                GetEnvParams().reward_function(cluster[from_ix], action_number);
        }
        for (size_t to_ix = 0; to_ix < num_particles; ++to_ix) {
            ConstMemoryView qfunc_primary_view = qfunc_primary_.ValueAtIndex(to_ix);
            for (size_t action_number = 0; action_number < num_actions; ++action_number) {
                FloatT stationary_density = density_estimator_->GetCluster().GetWeights()[to_ix];
                if (stationary_density < kParams.invariant_density_threshold) {
                    continue;
                }
                size_t reaction = policy.React(to_ix);
                FloatT weighted_density = ac_kernel->GetTransDensityConditionally(
                                              cluster[from_ix], cluster[to_ix], action_number) /
                                          stationary_density;

                if (qfunc_primary_.ValueAtIndex(to_ix)[reaction] > 1e2) {
                    LOG(FATAL) << "Bad qfunc value, terminating";
                }

                if (weighted_density > kParams.density_ratio_threshold) {
                    continue;
                }

                update_view[action_number] +=
                    GetGamma() * weighted_density * qfunc_primary_view[reaction] *
                    additional_weights_(from_ix, action_number) / num_particles;
            }
        }
    });

    std::swap(qfunc_primary_, qfunc_secondary_);
}

void StationaryBellmanOperator::Impl::UpdateParticleCluster(size_t num_iterations) {
    VLOG(4) << "Started ParticleCluster update";
#ifndef NDEBUG
    feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif
    PrevSampleReweighingHelper prev_sample_reweighing{
        prev_sampling_distribution_.get(),
        1 / pow(2 * kParams.init_radius, GetEnvParams().ac_kernel->GetSpaceDim())};

    QFuncEstForGreedy current_qfunc_estimator{GetEnvParams(), qfunc_primary_,
                                              // should be previous density here
                                              prev_sample_reweighing};
    GreedyPolicy policy{current_qfunc_estimator};
    std::unique_ptr<IMDPKernel> mdp_kernel = GetEnvParams().mdp_kernel->Clone();
    mdp_kernel->ResetPolicy(&policy);

    density_estimator_->ResetKernel(mdp_kernel.get());

    VLOG(4) << "Making stationary iterations";
    density_estimator_->MakeIteration(num_iterations, GetRD());
    VLOG(4) << "Finished stationary iterations";

    const VectorWeightedParticleCluster& invariant_distr = density_estimator_->GetCluster();

    // Computing qfunction on new sample
    {
        DiscreteQFuncEst new_estimate{NumParticles(GetNumParticles()),
                                      NumActions(GetEnvParams().ac_kernel->GetNumActions())};

        ParallelFor{0, GetNumParticles(), 255}([&](size_t state_ix) {
            MemoryView new_estimate_view = new_estimate.ValueAtIndex(state_ix);
            for (size_t action_num = 0; action_num < GetEnvParams().ac_kernel->GetNumActions();
                 ++action_num) {
                new_estimate_view[action_num] =
                    current_qfunc_estimator.ValueAtPoint(invariant_distr[state_ix], action_num);
            }
        });

        new_estimate.SetParticleCluster(density_estimator_->GetClusterPtr());
        qfunc_primary_ = std::move(new_estimate);
    }

    mdp_kernel->ResetPolicy(nullptr);
    density_estimator_->ResetKernel(nullptr);
    prev_sampling_distribution_ = std::make_unique<VectorWeightedParticleCluster>(invariant_distr);

    RecomputeWeights();
    VLOG(4) << "Finished particle cluster update";
}

void StationaryBellmanOperator::Impl::RecomputeWeights() {
    VLOG(4) << "Recomputing normalizing weights";
    const VectorWeightedParticleCluster& invariant_distr = density_estimator_->GetCluster();
    const auto num_actions = GetEnvParams().ac_kernel->GetNumActions();
    ParallelFor{0, GetNumParticles() * num_actions,
                32}([&, num_particles = GetNumParticles()](size_t t_p_a) {
        size_t action_num = t_p_a % num_actions;
        size_t target_ix = (t_p_a - action_num) / num_actions;
        FloatT sum = 0;
        for (size_t particle_ix = 0; particle_ix < num_particles; ++particle_ix) {
            const auto& particle = invariant_distr[particle_ix];

            // ensure that checks run only once
            if (!target_ix && !action_num) {
                assert(particle == qfunc_primary_.GetParticleCluster()[particle_ix]);
            }

            FloatT pmass = invariant_distr.GetWeights()[particle_ix];
            if (pmass == 0) {
                continue;
            }
            sum += GetEnvParams().ac_kernel->GetTransDensityConditionally(
                       qfunc_primary_.GetParticleCluster()[target_ix], particle, action_num) /
                   (pmass * num_particles);
        }
        additional_weights_(target_ix, action_num) = sum ? (1 / sum) : 0;

        /*
        if ((sum > 30) || (sum && (sum < 0.1))) {
            LOG(FATAL) << "Strange additional weights: " << 1 / sum;
        }
        */
    });
    VLOG(4) << "Finished recomputing normalizing weights";
}
