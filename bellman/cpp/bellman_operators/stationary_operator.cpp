#include <bellman/bellman_operators/stationary_operator.hpp>
#include <thread_pool/for_loop.hpp>
#include <glog/logging.h>

#ifndef NDEBUG
#include <fenv.h>
#endif

StationaryBellmanOperator::StationaryBellmanOperator(AbstractBellmanOperator::Params&& params,
                                                     Params&& stat_params)
    : AbstractBellmanOperator(std::move(params)), kParams(std::move(stat_params)) {
}
std::unique_ptr<StationaryBellmanOperator> StationaryBellmanOperator::Builder::BuildImpl(
    AbstractBellmanOperator::Params&& params) {
    LOG(INFO) << "Started building StationaryBellmanOperator";
    Params stat_params{invariant_density_threshold_.value(), density_ratio_threshold_.value(),
                       init_radius_.value()};

    auto op = std::unique_ptr<StationaryBellmanOperator>(
        new StationaryBellmanOperator(std::move(params), std::move(stat_params)));

    {
        LOG(INFO) << "Initializing QFunctions";
        auto initializer = [&] {
            return DiscreteQFuncEst{NumParticles{op->GetNumParticles()},
                                    NumActions{op->GetEnvParams().ac_kernel->GetNumActions()}};
        };
        op->qfunc_primary_ = initializer();
        op->qfunc_secondary_ = initializer();
    }
    op->additional_weights_ = Matrix<std::vector<FloatT>>(
        {static_cast<MatrixDims::value_type>(op->GetNumParticles()),
         static_cast<MatrixDims::value_type>(op->GetEnvParams().ac_kernel->GetNumActions())});

    assert(init_radius_ > 0);
    {
        std::uniform_real_distribution<FloatT> distr{-init_radius_.value(), init_radius_.value()};
        RandomVectorizingInitializer initializer{
            ParticleDim{op->GetEnvParams().ac_kernel->GetSpaceDim()}, op->GetRD(), distr,
            ClusterInitializationTag{}};

        LOG(INFO) << "Initializing density estimator";

        StationaryDensityEstimator::Builder builder;
        builder.SetClusterSize(op->GetNumParticles())
            .SetKernel(nullptr)
            .SetInitializer(initializer);

        op->qfunc_primary_.SetParticleCluster(builder.GetParticleClusterPtr());
        op->qfunc_secondary_.SetParticleCluster(builder.GetParticleClusterPtr());

        op->density_estimator_ = std::move(builder).Build();
    }
    {
        LOG(INFO) << "Initializing QFunction values";
        std::uniform_real_distribution<FloatT> q_init{-0.01, 0.01};
        op->qfunc_primary_.SetRandom(op->GetRD(), q_init);
        op->qfunc_secondary_.SetRandom(op->GetRD(), q_init);
    }

    LOG(INFO) << "Initializing particle cluster with " << num_burnin_.value() << " iterations";
    op->UpdateParticleCluster(num_burnin_.value());
    LOG(INFO) << "Cluster is initialized";

    return op;
}

const VectorWeightedParticleCluster& StationaryBellmanOperator::GetSamplingDistribution() const {
    if (!prev_sampling_distribution_.get()) {
        throw std::runtime_error("Sampling distribution has not been initialized");
    }
    return *prev_sampling_distribution_;
}

void StationaryBellmanOperator::MakeIteration() {
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

void StationaryBellmanOperator::UpdateParticleCluster(size_t num_iterations) {
    LOG(INFO) << "Started ParticleCluster update";
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

    LOG(INFO) << "Making stationary iterations";
    density_estimator_->MakeIteration(num_iterations, GetRD());
    LOG(INFO) << "Finished stationary iterations";

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
    LOG(INFO) << "Finished particle cluster update";
}

void StationaryBellmanOperator::RecomputeWeights() {
    LOG(INFO) << "Recomputing normalizing weights";
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
    LOG(INFO) << "Finished recomputing normalizing weights";
}
