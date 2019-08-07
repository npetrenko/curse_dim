#include <bellman/bellman_operators/uniform_operator.hpp>
#include <bellman/types.hpp>
#include <thread_pool/for_loop.hpp>
#include <glog/logging.h>

#ifndef NDEBUG
#include <fenv.h>
#endif

UniformBellmanOperator::UniformBellmanOperator(AbstractBellmanOperator::Params&& params,
                                               Params&& unif_params)
    : AbstractBellmanOperator(std::move(params)), kParams(std::move(unif_params)) {
}

std::unique_ptr<UniformBellmanOperator> UniformBellmanOperator::Builder::BuildImpl(
    AbstractBellmanOperator::Params&& params) && {
    VLOG(4) << "Started building UniformBellmanOperator";
    Params unif_params{init_radius_.value()};

    auto op = std::unique_ptr<UniformBellmanOperator>(
        new UniformBellmanOperator(std::move(params), std::move(unif_params)));

    op->additional_weights_ = Matrix(
        {static_cast<MatrixDims::value_type>(num_particles_.value()),
         static_cast<MatrixDims::value_type>(op->GetEnvParams().ac_kernel->GetNumActions())});

    op->qfunc_primary_ =
        DiscreteQFuncEst{NumParticles(num_particles_.value()),
                         NumActions(op->GetEnvParams().ac_kernel->GetNumActions())};

    op->qfunc_secondary_ =
        DiscreteQFuncEst{NumParticles(num_particles_.value()),
                         NumActions(op->GetEnvParams().ac_kernel->GetNumActions())};

    std::uniform_real_distribution<FloatT> distr{-init_radius_.value(), init_radius_.value()};
    RandomVectorizingInitializer initializer{
        ParticleDim{op->GetEnvParams().ac_kernel->GetSpaceDim()}, random_device_.value(), distr,
        ClusterInitializationTag()};

    VLOG(4) << "Setting ParticleCluster";
    {
        auto particle_cluster =
            std::make_shared<ParticleCluster>(NumParticles(num_particles_.value()), initializer);
        op->qfunc_primary_.SetParticleCluster(particle_cluster);
        op->qfunc_secondary_.SetParticleCluster(std::move(particle_cluster));
    }
    {
        VLOG(4) << "Initializing QFunctions";
        std::uniform_real_distribution<FloatT> q_init{-0.01, 0.01};
        op->qfunc_primary_.SetRandom(random_device_.value(), q_init);
        op->qfunc_secondary_.SetRandom(random_device_.value(), q_init);
    }

    op->NormalizeWeights();
    {
        FloatT weight =
            pow(1 / (2 * init_radius_.value()), op->GetEnvParams().ac_kernel->GetSpaceDim());
        VLOG(4) << "Initializing ConstantWeightedParticleCluster as sampling distribution";
        op->sampling_distribution_ = std::make_unique<ConstantWeightedParticleCluster>(
            op->qfunc_primary_.GetParticleCluster(), weight);
    }
    return op;
}

void UniformBellmanOperator::MakeIteration() {
#ifndef NDEBUG
    feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif
    const ParticleCluster& cluster = qfunc_primary_.GetParticleCluster();
    assert(additional_weights_.Dims()[0] == cluster.size());
    assert(additional_weights_.Dims()[1] == GetEnvParams().ac_kernel->GetNumActions());

    GreedyPolicy policy{qfunc_primary_};

    const auto* ac_kernel = GetEnvParams().ac_kernel.get();

    ParallelFor{0, cluster.size(), 32}([&, num_actions = ac_kernel->GetNumActions()](size_t i) {
        MemoryView update_view = qfunc_secondary_.ValueAtIndex(i);
        for (size_t action_number = 0; action_number < num_actions; ++action_number) {
            update_view[action_number] = GetEnvParams().reward_function(cluster[i], action_number);
        }
        for (size_t j = 0; j < cluster.size(); ++j) {
            ConstMemoryView qfunc_primary_view = qfunc_primary_.ValueAtIndex(j);
            for (size_t action_number = 0; action_number < num_actions; ++action_number) {
                size_t reaction = policy.React(j);
                FloatT density =
                    ac_kernel->GetTransDensityConditionally(cluster[i], cluster[j], action_number);
                update_view[action_number] +=
                    GetEnvParams().gamma * density * qfunc_primary_view[reaction] *
                    additional_weights_(i, action_number) / cluster.size();
            }
        }
    });

    std::swap(qfunc_primary_, qfunc_secondary_);
}

void UniformBellmanOperator::NormalizeWeights() {
    VLOG(4) << "Normalizing weights";
    const ParticleCluster& cluster = qfunc_primary_.GetParticleCluster();
    const auto num_actions = GetEnvParams().ac_kernel->GetNumActions();
    const auto cluster_size = cluster.size();
    ParallelFor{0, cluster_size * num_actions, 32}([&](size_t ac_i) {
        size_t action_number = ac_i % num_actions;
        size_t i = (ac_i - action_number) / num_actions;
        FloatT sum = 0;
        for (size_t j = 0; j < cluster_size; ++j) {
            sum += GetEnvParams().ac_kernel->GetTransDensityConditionally(cluster[i], cluster[j],
                                                                          action_number) /
                   cluster_size;
        }
        // Importance sampling correction is also included into sum, so it may be not
        // close to 1
        additional_weights_(i, action_number) = sum ? 1 / sum : 0;
    });
    VLOG(4) << "Finished normalizing weights";
}