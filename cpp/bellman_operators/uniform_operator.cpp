#include <include/bellman_operators/uniform_operator.hpp>
#include <include/types.hpp>

UniformBellmanOperator UniformBellmanOperator::Builder::BuildImpl() && {
    UniformBellmanOperator op;
    op.env_params_ = std::move(env_params_.value());
    op.radius_ = init_radius_.value();
    op.random_device_ = random_device_.value();
    op.additional_weights_ =
        Matrix({static_cast<MatrixDims::value_type>(num_particles_.value()),
                static_cast<MatrixDims::value_type>(op.env_params_.ac_kernel->GetNumActions())});

    op.qfunc_primary_ =
        DiscreteQFuncEst{num_particles_.value(), op.env_params_.ac_kernel->GetNumActions()};
    op.qfunc_secondary_ =
        DiscreteQFuncEst{num_particles_.value(), op.env_params_.ac_kernel->GetNumActions()};
    std::uniform_real_distribution<FloatT> distr{-init_radius_.value(), init_radius_.value()};
    RandomVectorizingInitializer<MemoryView, decltype(distr), std::mt19937> initializer{
        op.env_params_.ac_kernel->GetSpaceDim(), random_device_.value(), distr};

    op.qfunc_primary_.SetParticleCluster(ParticleCluster{num_particles_.value(), initializer});
    {
        std::uniform_real_distribution<FloatT> q_init{-0.01, 0.01};
        op.qfunc_primary_.SetRandom(random_device_.value(), q_init);
        op.qfunc_secondary_.SetRandom(random_device_.value(), q_init);
    }

    op.NormalizeWeights();
    {
        FloatT weight =
            pow(1 / (2 * init_radius_.value()), op.env_params_.ac_kernel->GetSpaceDim());
        op.sampling_distribution_ = std::make_unique<ConstantWeightedParticleCluster>(
            op.qfunc_primary_.GetParticleCluster(), weight);
    }
    return op;
}

void UniformBellmanOperator::MakeIteration() {
#ifndef NDEBUG
    feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif
    auto& cluster = qfunc_primary_.GetParticleCluster();
    assert(additional_weights_.Dims()[0] == cluster.size());
    assert(additional_weights_.Dims()[1] == env_params_.ac_kernel->GetNumActions());

    GreedyPolicy policy{qfunc_primary_};

    auto& ac_kernel = env_params_.ac_kernel;

    ParallelFor{0, cluster.size(), 32}([&](size_t i) {
        auto num_actions = ac_kernel->GetNumActions();
        for (size_t action_number = 0; action_number < num_actions; ++action_number) {
            qfunc_secondary_.ValueAtIndex(i)[action_number] =
                env_params_.reward_function(cluster[i], action_number);
        }
        for (size_t j = 0; j < cluster.size(); ++j) {
            for (size_t action_number = 0; action_number < num_actions; ++action_number) {
                size_t reaction = policy.React(j);
                FloatT density =
                    ac_kernel->GetTransDensityConditionally(cluster[i], cluster[j], action_number);
                qfunc_secondary_.ValueAtIndex(i)[action_number] +=
                    env_params_.kGamma * density * qfunc_primary_.ValueAtIndex(j)[reaction] *
                    additional_weights_(i, action_number) / cluster.size();
            }
        }
    });

    std::swap(qfunc_primary_, qfunc_secondary_);
    qfunc_primary_.SetParticleCluster(std::move(qfunc_secondary_.GetParticleCluster()));
}

void UniformBellmanOperator::NormalizeWeights() {
    auto& cluster = qfunc_primary_.GetParticleCluster();
    auto num_actions = env_params_.ac_kernel->GetNumActions();
    for (size_t action_number = 0; action_number < num_actions; ++action_number) {
        ParallelFor{0, cluster.size(), 1}([&](size_t i) {
            FloatT sum = 0;
            for (size_t j = 0; j < cluster.size(); ++j) {
                sum += env_params_.ac_kernel->GetTransDensityConditionally(
                           /*from*/ cluster[i], cluster[j], action_number) /
                       cluster.size();
            }
            // Importance sampling correction is also included into sum, so it may be not
            // close to 1
            additional_weights_(i, action_number) = 1 / sum;
        });
    }
}
