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
    LOG(INFO) << "Started building UniformBellmanoperator";
    Params unif_params{init_radius_.value()};

    auto op = std::make_unique<UniformBellmanOperator>(
        UniformBellmanOperator(std::move(params), std::move(unif_params)));

    op->additional_weights_ = Matrix(
        {static_cast<MatrixDims::value_type>(num_particles_.value()),
         static_cast<MatrixDims::value_type>(op->GetEnvParams().ac_kernel->GetNumActions())});

    op->qfunc_primary_ =
        DiscreteQFuncEst{num_particles_.value(), op->GetEnvParams().ac_kernel->GetNumActions()};

    op->qfunc_secondary_ =
        DiscreteQFuncEst{num_particles_.value(), op->GetEnvParams().ac_kernel->GetNumActions()};

    std::uniform_real_distribution<FloatT> distr{-init_radius_.value(), init_radius_.value()};
    RandomVectorizingInitializer<MemoryView, decltype(distr), std::mt19937> initializer{
        ParticleDim{op->GetEnvParams().ac_kernel->GetSpaceDim()}, random_device_.value(), distr};

    LOG(INFO) << "Setting Particle cluster";
    op->qfunc_primary_.SetParticleCluster(ParticleCluster{num_particles_.value(), initializer});
    {
        LOG(INFO) << "Initializing QFunctions";
        std::uniform_real_distribution<FloatT> q_init{-0.01, 0.01};
        op->qfunc_primary_.SetRandom(random_device_.value(), q_init);
        op->qfunc_secondary_.SetRandom(random_device_.value(), q_init);
    }

    op->NormalizeWeights();
    {
        FloatT weight =
            pow(1 / (2 * init_radius_.value()), op->GetEnvParams().ac_kernel->GetSpaceDim());
        LOG(INFO) << "Initializing Constantweightedparticlecluster as sampling distribution";
        op->sampling_distribution_ = std::make_unique<ConstantWeightedParticleCluster>(
            op->qfunc_primary_.GetParticleCluster(), weight);
    }
    return op;
}

void UniformBellmanOperator::MakeIteration() {
#ifndef NDEBUG
    feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif
    auto& cluster = qfunc_primary_.GetParticleCluster();
    assert(additional_weights_.Dims()[0] == cluster.size());
    assert(additional_weights_.Dims()[1] == GetEnvParams().ac_kernel->GetNumActions());

    GreedyPolicy policy{qfunc_primary_};

    const auto& ac_kernel = GetEnvParams().ac_kernel;

    ParallelFor{0, cluster.size(), 32}([&](size_t i) {
        auto num_actions = ac_kernel->GetNumActions();
        for (size_t action_number = 0; action_number < num_actions; ++action_number) {
            qfunc_secondary_.ValueAtIndex(i)[action_number] =
                GetEnvParams().reward_function(cluster[i], action_number);
        }
        for (size_t j = 0; j < cluster.size(); ++j) {
            for (size_t action_number = 0; action_number < num_actions; ++action_number) {
                size_t reaction = policy.React(j);
                FloatT density =
                    ac_kernel->GetTransDensityConditionally(cluster[i], cluster[j], action_number);
                qfunc_secondary_.ValueAtIndex(i)[action_number] +=
                    GetEnvParams().gamma * density * qfunc_primary_.ValueAtIndex(j)[reaction] *
                    additional_weights_(i, action_number) / cluster.size();
            }
        }
    });

    std::swap(qfunc_primary_, qfunc_secondary_);
    qfunc_primary_.SetParticleCluster(std::move(qfunc_secondary_.GetParticleCluster()));
}

void UniformBellmanOperator::NormalizeWeights() {
    LOG(INFO) << "Normalizing weights";
    auto& cluster = qfunc_primary_.GetParticleCluster();
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
    LOG(INFO) << "Finished normalizing wieghts";
}
