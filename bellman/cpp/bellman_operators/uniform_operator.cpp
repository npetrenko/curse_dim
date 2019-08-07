#include <bellman/bellman_operators/uniform_operator.hpp>
#include <bellman/types.hpp>
#include <thread_pool/for_loop.hpp>
#include <glog/logging.h>
#include <bellman/matrix.hpp>

#ifndef NDEBUG
#include <fenv.h>
#endif

class UniformBellmanOperator::Impl final : public AbstractBellmanOperator {
    friend class UniformBellmanOperator::Builder;

public:
    Impl(UniformBellmanOperator::Builder&&);
    void MakeIteration() override;

    const DiscreteQFuncEst& GetQFunc() const& override {
        return qfunc_primary_;
    }

    DiscreteQFuncEst GetQFunc() && override {
        return std::move(qfunc_primary_);
    }

    const ConstantWeightedParticleCluster& GetSamplingDistribution() const override {
        return *sampling_distribution_;
    }

private:
    struct Params {
        FloatT init_radius;
    };

    const Params kParams;

    void NormalizeWeights();

    Matrix<std::vector<FloatT>> additional_weights_;

    DiscreteQFuncEst qfunc_primary_, qfunc_secondary_;
    std::unique_ptr<ConstantWeightedParticleCluster> sampling_distribution_;
};

const DiscreteQFuncEst& UniformBellmanOperator::GetQFunc() const& {
    return impl_->GetQFunc();
}

DiscreteQFuncEst UniformBellmanOperator::GetQFunc() && {
    return std::move(*impl_).GetQFunc();
}

void UniformBellmanOperator::MakeIteration() {
    impl_->MakeIteration();
}

const ConstantWeightedParticleCluster& UniformBellmanOperator::GetSamplingDistribution() const {
    return impl_->GetSamplingDistribution();
}

UniformBellmanOperator::~UniformBellmanOperator() = default;
UniformBellmanOperator::UniformBellmanOperator(Builder&& builder)
    : impl_(std::make_unique<Impl>(std::move(builder))) {
}

UniformBellmanOperator::Impl::Impl(UniformBellmanOperator::Builder&& builder)
    : AbstractBellmanOperator(std::move(builder)), kParams{builder.init_radius_.value()} {
}

std::unique_ptr<UniformBellmanOperator> UniformBellmanOperator::Builder::BuildImpl() && {
    VLOG(4) << "Started building UniformBellmanOperator";
    auto op = std::unique_ptr<UniformBellmanOperator>(new UniformBellmanOperator(std::move(*this)));
    auto* impl = op->impl_.get();

    impl->additional_weights_ = Matrix(
        {static_cast<MatrixDims::value_type>(num_particles_.value()),
         static_cast<MatrixDims::value_type>(impl->GetEnvParams().ac_kernel->GetNumActions())});

    impl->qfunc_primary_ =
        DiscreteQFuncEst{NumParticles(num_particles_.value()),
                         NumActions(impl->GetEnvParams().ac_kernel->GetNumActions())};

    impl->qfunc_secondary_ =
        DiscreteQFuncEst{NumParticles(num_particles_.value()),
                         NumActions(impl->GetEnvParams().ac_kernel->GetNumActions())};

    std::uniform_real_distribution<FloatT> distr{-init_radius_.value(), init_radius_.value()};
    RandomVectorizingInitializer initializer{
        ParticleDim{impl->GetEnvParams().ac_kernel->GetSpaceDim()}, random_device_.value(), distr,
        ClusterInitializationTag()};

    VLOG(4) << "Setting ParticleCluster";
    {
        auto particle_cluster =
            std::make_shared<ParticleCluster>(NumParticles(num_particles_.value()), initializer);
        impl->qfunc_primary_.SetParticleCluster(particle_cluster);
        impl->qfunc_secondary_.SetParticleCluster(std::move(particle_cluster));
    }
    {
        VLOG(4) << "Initializing QFunctions";
        std::uniform_real_distribution<FloatT> q_init{-0.01, 0.01};
        impl->qfunc_primary_.SetRandom(random_device_.value(), q_init);
        impl->qfunc_secondary_.SetRandom(random_device_.value(), q_init);
    }

    impl->NormalizeWeights();
    {
        FloatT weight =
            pow(1 / (2 * init_radius_.value()), impl->GetEnvParams().ac_kernel->GetSpaceDim());
        VLOG(4) << "Initializing ConstantWeightedParticleCluster as sampling distribution";
        impl->sampling_distribution_ = std::make_unique<ConstantWeightedParticleCluster>(
            impl->qfunc_primary_.GetParticleCluster(), weight);
    }
    return op;
}

void UniformBellmanOperator::Impl::MakeIteration() {
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

void UniformBellmanOperator::Impl::NormalizeWeights() {
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
