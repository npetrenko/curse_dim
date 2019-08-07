#include <bellman/initializer.hpp>
#include <bellman/particle.hpp>
#include <bellman/kernel.hpp>
#include <bellman/particle_storage.hpp>
#include <bellman/bellman.hpp>
#include <bellman/bellman_operators/uniform_operator.hpp>
#include <bellman/bellman_operators/stationary_operator.hpp>
#include <bellman/bellman_operators/qfunc.hpp>
#include <bellman/bellman_operators/environment.hpp>
#include <bellman/density_estimators/stationary_estimator.hpp>

#include <gtest/gtest.h>
#include <glog/logging.h>
#include <random>

template <int step>
class DetermKernel final : public EnableClone<DetermKernel<step>, InheritFrom<RNGKernel>> {
    inline void EvolveImpl(TypeErasedParticleRef from, TypeErasedParticlePtr to,
                           std::mt19937*) const override {
        for (size_t i = 0; i < from.GetDim(); ++i) {
            (*to)[i] = from[i] + step;
        }
    }

    FloatT GetTransDensityImpl(TypeErasedParticleRef, TypeErasedParticleRef) const override {
        throw NotImplementedError();
    }

public:
    size_t GetSpaceDim() const override {
        return 0;
    }
};

class DummyPolicy final : public EnableClone<DummyPolicy, InheritFrom<IAgentPolicy>> {
public:
    size_t React(size_t) const override {
        throw NotImplementedError();
    }

    size_t React(TypeErasedParticleRef) const override {
        return (step++) != 0;
    }

private:
    mutable size_t step = 0;
};

TEST(Basic, AbstractKernelWorks) {
    ParticleStorage storage{1024};
    DetermKernel<1> kernel{};

    Particle test_particle{ZeroInitializer(ParticleDim{8}, &storage)};

    kernel.Evolve(test_particle, &test_particle);
    {
        auto expected = Particle{ConstantInitializer(1., ParticleDim{8})};
        ASSERT_EQ(test_particle, expected);
    }

    kernel.Evolve(test_particle, &test_particle);
    {
        auto expected = Particle{ConstantInitializer(2., ParticleDim{8})};
        ASSERT_EQ(test_particle, expected);
    }
}

TEST(Basic, ActionConditionedKernelWorks) {
    ParticleStorage storage{1024};
    ActionConditionedKernel kernel{DetermKernel<1>{}, DetermKernel<2>{}};

    Particle test_particle{ZeroInitializer(ParticleDim{8}, &storage)};

    kernel.EvolveConditionally(test_particle, &test_particle, 0);
    ASSERT_EQ(test_particle, Particle{ConstantInitializer(1., ParticleDim{8})});

    kernel.EvolveConditionally(test_particle, &test_particle, 1);
    ASSERT_EQ(test_particle, Particle{ConstantInitializer(3., ParticleDim{8})});
}

TEST(Basic, DeterministicKernelWorks) {
    ParticleStorage storage{1024};
    DummyPolicy policy;
    ActionConditionedKernel ac_kernel{DetermKernel<1>{}, DetermKernel<2>{}};
    /*
    static_assert(
        std::is_same_v<
            type_traits::DeepestCRTPType<AbstractConditionedKernel<decltype(ac_kernel), false>>,
            decltype(ac_kernel)>);
    */
    MDPKernel kernel{ActionConditionedKernel{DetermKernel<1>{}, DetermKernel<2>{}}, &policy};

    Particle test_particle{ZeroInitializer(ParticleDim{8}, &storage)};

    kernel.Evolve(test_particle, &test_particle);
    ASSERT_EQ(test_particle, Particle{ConstantInitializer(1., ParticleDim{8})});

    kernel.Evolve(test_particle, &test_particle);
    ASSERT_EQ(test_particle, Particle{ConstantInitializer(3., ParticleDim{8})});
}

class ARKernel final : public EnableClone<ARKernel, InheritFrom<RNGKernel>> {
    using BaseT = EnableClone;

public:
    ARKernel(std::mt19937* random_device) : BaseT{random_device} {
    }

    size_t GetSpaceDim() const override {
        throw NotImplementedError();
    }

private:
    void EvolveImpl(TypeErasedParticleRef from, TypeErasedParticlePtr to,
                    std::mt19937* rd) const override {
        std::normal_distribution<FloatT> normal{0., 1.};

        for (size_t i = 0; i < from.GetDim(); ++i) {
            (*to)[i] = 0.5 * from[i] + normal(*rd);
        }
    }

    FloatT GetTransDensityImpl(TypeErasedParticleRef from,
                               TypeErasedParticleRef to) const override {
        thread_local ParticleStorage tmp_storage_{1024};

        Particle<MemoryView> res{ValueInitializer{from, &tmp_storage_}};
        // Particle<ParticleStorage> res{from};
        res *= -0.5;
        res += to;
        FloatT norm = res.NormSquared();
        FloatT result = exp(-norm / 2) / pow(2 * double(M_PI), from.GetDim() / 2);
        tmp_storage_.Clear();
        return result;
    }
};

namespace SimpleModel {
static const FloatT kActionDelta = 0.07;

template <int direction>
class Kernel final : public EnableClone<Kernel<direction>, InheritFrom<RNGKernel>> {
    using BaseT = EnableClone<Kernel<direction>, InheritFrom<RNGKernel>>;

public:
    Kernel() = delete;

    Kernel(std::mt19937* random_device) noexcept : BaseT{random_device} {
    }

    inline size_t GetSpaceDim() const override {
        return 1;
    }

private:
    void EvolveImpl(TypeErasedParticleRef from, TypeErasedParticlePtr to,
                    std::mt19937* random_device) const override {
        auto [travel_down, travel_up] = MaxAllowedTravelDists(from);
        std::uniform_real_distribution<FloatT> delta{travel_down, travel_up};

        (*to)[0] = delta(*random_device);
    }

    FloatT GetTransDensityImpl(TypeErasedParticleRef from,
                               TypeErasedParticleRef to) const override {
        auto [travel_down, travel_up] = MaxAllowedTravelDists(from);
        if (to[0] > travel_up || to[0] < travel_down) {
            return 0;
        } else {
            return 1. / (travel_up - travel_down);
        }
    }

    std::pair<FloatT, FloatT> MaxAllowedTravelDists(TypeErasedParticleRef from) const {
        auto truncate = [](FloatT val) { return std::max(-1., std::min(val, 1.)); };

        const FloatT disp = 0.1;
        FloatT travel_up = truncate(from[0] + kActionDelta * direction + disp);
        FloatT travel_down = truncate(from[0] + kActionDelta * direction - disp);

        if (travel_up >= 0.99) {
            travel_down = travel_up - 2 * disp;
        }

        if (travel_down <= -0.99) {
            travel_up = travel_down + 2 * disp;
        }

        return {travel_down, travel_up};
    }
};

struct RewardFunc {
    FloatT operator()(TypeErasedParticleRef state, size_t /*action*/) const {
        return state[0];
    }
};
}  // namespace SimpleModel

template <class KernelT>
FloatT GetExpectedMass(const size_t kDim) {
    std::mt19937 random_device{1234};

    KernelT kernel{&random_device};
    using InitT = RandomVectorizingInitializer<MemoryView, std::uniform_real_distribution<FloatT>,
                                               std::mt19937>;
    ParticleCluster cluster{
        NumParticles(1024 * 256),
        InitT{ParticleDim{kDim}, &random_device, std::uniform_real_distribution<FloatT>{-6, 6}}};
    Particle<ParticleStorage> origin{ZeroInitializer(ParticleDim{kDim})};

    FloatT prob = 0;
    for (const auto& elem : cluster) {
        prob += kernel.GetTransDensity(origin, elem) * (pow(12, kDim) / cluster.size());
    }

    return prob;
}

TEST(Probability, ARKernelSummsToOne) {
    FloatT prob = GetExpectedMass<ARKernel>(2);
    ASSERT_TRUE(prob >= 0.99 && prob <= 1.01);
}

TEST(Probability, SimpleModelKernelSummsToOne) {
    {
        FloatT prob = GetExpectedMass<SimpleModel::Kernel<-1>>(1);
        ASSERT_NEAR(prob, 1., 0.03);
    }
    {
        FloatT prob = GetExpectedMass<SimpleModel::Kernel<0>>(1);
        ASSERT_NEAR(prob, 1., 0.03);
    }
    {
        FloatT prob = GetExpectedMass<SimpleModel::Kernel<1>>(1);
        ASSERT_NEAR(prob, 1., 0.03);
    }
}

TEST(StationaryEstim, SimpleModel) {
    std::mt19937 rd{423};
    const size_t kClusterSize{4096 * 4};
    SimpleModel::Kernel<0> kernel{&rd};

    std::uniform_real_distribution<FloatT> init_distr{0., 0.2};

    std::unique_ptr<StationaryDensityEstimator> estimator;
    {
        StationaryDensityEstimator::Builder builder;
        builder.SetKernel(&kernel)
            .SetInitializer(RandomVectorizingInitializer{ParticleDim{1}, &rd, init_distr,
                                                         ClusterInitializationTag()})
            .SetClusterSize(kClusterSize);
        estimator = std::move(builder).Build();
    }
    estimator->MakeIteration(100, &rd);

    size_t leq_point{0};
    const FloatT point = 0.5;
    const FloatT perc = FloatT(3) / 4;
    for (const Particle<MemoryView>& part : estimator->GetCluster()) {
        if (part[0] < point) {
            ++leq_point;
        }
    }
    FloatT result = FloatT(leq_point) / kClusterSize;

    ASSERT_NEAR(result, perc, 0.08);

    FloatT density = estimator->GetCluster().GetWeights()[0];

    ASSERT_NEAR(density, 0.5, 0.1);
}

template <class MDPK, class QF>
void TraceTrajectoriesForSimpleModes(const MDPK& mdp_kernel, const QF& qfunc_est) {
    LOG(INFO) << "Testing policy";
    for (FloatT init : std::array{0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.}) {
        Particle state{ConstantInitializer(init, ParticleDim{1})};
        LOG(INFO) << "Init state: " << state;
        for (int i = 0; i < 10; ++i) {
            LOG(INFO) << state << " QFunc values: " << qfunc_est.ValueAtPoint(state) << "\n";
            mdp_kernel.Evolve(state, &state);
        }
        LOG(INFO) << "FINISHED TRACING\n";
        ASSERT_TRUE(state[0]);
    }
}

TEST(UniformBellman, SimpleModel) {
    LOG(INFO) << "Starting UniformBellman test";
    std::mt19937 rd{1234};
    std::unique_ptr<UniformBellmanOperator> bellman_op;
    ActionConditionedKernel action_conditioned_kernel{
        SimpleModel::Kernel<1>{&rd}, SimpleModel::Kernel<0>{&rd}, SimpleModel::Kernel<-1>{&rd}};

    EnvParams env_params{action_conditioned_kernel, SimpleModel::RewardFunc{}, 0.95};

    {
        UniformBellmanOperator::Builder builder;
        builder.SetEnvParams(env_params)
            .SetInitRadius(1.)
            .SetRandomDevice(&rd)
            .SetNumParticles(2048);
        bellman_op = std::move(builder).Build();
    }

    LOG(INFO) << "Starting iterations";
    for (int i = 0; i < 20; ++i) {
        bellman_op->MakeIteration();
    }
    LOG(INFO) << "Finished iterations";

    QFuncEstForGreedy qfunc_est{env_params, bellman_op->GetQFunc(),
                                // Correction for importance sampling
                                [](auto) { return 2.; }};
    GreedyPolicy policy{qfunc_est};
    MDPKernel mdp_kernel{action_conditioned_kernel, &policy};

    TraceTrajectoriesForSimpleModes(mdp_kernel, qfunc_est);
}

TEST(StationaryBellmanOperator, SimpleModel) {
    LOG(INFO) << "Starting StationaryBellman test";
    std::mt19937 rd{1234};
    ActionConditionedKernel action_conditioned_kernel{
        SimpleModel::Kernel<1>{&rd}, SimpleModel::Kernel<0>{&rd}, SimpleModel::Kernel<-1>{&rd}};
    EnvParams env_params{action_conditioned_kernel, SimpleModel::RewardFunc{}, 0.95};

    std::unique_ptr<StationaryBellmanOperator> bellman_op;
    {
        StationaryBellmanOperator::Builder builder;
        builder.SetEnvParams(env_params)
            .SetNumParticles(2048)
            .SetDensityRatioThreshold(100.)
            .SetInitRadius(1.)
            .SetInvariantDensityThreshold(1e-3)
            .SetNumBurninIter(1)

            .SetRandomDevice(&rd);

        bellman_op = std::move(builder).Build();
    }

    LOG(INFO) << "Starting iterations";
    for (int i = 0; i < 20; ++i) {
        bellman_op->MakeIteration();
    }
    LOG(INFO) << "Finished iterations";

    PrevSampleReweighingHelper rew_helper{&bellman_op->GetSamplingDistribution(), std::nullopt};
    QFuncEstForGreedy qfunc_est{env_params, bellman_op->GetQFunc(), rew_helper};

    GreedyPolicy policy{qfunc_est};
    MDPKernel mdp_kernel{action_conditioned_kernel, &policy};

    TraceTrajectoriesForSimpleModes(mdp_kernel, qfunc_est);
}
