#include <src/initializer.hpp>
#include <src/particle.hpp>
#include <src/kernel.hpp>
#include <src/particle_storage.hpp>
#include <src/bellman.hpp>
#include <src/bellman_operators/uniform_operator.hpp>
#include <src/bellman_operators/stationary_operator.hpp>
#include <src/bellman_operators/qfunc.hpp>
#include <src/bellman_operators/environment.hpp>

#include <gtest/gtest.h>
#include <random>

template <int step>
class DetermKernel : public AbstractKernel<DetermKernel<step>, std::false_type> {
    friend class AbstractKernel<DetermKernel<step>, std::false_type>;

private:
    template <class S1, class S2, class RandomDeviceT>
    inline void EvolveImpl(const Particle<S1>& from, Particle<S2>* to, RandomDeviceT*) const {
        for (size_t i = 0; i < from.GetDim(); ++i) {
            (*to)[i] = from[i] + step;
        }
    }
};

class DummyPolicy : public AbstractAgentPolicy<DummyPolicy> {
public:
    template <class S>
    size_t React(const Particle<S>&) {
        return (step++) != 0;
    }

private:
    size_t step = 0;
};

TEST(Basic, AbstractKernelWorks) {
    ParticleStorage storage{1024};
    DetermKernel<1> kernel{};

    Particle test_particle{ZeroInitializer(8, &storage)};

    kernel.Evolve(test_particle, &test_particle);
    {
        auto expected = Particle{ConstantInitializer(1., 8)};
        ASSERT_EQ(test_particle, expected);
    }

    kernel.Evolve(test_particle, &test_particle);
    {
        auto expected = Particle{ConstantInitializer(2., 8)};
        ASSERT_EQ(test_particle, expected);
    }
}

TEST(Basic, ActionConditionedKernelWorks) {
    ParticleStorage storage{1024};
    ActionConditionedKernel kernel{DetermKernel<1>{}, DetermKernel<2>{}};

    Particle test_particle{ZeroInitializer(8, &storage)};

    kernel.EvolveConditionally(test_particle, &test_particle, 0);
    ASSERT_EQ(test_particle, Particle{ConstantInitializer(1., 8)});

    kernel.EvolveConditionally(test_particle, &test_particle, 1);
    ASSERT_EQ(test_particle, Particle{ConstantInitializer(3., 8)});
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

    Particle test_particle{ZeroInitializer(8, &storage)};

    kernel.Evolve(test_particle, &test_particle);
    ASSERT_EQ(test_particle, Particle{ConstantInitializer(1., 8)});

    kernel.Evolve(test_particle, &test_particle);
    ASSERT_EQ(test_particle, Particle{ConstantInitializer(3., 8)});
}

class ARKernel : public AbstractKernel<ARKernel> {
    friend class AbstractKernel<ARKernel>;
    using BaseT = AbstractKernel<ARKernel>;

public:
    ARKernel(std::mt19937* random_device) : BaseT{random_device} {
    }

private:
    template <class S1, class S2, class RandomDeviceT>
    inline void EvolveImpl(const Particle<S1>& from, Particle<S2>* to, RandomDeviceT* rd) const {
        std::normal_distribution<FloatT> normal{0., 1.};

        for (size_t i = 0; i < from.GetDim(); ++i) {
            (*to)[i] = 0.5 * from[i] + normal(*rd);
        }
    }

    template <class S1, class S2>
    inline FloatT GetTransDensityImpl(const Particle<S1>& from, const Particle<S2>& to) const {
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
class Kernel : public AbstractKernel<Kernel<direction>> {
    friend class AbstractKernel<Kernel<direction>>;
    using BaseT = AbstractKernel<Kernel<direction>>;

public:
    Kernel(std::mt19937* random_device) noexcept : BaseT{random_device} {
    }

    inline size_t GetSpaceDim() const {
        return 1;
    }

private:
    template <class S1, class S2, class RandomDeviceT>
    void EvolveImpl(const Particle<S1>& from, Particle<S2>* to,
                    RandomDeviceT* random_device) const {
        auto [travel_down, travel_up] = MaxAllowedTravelDists(from);
        std::uniform_real_distribution<FloatT> delta{travel_down, travel_up};

        (*to)[0] = delta(*random_device);
    }

    template <class S1, class S2>
    FloatT GetTransDensityImpl(const Particle<S1>& from, const Particle<S2>& to) const {
        auto [travel_down, travel_up] = MaxAllowedTravelDists(from);
        if (to[0] > travel_up || to[0] < travel_down) {
            return 0;
        } else {
            return 1. / (travel_up - travel_down);
        }
    }

    template <class S1>
    std::pair<FloatT, FloatT> MaxAllowedTravelDists(const Particle<S1>& from) const {
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
    std::mt19937* rd_;
};

struct RewardFunc {
    template <class S>
    FloatT operator()(const Particle<S>& state, size_t /*action*/) const {
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
        1024 * 1024, InitT{kDim, &random_device, std::uniform_real_distribution<FloatT>{-6, 6}}};
    Particle<ParticleStorage> origin{ZeroInitializer(kDim)};

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
        ASSERT_TRUE(prob >= 0.99 && prob <= 1.01);
    }
    {
        FloatT prob = GetExpectedMass<SimpleModel::Kernel<0>>(1);
        ASSERT_TRUE(prob >= 0.99 && prob <= 1.01);
    }
    {
        FloatT prob = GetExpectedMass<SimpleModel::Kernel<1>>(1);
        ASSERT_TRUE(prob >= 0.99 && prob <= 1.01);
    }
}

TEST(StationaryEstim, SimpleModel) {
    std::mt19937 rd{423};
    const size_t kClusterSize{4096};
    SimpleModel::Kernel<0> kernel{&rd};

    std::uniform_real_distribution<FloatT> init_distr{0., 0.2};

    StationaryDensityEstimator estimator{
        &kernel,
        RandomVectorizingInitializer<MemoryView, decltype(init_distr), std::mt19937>{1, &rd,
                                                                                     init_distr},
        kClusterSize};
    estimator.MakeIteration(100);

    size_t leq_point{0};
    const FloatT point = 0.5;
    const FloatT perc = FloatT(3) / 4;
    for (const Particle<MemoryView>& part : estimator.GetCluster()) {
        if (part[0] < point) {
            ++leq_point;
        }
    }
    FloatT result = FloatT(leq_point) / kClusterSize;

    ASSERT_TRUE(result < perc + 0.08);
    ASSERT_TRUE(result > perc - 0.08);

    FloatT density = estimator.GetCluster().GetWeights()[0];

    ASSERT_TRUE(density > 0.5 - 0.1);
    ASSERT_TRUE(density < 0.5 + 0.1);
}

TEST(UniformBellman, SimpleModel) {
    std::mt19937 rd{1234};
    ActionConditionedKernel action_conditioned_kernel{
        SimpleModel::Kernel<1>{&rd}, SimpleModel::Kernel<0>{&rd}, SimpleModel::Kernel<-1>{&rd}};

    EnvParams env_params{action_conditioned_kernel, SimpleModel::RewardFunc{}, 0.95};

    UniformBellmanOperator bellman_op{env_params, 2048, 1., &rd};
    for (int i = 0; i < 20; ++i) {
        bellman_op.MakeIteration();
    }

    QFuncEstForGreedy qfunc_est{env_params, bellman_op.GetQFunc(),
                                // Correction for importance sampling
                                [](auto) { return 2.; }};
    GreedyPolicy policy{qfunc_est};
    MDPKernel mdp_kernel{action_conditioned_kernel, &policy};

    for (FloatT init : std::array{0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.}) {
        std::cout << "\n///////////////////////////////////////////"
                  << "\n";
        Particle state{ConstantInitializer(init, 1)};
        for (int i = 0; i < 10; ++i) {
            std::cout << state << " " << qfunc_est.ValueAtPoint(state) << "\n";
            mdp_kernel.Evolve(state, &state);
        }
        ASSERT_TRUE(state[0]);
    }
}

TEST(StationaryBellmanOperator, SimpleModel) {
    std::mt19937 rd{1234};
    ActionConditionedKernel action_conditioned_kernel{
        SimpleModel::Kernel<1>{&rd}, SimpleModel::Kernel<0>{&rd}, SimpleModel::Kernel<-1>{&rd}};
    EnvParams env_params{action_conditioned_kernel, SimpleModel::RewardFunc{}, 0.95};

    StationaryBellmanOperatorParams operator_params{
        2048 /*num_samples*/, 100. /*density ratio threshold*/, 1. /*radius*/,
        1e-3 /*invariant density threshold*/, 1 /*burnin iterations*/};

    StationaryBellmanOperator bellman_op{env_params, operator_params, &rd};
    for (int i = 0; i < 20; ++i) {
        bellman_op.MakeIteration();
    }

    PrevSampleReweighingHelper rew_helper{&bellman_op.GetSamplingDistribution(), std::nullopt};
    QFuncEstForGreedy qfunc_est{env_params, bellman_op.GetQFunc(), rew_helper};

    GreedyPolicy policy{qfunc_est};
    MDPKernel mdp_kernel{action_conditioned_kernel, &policy};

    for (FloatT init : std::array{0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.}) {
        std::cout << "\n///////////////////////////////////////////"
                  << "\n";
        Particle state{ConstantInitializer(init, 1)};
        for (int i = 0; i < 10; ++i) {
            std::cout << state << " " << qfunc_est.ValueAtPoint(state) << "\n";
            mdp_kernel.Evolve(state, &state);
        }
        ASSERT_TRUE(state[0]);
    }
}
