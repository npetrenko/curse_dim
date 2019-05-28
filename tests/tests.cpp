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
class DetermKernel : public AbstractKernel<DetermKernel<step>> {
public:
    template <class S1, class S2>
    inline void Evolve(const Particle<S1>& from, Particle<S2>* to) const {
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

TEST(Basic, DeterministicKernelWorks) {
    ParticleStorage storage{10};
    DummyPolicy policy;
    MDPKernel kernel{ActionConditionedKernel{DetermKernel<1>(), DetermKernel<2>()}, &policy};

    Particle test_particle{ZeroInitializer(8, &storage)};

    kernel.Evolve(test_particle, &test_particle);
    ASSERT_EQ(test_particle, Particle{ConstantInitializer(1., 8)});

    kernel.Evolve(test_particle, &test_particle);
    ASSERT_EQ(test_particle, Particle{ConstantInitializer(3., 8)});
}

class ARKernel : public AbstractKernel<ARKernel> {
public:
    ARKernel(std::mt19937* random_device) : rd_(random_device), tmp_storage_{1024} {
    }

    template <class S1, class S2>
    inline void Evolve(const Particle<S1>& from, Particle<S2>* to) {
        std::normal_distribution<FloatT> normal{0., 1.};

        for (size_t i = 0; i < from.GetDim(); ++i) {
            (*to)[i] = 0.5 * from[i] + normal(*rd_);
        }
    }

    template <class S1, class S2>
    inline FloatT GetTransDensity(const Particle<S1>& from, const Particle<S2>& to) {
        Particle<MemoryView> res{ValueInitializer{from, &tmp_storage_}};
        // Particle<ParticleStorage> res{from};
        res *= -0.5;
        res += to;
        FloatT norm = res.NormSquared();
        FloatT result = exp(-norm / 2) / pow(2 * double(M_PI), from.GetDim() / 2);
        tmp_storage_.Clear();
        return result;
    }

private:
    std::mt19937* rd_;
    ParticleStorage tmp_storage_;
};

TEST(Probability, SummsToOne) {
    static const size_t kDim = 2;
    std::mt19937 random_device{1234};

    ARKernel kernel{&random_device};
    using InitT = RandomVectorizingInitializer<MemoryView, std::uniform_real_distribution<FloatT>,
                                               std::mt19937>;
    ParticleCluster cluster{
        1024 * 1024, InitT{kDim, &random_device, std::uniform_real_distribution<FloatT>{-6, 6}}};
    Particle<ParticleStorage> origin{ZeroInitializer(kDim)};

    FloatT prob = 0;
    for (const auto& elem : cluster) {
        prob += kernel.GetTransDensity(origin, elem) * (pow(12, kDim) / cluster.size());
    }

    ASSERT_TRUE(prob >= 0.99 && prob <= 1.01);
}

namespace SimpleModel {
static const FloatT kActionDelta = 0.07;

template <int direction>
class Kernel : public AbstractKernel<Kernel<direction>> {
public:
    Kernel(std::mt19937* random_device) noexcept : rd_(random_device) {
    }

    template <class S1, class S2>
    void Evolve(const Particle<S1>& from, Particle<S2>* to) const {
        auto [travel_down, travel_up] = MaxAllowedTravelDists(from);
        std::uniform_real_distribution<FloatT> delta{travel_down, travel_up};

        (*to)[0] = delta(*rd_);
    }

    template <class S1, class S2>
    FloatT GetTransDensity(const Particle<S1>& from, const Particle<S2>& to) const {
        auto [travel_down, travel_up] = MaxAllowedTravelDists(from);
        if (to[0] > travel_up || to[0] < travel_down) {
            return 0;
        } else {
            return 1. / (travel_up - travel_down);
        }
    }

    inline size_t GetSpaceDim() const {
        return 1;
    }

private:
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

TEST(DISABLED_UniformBellman, SimpleModel) {
    std::mt19937 rd{1234};
    ActionConditionedKernel action_conditioned_kernel{
        SimpleModel::Kernel<1>{&rd}, SimpleModel::Kernel<0>{&rd}, SimpleModel::Kernel<-1>{&rd}};

    EnvParams env_params{action_conditioned_kernel, SimpleModel::RewardFunc{}, 0.95};

    UniformBellmanOperator bellman_op{env_params, 4096*2, 1., &rd};
    for (int i = 0; i < 100; ++i) {
        bellman_op.MakeIteration();
    }

    QFuncEstForGreedy qfunc_est{env_params, std::move(bellman_op.GetQFunc()),
                                [](auto) { return 1.; }};
    // std::cout << qfunc_est << "\n";
    GreedyPolicy policy{qfunc_est};
    MDPKernel mdp_kernel{action_conditioned_kernel, &policy};

    Particle state{ZeroInitializer(1)};
    for (int i = 0; i < 200; ++i) {
        std::cout << state << " " << qfunc_est.ValueAtPoint(state, 0) << " "
                  << qfunc_est.ValueAtPoint(state, 1) << " " << qfunc_est.ValueAtPoint(state, 2)
                  << "\n";
        mdp_kernel.Evolve(state, &state);
    }

    ASSERT_TRUE(state[0]);
}

TEST(DISABLED_StationaryBellmanOperator, SimpleModel) {
    std::mt19937 rd{1234};
    ActionConditionedKernel action_conditioned_kernel{
        SimpleModel::Kernel<1>{&rd}, SimpleModel::Kernel<0>{&rd}, SimpleModel::Kernel<-1>{&rd}};
    EnvParams env_params{action_conditioned_kernel, SimpleModel::RewardFunc{}, 0.95};

    StationaryBellmanOperatorParams operator_params{
        2048 /*num_samples*/, 100. /*density threshold*/,           1. /*radius*/,
        -1. /*unused-uniform-sampling-ratio*/,      1e-3 /*invariant density threshold*/, 2 /*burnin iterations*/};
    StationaryBellmanOperator bellman_op{env_params, operator_params, &rd};
    for (int i = 0; i < 3; ++i) {
        bellman_op.MakeIteration();
    }

    PrevSampleReweighingHelper rew_helper{bellman_op.GetSamplingDistribution()};
    QFuncEstForGreedy qfunc_est{env_params, bellman_op.GetQFunc(), rew_helper};

    // std::cout << qfunc_est << "\n";
    GreedyPolicy policy{qfunc_est};
    MDPKernel mdp_kernel{action_conditioned_kernel, &policy};

    Particle state{ZeroInitializer(1)};
    for (int i = 0; i < 200; ++i) {
        std::cout << state << " " << qfunc_est.ValueAtPoint(state, 0) << " "
                  << qfunc_est.ValueAtPoint(state, 1) << " " << qfunc_est.ValueAtPoint(state, 2)
                  << "\n";
        mdp_kernel.Evolve(state, &state);
    }

    ASSERT_TRUE(state[0]);
}
