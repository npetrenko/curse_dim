#include <src/initializer.hpp>
#include <src/particle.hpp>
#include <src/kernel.hpp>
#include <src/particle_storage.hpp>
#include <src/bellman.hpp>
#include <src/bellman_operators/uniform_operator.hpp>
#include <src/bellman_operators/qfunc.hpp>

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
	    (*to)[i] = 0.5*from[i] + normal(*rd_);
	}
    }

    template <class S1, class S2>
    inline FloatT GetTransDensity(const Particle<S1>& from, const Particle<S2>& to) {
        Particle<MemoryView> res{ValueInitializer{from, &tmp_storage_}};
        // Particle<ParticleStorage> res{from};
	res *= -0.5;
        res += to;
	FloatT norm = res.NormSquared();
	FloatT result = exp(-norm/2)/pow(2*double(M_PI), from.GetDim()/2);
	tmp_storage_.Clear();
	return result;
    }

private:
    std::mt19937* rd_;
    ParticleStorage tmp_storage_;
};

TEST(Probability, SummsToOne) {
    static const size_t kDim = 2;
    std::mt19937 random_device{std::random_device{}()};

    ARKernel kernel{&random_device};
    using InitT = RandomVectorizingInitializer<MemoryView, std::uniform_real_distribution<FloatT>,
                                               std::mt19937>;
    ParticleCluster cluster{
        1024 * 1024, InitT{kDim, &random_device, std::uniform_real_distribution<FloatT>{-6, 6}}};
    Particle<ParticleStorage> origin{ZeroInitializer(kDim)};

    FloatT prob = 0;
    for (const auto& elem : cluster) {
        prob += kernel.GetTransDensity(origin, elem) * (pow(12, kDim)/ cluster.size());
    }

    ASSERT_TRUE(prob >= 0.99 && prob <= 1.01);
}

namespace SimpleModel {
static const FloatT kActionDelta = 0.3;

template <int direction>
class Kernel : public AbstractKernel<Kernel<direction>> {
public:
    Kernel(std::mt19937* random_device) noexcept : rd_(random_device) {
    }

    template <class S1, class S2>
    void Evolve(const Particle<S1>& from, Particle<S2>* to) const {
        auto [travel_dist_down, travel_dist_up] = MaxAllowedTravelDists(from);
        std::uniform_real_distribution<FloatT> delta{travel_dist_down, travel_dist_up};

        (*to)[0] = from[0] + delta(*rd_);
    }

    template <class S1, class S2>
    FloatT GetTransDensity(const Particle<S1>& from, const Particle<S2>& to) const {
        auto [travel_dist_down, travel_dist_up] = MaxAllowedTravelDists(from);
        FloatT dist = to[0] - from[0];
        if (dist > travel_dist_up || dist < travel_dist_down) {
            return 0;
        } else {
            return 1 / (travel_dist_up - travel_dist_down);
        }
    }

    inline size_t GetSpaceDim() const {
	return 1;
    }

private:
    template <class S1>
    std::pair<FloatT, FloatT> MaxAllowedTravelDists(const Particle<S1>& from) const {
        FloatT dist_bound = 0.05;
        FloatT travel_dist_up = std::min(1 - from[0] - kActionDelta * direction, dist_bound);
        FloatT travel_dist_down = std::min(-(-1 - from[0] + kActionDelta * direction), dist_bound);
        return {-travel_dist_down, travel_dist_up};
    }
    std::mt19937* rd_;
};

struct RewardFunc {
    template <class S>
    FloatT operator()(const Particle<S>& state, size_t/*action*/) const {
        return state[0];
    }
};
}  // namespace SimpleModel

TEST(UniformBellman, SimpleModel) {
    std::mt19937 rd{1234};
    ActionConditionedKernel action_conditioned_kernel{
        SimpleModel::Kernel<1>{&rd}, SimpleModel::Kernel<0>{&rd}, SimpleModel::Kernel<-1>{&rd}};

    UniformBellmanOperator bellman_op{action_conditioned_kernel, SimpleModel::RewardFunc{}, 1024,
                                      1., &rd};
    for (int i = 0; i < 1; ++i) {
        bellman_op.MakeIteration();
    }

    std::cout << bellman_op.GetQFunc() << "\n";
    QFuncEstForGreedy qfunc_est(action_conditioned_kernel, std::move(bellman_op.GetQFunc()),
                                [](auto) { return 1.; });
    GreedyPolicy policy{qfunc_est};
    MDPKernel mdp_kernel{action_conditioned_kernel, &policy};

    Particle state{ZeroInitializer(1)};
    for (int i = 0; i < 100;  ++i) {
        std::cout << state << "\n";
	mdp_kernel.Evolve(state, &state);
    }
}
