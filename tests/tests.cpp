#include <src/initializer.hpp>
#include <src/particle.hpp>
#include <src/kernel.hpp>
#include <src/particle_storage.hpp>
#include <src/bellman.hpp>

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
