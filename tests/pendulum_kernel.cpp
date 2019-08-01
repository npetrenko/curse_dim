#include <curse_dim/pendulum.hpp>
#include <gtest/gtest.h>

namespace {
FloatT GetExpectedMass(const IKernel& kernel, std::mt19937* rd, const FloatT kInitRadius,
                       const size_t kNumParticles, TypeErasedParticleRef origin) {
    const size_t kDim = kernel.GetSpaceDim();
    auto distr = std::uniform_real_distribution<FloatT>{-kInitRadius, kInitRadius};
    auto lamb_init = [&](size_t) { return distr(*rd); };
    auto initializer = LambdaInitializer(lamb_init, ParticleDim{kDim}, ClusterInitializationTag{});

    ParticleCluster cluster{kNumParticles, initializer};

    FloatT prob = 0;
    for (const auto& elem : cluster) {
        prob += kernel.GetTransDensity(origin, elem) * pow(2 * kInitRadius, kDim) / cluster.size();
    }

    return prob;
}
}  // namespace

TEST(Pendulum, SummsToOne) {
    std::mt19937 rd{123};
    std::uniform_real_distribution<FloatT> init_distr(-M_PI, M_PI);
    for (size_t num_pends : {1}) {
        for (size_t i = 0; i < 10; ++i) {
            RandomVectorizingInitializer particle_init(ParticleDim{num_pends * 2}, &rd, init_distr);
            Particle<ParticleStorage> origin(particle_init);
            Pendulum::Kernel<0> kernel(NumPendulums(num_pends), &rd);

            std::cout << origin << "\n";
            FloatT emass = GetExpectedMass(kernel, &rd, M_PI, 16 * 1024 * 1024, origin);
            ASSERT_NEAR(emass, 1., 0.05);
        }
    }
}
