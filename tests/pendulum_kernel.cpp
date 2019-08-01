#include <curse_dim/pendulum.hpp>
#include <gtest/gtest.h>

namespace {
FloatT GetExpectedMass(const IKernel& kernel, std::mt19937* rd, const FloatT kInitRadius) {
    const size_t kDim = kernel.GetSpaceDim();
    using InitT = RandomVectorizingInitializer<MemoryView, std::uniform_real_distribution<FloatT>,
                                               std::mt19937>;
    ParticleCluster cluster{
        1024 * 1024, InitT{ParticleDim{kDim}, rd,
                           std::uniform_real_distribution<FloatT>{-kInitRadius, kInitRadius}}};
    Particle<ParticleStorage> origin{ZeroInitializer(ParticleDim{kDim})};

    FloatT prob = 0;
    for (const auto& elem : cluster) {
        prob += kernel.GetTransDensity(origin, elem) * (pow(12, kDim) / cluster.size());
    }

    return prob;
}
}  // namespace

TEST(Pendulum, SummsToOne) {
    std::mt19937 rd{123};
    Pendulum::Kernel<0> kernel(NumPendulums(10), &rd);

    FloatT emass = GetExpectedMass(kernel, &rd, M_PI);
    ASSERT_NEAR(emass, 1., 0.1);
}
