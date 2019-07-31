#include <gtest/gtest.h>
#include <include/bellman_operators/uniform_operator.hpp>

namespace {
class Kernel final : public EnableClone<Kernel, InheritFrom<RNGKernel>> {
public:
    size_t GetSpaceDim() const override {
        return 1;
    }

private:
    void EvolveImpl(TypeErasedParticleRef from, TypeErasedParticlePtr to, std::mt19937*) const override {
	*to = from;
    }

    FloatT GetTransDensityImpl(TypeErasedParticleRef, TypeErasedParticleRef) const override {
        return 1.;
    }
};
}

TEST(UniformOperator, Builds) {
    std::mt19937 rd{123};
    UniformBellmanOperatorPtr bellman_op;
    {
        ActionConditionedKernel ac_kernel(Kernel{});
        EnvParams env_params(
            std::move(ac_kernel), [](auto, auto) { return 1.; }, 0.95);

        UniformBellmanOperator::Builder builder;
        builder.SetInitRadius(1.)
            .SetEnvParams(std::move(env_params))
            .SetNumParticles(128)
            .SetRandomDevice(&rd);
        bellman_op = std::move(builder).Build();
    }

    bellman_op->MakeIteration();
}
