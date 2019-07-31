#pragma once

#include <include/kernel.hpp>

namespace Pendulum {
inline constexpr FloatT kActionForce = 1.;
inline constexpr FloatT kDeltaTime = 0.1;
inline constexpr FloatT kG = 9.8;
inline constexpr FloatT kResistance = 0.1;
inline constexpr std::array<FloatT, 2> kNoiseStrength = {0.05, 0.05};

template <int action_direction>
class Kernel1D final : public EnableClone<Kernel1D<action_direction>, InheritFrom<RNGKernel>> {
    using BaseT = EnableClone<Kernel1D<action_direction>, InheritFrom<RNGKernel>>;

public:
    inline explicit Kernel1D(std::mt19937* rd) : BaseT{rd} {
    }

    inline size_t GetSpaceDim() const override {
        return 2;
    }

private:
    void EvolveImpl(TypeErasedParticleRef from, TypeErasedParticlePtr to,
                    std::mt19937* rd) const override;

    FloatT GetTransDensityImpl(TypeErasedParticleRef from, TypeErasedParticleRef to) const override;
};

template <int action_direction>
class Kernel final : public EnableClone<Kernel<action_direction>, InheritFrom<IKernel>> {
    using BaseT = EnableClone<Kernel<action_direction>, InheritFrom<IKernel>>;

public:
    Kernel(size_t num_pendulums, std::mt19937* rd);

    inline size_t GetSpaceDim() const override {
        return pendulums_.size() * 2;
    }

    void Evolve(TypeErasedParticleRef from, TypeErasedParticlePtr to) const override;

    void Evolve(TypeErasedParticleRef from, TypeErasedParticlePtr to,
                std::mt19937* rd) const override;

    FloatT GetTransDensity(TypeErasedParticleRef from, TypeErasedParticleRef to) const override;

private:
    template <class RNGType>
    void EvolveHelper(TypeErasedParticleRef, TypeErasedParticlePtr, RNGType* rd) const;
    void CheckArgs(TypeErasedParticleRef part) const;

    const std::vector<Pendulum::Kernel1D<action_direction>> pendulums_;
};

struct RewardFunc {
    FloatT operator()(TypeErasedParticleRef state, size_t /*action*/) const;
};

}  // namespace Pendulum
