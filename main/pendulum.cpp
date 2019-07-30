#include <include/kernel.hpp>
#include <cassert>

#include "pendulum.hpp"

namespace Pendulum {

template <int action_direction>
void Kernel1D<action_direction>::EvolveImpl(TypeErasedParticleRef from, TypeErasedParticlePtr to,
                                            std::mt19937* rd) const {
    ChechArgs(from);
    ChechArgs(*to);
    std::uniform_real_distribution<FloatT> pos_noise{-kNoiseStrength[0], kNoiseStrength[0]};
    std::uniform_real_distribution<FloatT> speed_noise{-kNoiseStrength[1], kNoiseStrength[1]};
    std::array new_to = CalcEvolveWithFixedNoise(from, {pos_noise(*rd), speed_noise(*rd)});
    std::copy(new_to.begin(), new_to.end(), to->begin());
}

template <int action_direction>
std::array<FloatT, 2> Kernel1D<action_direction>::CalcEvolveWithFixedNoise(
    TypeErasedParticleRef from, std::array<FloatT, 2> noise) const {
    std::array<FloatT, 2> result;

    result[0] = from[0] + kDeltaTime * (from[1] + noise[0]);
    result[1] = kDeltaTime * (-kG * sin(from[0]) - kResistance * from[1] +
                              kActionForce * action_direction + noise[1]);
    return result;
}

template <int action_direction>
FloatT Kernel1D<action_direction>::GetTransDensityImpl(TypeErasedParticleRef from,
                                                       TypeErasedParticleRef to) const {
    ChechArgs(from);
    ChechArgs(to);
    auto calc_corner = [from = &from, this](FloatT pos_noise, FloatT speed_noise) {
        return CalcEvolveWithFixedNoise(*from, {pos_noise, speed_noise});
    };
    std::array<FloatT, 2> lower_left = calc_corner(-kNoiseStrength[0], -kNoiseStrength[1]);
    std::array<FloatT, 2> upper_right = calc_corner(kNoiseStrength[0], kNoiseStrength[1]);

    if ((to[0] < lower_left[0]) || (to[1] < lower_left[1]) || (to[0] > upper_right[0]) ||
        (to[1] > upper_right[1])) {
        return 0;
    }
    return 1 / (2 * kNoiseStrength[0] * 2 * kNoiseStrength[1]);
}

template <int action_direction>
void Kernel1D<action_direction>::ChechArgs(TypeErasedParticleRef particle) const {
    (void)particle;
    assert(particle.GetDim() == 2);
}

template <int action_direction>
Kernel<action_direction>::Kernel(size_t num_pendulums, std::mt19937* rd)
    : BaseT(), pendulums_(num_pendulums, Kernel1D<action_direction>{rd}) {
}

template <int action_direction>
void Kernel<action_direction>::Evolve(TypeErasedParticleRef from, TypeErasedParticlePtr to,
                                      std::mt19937* rd) const {
    EvolveHelper(from, to, rd);
}

template <int action_direction>
void Kernel<action_direction>::Evolve(TypeErasedParticleRef from, TypeErasedParticlePtr to) const {
    EvolveHelper<NullType>(from, to, nullptr);
}

template <int action_direction>
template <class RNGType>
void Kernel<action_direction>::EvolveHelper(TypeErasedParticleRef from, TypeErasedParticlePtr to,
                                            RNGType* rd) const {

    CheckArgs(from);
    CheckArgs(*to);
    for (size_t pend_ix = 0; pend_ix < pendulums_.size(); ++pend_ix) {
        Particle<ConstMemoryView> from_p{ConstMemoryView{&from[2 * pend_ix], 2}};
        Particle<MemoryView> to_p{MemoryView{&(*to)[2 * pend_ix], 2}};
        if constexpr (std::is_same_v<RNGType, NullType>) {
            (void)rd;
            pendulums_[pend_ix].Evolve(from_p, &to_p);
        } else {
            pendulums_[pend_ix].Evolve(from_p, &to_p, rd);
        }
    }
}

template <int action_direction>
FloatT Kernel<action_direction>::GetTransDensity(TypeErasedParticleRef from,
                                                 TypeErasedParticleRef to) const {
    CheckArgs(from);
    CheckArgs(to);
    FloatT result = 1.;
    for (size_t pend_ix = 0; pend_ix < pendulums_.size(); ++pend_ix) {
        Particle<ConstMemoryView> from_p{ConstMemoryView{&*(from.begin() + 2 * pend_ix), 2}};
        Particle<ConstMemoryView> to_p{ConstMemoryView{&*(to.begin() + 2 * pend_ix), 2}};
        result *= pendulums_[pend_ix].GetTransDensity(from_p, to_p);
    }
    return result;
}

template <int action_direction>
void Kernel<action_direction>::CheckArgs(TypeErasedParticleRef part) const {
    (void)part;
    assert(part.GetDim() == pendulums_.size() * 2);
}

FloatT RewardFunc::operator()(TypeErasedParticleRef state, size_t /*action*/) const {
    FloatT val = 1.;
    for (size_t i = 0; i < state.GetDim(); i += 2) {
        val = std::min(cos(state[i]), val);
    }
    return val;
}

template class Kernel<-1>;
template class Kernel<0>;
template class Kernel<1>;

}  // namespace Pendulum
