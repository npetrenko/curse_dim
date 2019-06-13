#pragma once

#include <src/kernel.hpp>

namespace Pendulum {
const FloatT kActionForce = 1.;
const FloatT kDeltaTime = 0.1;
const FloatT kG = 9.8;
const FloatT kResistance = 0.1;
const std::array<FloatT, 2> kNoiseStrength = {0.05, 0.05};

namespace {

template <int action_direction>
class Kernel1D : public AbstractKernel<Kernel1D<action_direction>> {
    friend class AbstractKernel<Kernel1D<action_direction>>;
    using BaseT = AbstractKernel<Kernel1D<action_direction>>;

public:
    Kernel1D(std::mt19937* rd) : BaseT{rd} {
    }

    size_t GetSpaceDim() const {
        return 2;
    }

private:
    template <class S1, class S2, class RandomDeviceT>
    void EvolveImpl(const Particle<S1>& from, Particle<S2>* to, RandomDeviceT* rd) const {
        ChechArgs(from);
        ChechArgs(*to);
        std::uniform_real_distribution<FloatT> pos_noise{-kNoiseStrength[0], kNoiseStrength[0]};
        std::uniform_real_distribution<FloatT> speed_noise{-kNoiseStrength[1], kNoiseStrength[1]};
        std::array<FloatT&, 2>{(*to)[0], (*to)[1]} =
            CalcEvolveWithFixedNoise(from, {pos_noise(*rd), speed_noise(*rd)});
    }

    template <class S>
    std::array<FloatT, 2> CalcEvolveWithFixedNoise(const Particle<S>& from,
                                                   std::array<FloatT, 2> noise) const {
        std::array<FloatT, 2> result;

        result[0] = from[0] + kDeltaTime * (from[1] + noise[0]);
        result[1] = kDeltaTime * (-kG * sin(from[0]) - kResistance * from[1] +
                                  kActionForce * action_direction + noise[1]);
        return result;
    }

    template <class S1, class S2>
    FloatT GetTransDensityImpl(const Particle<S1>& from, const Particle<S2>& to) const {
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

    template <class S1>
    void ChechArgs(const Particle<S1>& particle) const {
        (void)particle;
        assert(particle.GetDim() == 2);
    }
};

}  // namespace

template <int action_direction>
class Kernel : public AbstractKernel<Kernel<action_direction>, std::false_type> {
    friend class AbstractKernel<Kernel<action_direction>, std::false_type>;
    using BaseT = AbstractKernel<Kernel<action_direction>, std::false_type>;

public:
    Kernel(size_t num_pendulums, std::mt19937* rd) : BaseT{}, pendulums_(num_pendulums, {rd}) {
    }

    size_t GetSpaceDim() const {
        return pendulums_.size() * 2;
    }

private:
    template <class S1, class S2, class RandomDeviceT>
    void EvolveImpl(const Particle<S1>& from, Particle<S2>* to, RandomDeviceT* rd) const {
        CheckArgs(from);
        CheckArgs(*to);
        for (size_t pend_ix = 0; pend_ix < pendulums_.size(); ++pend_ix) {
            Particle<ConstMemoryView> from_p{ConstMemoryView{&from[2 * pend_ix], 2}};
            Particle<MemoryView> to_p{MemoryView{&(*to)[2 * pend_ix], 2}};
            pendulums_[pend_ix].Evolve(from_p, &to_p, rd);
        }
    }

    template <class S1, class S2>
    FloatT GetTransDensityImpl(const Particle<S1>& from, const Particle<S2>& to) const {
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

    template <class S>
    void CheckArgs(const Particle<S>& part) const {
        (void)part;
        assert(part.GetDim() == pendulums_.size() * 2);
    }

    const std::vector<Pendulum::Kernel1D<action_direction>> pendulums_;
};

struct RewardFunc {
    template <class S>
    FloatT operator()(const Particle<S>& state, size_t /*action*/) const {
        FloatT val = 1.;
        for (size_t i = 0; i < state.GetDim(); i += 2) {
            val = std::min(state[i], val);
        }
        return val;
    }
};

}  // namespace Pendulum
