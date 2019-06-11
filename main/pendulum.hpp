#include <src/kernel.hpp>

namespace Pendulum {
using RandomEngT = std::mt19937;
const FloatT kActionForce = 1.;
const FloatT kDeltaTime = 0.1;
const FloatT kG = 9.8;
const FloatT kResistance = 0.1;
const FloatT kNoiseStrength = 0.05;

namespace {

template <int action_direction>
class Kernel1D : public AbstractKernel<Kernel1D<action_direction>> {
    friend class AbstractKernel<Kernel1D<action_direction>>;
    Kernel1D(RandomEngT* rd) : rd_{rd} {
    }

private:
    template <class S1, class S2, class RandomDeviceT>
    void EvolveImpl(const Particle<S1>& from, Particle<S2>* to, RandomDeviceT* rd) {
        ChechArgs(from);
        ChechArgs(*to);
        std::uniform_real_distribution<FloatT> noise{-kNoiseStrength, kNoiseStrength};
        (*to)[1] = kDeltaTime *
                   (-kG * sin(from[0]) - kResistance * from[1] + kActionForce * action_direction);
        (*to)[0] = from[0] + kDeltaTime * from[1];
    }

    template <class S1, class S2>
    FloatT GetTransDensityImpl(const Particle<S1>& from, const Particle<S2>& to) {
        ChechArgs(from);
        ChechArgs(to);
    }

    template <class S1>
    void ChechArgs(const Particle<S1>& particle) {
        assert(particle.GetDim() == 2);
    }

    RandomEngT* rd_;
};

}  // namespace

template <int action_direction>
class Kernel : public AbstractKernel<Kernel<action_direction>> {
    friend class AbstractKernel<Kernel<action_direction>>;
    Kernel(size_t num_pendulums, RandomEngT* rd) : num_pendulums_{num_pendulums}, rd_{rd} {
    }

private:
    template <class S1, class S2, class RandomDeviceT = NullType>
    void EvolveImpl(const Particle<S1>& from, Particle<S2>* to, RandomDeviceT* rd = nullptr) {
    }

    template <class S1, class S2>
    FloatT GetTransDensityImpl(const Particle<S1>& from, const Particle<S2>& to) {
    }

    size_t num_pendulums_;
    RandomEngT* rd_;
};

}  // namespace Pendulum
