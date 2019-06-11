#include <src/particle.hpp>
#include <src/util.hpp>
#include <src/type_traits.hpp>

#include <random>
#include <cassert>

// Utility structure for template magic
template <bool has_rng>
struct RNGHolder;

template <>
struct RNGHolder<true> {
    RNGHolder() noexcept {
        // One should not default construct AbstractKernel without RNG
        assert(false);
    }
    RNGHolder(std::mt19937* rd) noexcept : rd_{rd} {
    }
    std::mt19937* rd_;
};

template <>
struct RNGHolder<false> {
    RNGHolder() = default;
    RNGHolder(std::mt19937*) noexcept {
        // One should not initialize no-rng with rng
        assert(false);
    }

    NullType* rd_{nullptr};
};

template <class DerivedT, bool HasRNG = true>
class AbstractKernel : public CRTPDerivedCaster<DerivedT> {
    using Caster = CRTPDerivedCaster<DerivedT>;

public:
    AbstractKernel() = default;
    AbstractKernel(std::mt19937* random_device) noexcept : rng_holder_{random_device} {
    }

    template <class S1, class S2, class RandomDeviceT = NullType>
    inline void Evolve(const Particle<S1>& from, Particle<S2>* output,
                       RandomDeviceT* rd = nullptr) const {
        assert(from.GetDim() == output->GetDim());
        if constexpr (std::is_same_v<NullType, RandomDeviceT>) {
            (void)rd;
            Caster::GetDerived()->EvolveImpl(from, output, rng_holder_.rd_);
        } else {
            Caster::GetDerived()->EvolveImpl(from, output, rd);
        }
    }

    template <class S1, class S2>
    inline FloatT GetTransDensity(const Particle<S1>& from, const Particle<S2>& to) const {
        assert(from.GetDim() == to.GetDim());
        return Caster::GetDerived()->GetTransDensityImpl(from, to);
    }

    inline size_t GetSpaceDim() const {
        return Caster::GetDerived()->GetSpaceDim();
    }

private:
    RNGHolder<HasRNG> rng_holder_;
};

template <class DerivedT, bool HasRNG = true>
class AbstractConditionedKernel : public CRTPDerivedCaster<DerivedT> {
    using Caster = CRTPDerivedCaster<DerivedT>;

public:
    AbstractConditionedKernel() = default;
    AbstractConditionedKernel(std::mt19937* random_device) : rng_holder_{random_device} {
    }

    template <class S1, class S2, class RandomDeviceT = NullType>
    inline void EvolveConditionally(const Particle<S1>& from, Particle<S2>* output,
                                    size_t condition, RandomDeviceT* rd = nullptr) const {
        if constexpr (std::is_same_v<NullType, RandomDeviceT>) {
            (void)rd;
            Caster::GetDerived()->EvolveConditionallyImpl(from, output, condition, rng_holder_.rd_);
        } else {
            Caster::GetDerived()->EvolveConditionallyImpl(from, output, condition, rd);
        }
    }

    template <class S1, class S2>
    inline FloatT GetTransDensityConditionally(const Particle<S1>& from, const Particle<S2>& to,
                                               size_t condition) const {
        return Caster::GetDerived()->GetTransDensityConditionallyImpl(from, to, condition);
    }

    inline size_t GetSpaceDim() const {
        return Caster::GetDerived()->GetSpaceDim();
    }

private:
    RNGHolder<HasRNG> rng_holder_{};
};
