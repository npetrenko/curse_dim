#pragma once

#include <src/particle.hpp>
#include <src/util.hpp>
#include <src/type_traits.hpp>

#include <random>
#include <cassert>

// Utility structure for template magic
template <class HasRNGTag>
class RNGHolder;

template <>
class RNGHolder<std::true_type> {
public:
    RNGHolder() = delete;
    RNGHolder(std::mt19937* rd) noexcept : rd_{rd} {
    }
    std::mt19937* rd_;
};

template <>
class RNGHolder<std::false_type> {
public:
    RNGHolder() = default;
    NullType* rd_{nullptr};
};

template <class DerivedT, class HasRNGTag = std::true_type>
class AbstractKernel : public CRTPDerivedCaster<DerivedT> {
    using Caster = CRTPDerivedCaster<DerivedT>;

public:
    // Can only be instantiated if HasRNG == false
    AbstractKernel() = default;

    // Can only be instantiated if HasRNG == true
    AbstractKernel(std::mt19937* random_device) noexcept : rng_holder_(random_device) {
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
    RNGHolder<HasRNGTag> rng_holder_;
};

template <class DerivedT, class HasRNGTag = std::true_type>
class AbstractConditionedKernel : public CRTPDerivedCaster<DerivedT> {
    using Caster = CRTPDerivedCaster<DerivedT>;

public:
    // Can only be instantiated if HasRNG == false
    AbstractConditionedKernel() = default;

    // Can only be instantiated if HasRNG == true
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
    RNGHolder<HasRNGTag> rng_holder_{};
};
