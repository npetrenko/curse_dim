#pragma once

#include <src/particle.hpp>
#include <src/util.hpp>
#include <src/type_traits.hpp>

#include <random>
#include <cassert>

class IKernel {
public:
    virtual void Evolve(TypeErasedParticleRef from, TypeErasedParticlePtr output, std::mt19937* rd = nullptr) = 0;
    virtual FloatT GetTransDensity(TypeErasedParticleRef from, TypeErasedParticleRef to) = 0;
    virtual size_t GetSpaceDim() const = 0;
};

class ICondKernel {
public:
    virtual void EvolveConditionally(TypeErasedParticleRef from, TypeErasedParticlePtr output, size_t condition, std::mt19937* rd = nullptr) = 0;
    virtual FloatT GetTransDensityConditionally(TypeErasedParticleRef from, TypeErasedParticle to, size_t condition) = 0;
    virtual size_t GetSpaceDim() const = 0;
};

class RNGKernel : public IKernel {
public:
    RNGKernel() = default;

    // Can only be instantiated if HasRNG == true
    RNGKernel(std::mt19937* random_device) noexcept : this_rd_(random_device) {
    }

    inline void Evolve(TypeErasedParticleRef from, TypeErasedParticlePtr output,
                       std::mt19937* rd = nullptr) const final override {
	if (rd) {
	    EvolveImpl(from, output, rd);
	} else {
	    EvolveImpl(from, output, this_rd_);
	}
    }

    inline FloatT GetTransDensity(TypeErasedParticleRef from, TypeErasedParticleRef to) const final override {
        assert(from.GetDim() == to.GetDim());
        return GetTransDensityImpl(from, to);
    }

private:
    virtual void EvolveImpl(TypeErasedParticleRef from, TypeErasedParticlePtr to, std::mt19937* rd) = 0;
    virtual FloatT GetTransDensityImpl(TypeErasedParticleRef from, TypeErasedParticleRef to) const = 0;
    std::mt19937* const this_rd_{nullptr};
};

class ConditionedRNGKernel : public ICondKernel {
public:
    ConditionedRNGKernel() = default;

    ConditionedRNGKernel(std::mt19937* random_device) noexcept : rng_holder_{random_device} {
    }

    inline void EvolveConditionally(TypeErasedParticleRef from, TypeErasedParticlePtr output,
                                    size_t condition, std::mt19937* rd = nullptr) const final {
        if (rd) {
	    EvolveConditionallyImpl(from, output, condition, rd);
        } else {
	    EvolveConditionallyImpl(from, output, condition, this_rd_);
        }
    }

    inline FloatT GetTransDensityConditionally(TypeErasedParticleRef  from, TypeErasedParticleRef to,
                                               size_t condition) const final {
        return GetTransDensityConditionallyImpl(from, to, condition);
    }

private:
    virtual void EvolveConditionallyImpl(TypeErasedParticleRef, TypeErasedParticlePtr, size_t) = 0;
    virtual FloatT GetTransDensityConditionally(TypeErasedParticleRef, TypeErasedParticleRef, size_t) = 0;

    std::mt19937* const rng_holder_;
};
