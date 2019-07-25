#pragma once

#include "particle.hpp"
#include "util.hpp"
#include "type_traits.hpp"
#include "types.hpp"
#include "cloneable.hpp"

#include <random>
#include <cassert>

class IKernel {
public:
    virtual void Evolve(TypeErasedParticleRef from, TypeErasedParticlePtr output) const = 0;
    virtual void Evolve(TypeErasedParticleRef from, TypeErasedParticlePtr output,
                        std::mt19937* rd) const = 0;
    virtual FloatT GetTransDensity(TypeErasedParticleRef from, TypeErasedParticleRef to) const = 0;
    virtual size_t GetSpaceDim() const = 0;
    virtual ~IKernel() = default;
};

class ICondKernel {
public:
    virtual void EvolveConditionally(TypeErasedParticleRef from, TypeErasedParticlePtr output,
                                     size_t condition) const = 0;
    virtual void EvolveConditionally(TypeErasedParticleRef from, TypeErasedParticlePtr output,
                                     size_t condition, std::mt19937* rd) const = 0;
    virtual FloatT GetTransDensityConditionally(TypeErasedParticleRef from,
                                                TypeErasedParticleRef to,
                                                size_t condition) const = 0;
    virtual size_t GetSpaceDim() const = 0;
    virtual ~ICondKernel() = default;
};

class RNGKernel : public EnableCloneInterface<RNGKernel, InheritFrom<IKernel>> {
public:
    RNGKernel() = default;

    // Can only be instantiated if HasRNG == true
    RNGKernel(std::mt19937* random_device) noexcept : this_rd_(random_device) {
    }

    inline void Evolve(TypeErasedParticleRef from, TypeErasedParticlePtr output,
                       std::mt19937* rd) const final override {
        EvolveImpl(from, output, rd);
    }

    inline void Evolve(TypeErasedParticleRef from,
                       TypeErasedParticlePtr output) const final override {
        EvolveImpl(from, output, this_rd_);
    }

    inline FloatT GetTransDensity(TypeErasedParticleRef from,
                                  TypeErasedParticleRef to) const final override {
        assert(from.GetDim() == to.GetDim());
        return GetTransDensityImpl(from, to);
    }

private:
    virtual void EvolveImpl(TypeErasedParticleRef, TypeErasedParticlePtr, std::mt19937*) const = 0;
    virtual FloatT GetTransDensityImpl(TypeErasedParticleRef, TypeErasedParticleRef) const = 0;
    std::mt19937* const this_rd_{nullptr};
};

class ConditionedRNGKernel
    : public EnableCloneInterface<ConditionedRNGKernel, InheritFrom<ICondKernel>> {
public:
    ConditionedRNGKernel() = default;

    ConditionedRNGKernel(std::mt19937* random_device) noexcept : this_rd_{random_device} {
    }

    inline void EvolveConditionally(TypeErasedParticleRef from, TypeErasedParticlePtr output,
                                    size_t condition, std::mt19937* rd) const final override {
        EvolveConditionallyImpl(from, output, condition, rd);
    }

    inline void EvolveConditionally(TypeErasedParticleRef from, TypeErasedParticlePtr output,
                                    size_t condition) const final override {
        EvolveConditionallyImpl(from, output, condition, this_rd_);
    }

    inline FloatT GetTransDensityConditionally(TypeErasedParticleRef from, TypeErasedParticleRef to,
                                               size_t condition) const final override {
        return GetTransDensityConditionallyImpl(from, to, condition);
    }

private:
    virtual void EvolveConditionallyImpl(TypeErasedParticleRef, TypeErasedParticlePtr, size_t,
                                         std::mt19937*) const = 0;

    virtual FloatT GetTransDensityConditionallyImpl(TypeErasedParticleRef, TypeErasedParticleRef,
                                                    size_t) const = 0;
    std::mt19937* const this_rd_{nullptr};
};
