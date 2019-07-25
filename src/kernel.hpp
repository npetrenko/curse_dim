#pragma once

#include <src/types.hpp>
#include <src/particle.hpp>
#include <src/exceptions.hpp>
#include <src/agent_policy.hpp>
#include <src/util.hpp>
#include <src/type_traits.hpp>

#include <src/abstract_kernel.hpp>

#include <array>
#include <memory>
#include <exception>
#include <functional>

template <class... T>
class ActionConditionedKernel final
    : public ConditionedRNGKernel {
public:
    ActionConditionedKernel(T&&... args) : fixed_action_kernels_{std::forward<T>(args)...} {
    }

    inline size_t GetSpaceDim() const override {
        size_t result = std::get<0>(fixed_action_kernels_).GetSpaceDim();
// Need to add checks
#ifndef NDEBUG
#endif
        return result;
    }

    static inline size_t GetDim() {
        return sizeof...(T);
    }

private:
    void EvolveConditionallyImpl(TypeErasedParticleRef from, TypeErasedParticlePtr output, size_t action_number, std::mt19937* rd) const override {
        EvolveHelper helper{from, output, rd};
        CallOnTupleIx(std::move(helper), fixed_action_kernels_, action_number);
    }

    FloatT GetTransDensityConditionallyImpl(TypeErasedParticleRef from, TypeErasedParticlePtr to,
                                            size_t action_number) const override {
        FloatT result;
        DensityHelper helper{from, to, result};
        CallOnTupleIx(std::move(helper), fixed_action_kernels_, action_number);
        return result;
    }

    struct EvolveHelper {
        TypeErasedParticleRef from;
        TypeErasedParticlePtr to;
	std::mt19937* rd;

        template <class Ker>
        inline void operator()(const Ker& kernel) {
            if constexpr (std::is_same_v<NullType, RandomDeviceT>) {
                kernel.Evolve(from, to);
            } else {
                kernel.Evolve(from, to, rd);
            }
        }
    };

    struct DensityHelper {
        TypeErasedParticleRef from;
        TypeErasedParticleRef to;
        FloatT& result;

        template <class Ker>
        inline void operator()(const Ker& kernel) {
            result = kernel.GetTransDensity(from, to);
        }
    };

    std::tuple<T...> fixed_action_kernels_;
};

class HintableKernel : public AbstractRNGKernel {

public:
    HintableKernel() = default;

    HintableKernel(std::mt19937* rd) : AbstractRNGKernel{rd} {
    }

    using HintT = size_t;

    virtual HintT CalculateHint(TypeErasedParticleRef from) const = 0;

    virtual FloatT GetTransDensityWithHint(TypeErasedParticleRef from, TypeErasedParticleRef to, HintT* hint) const = 0;
};

template <class DerivedPolicy, class T>
class MDPKernel final : public HintableKernel {
public:
    MDPKernel(const ConditionedRNGKernel& action_conditioned_kernel,
              AbstractAgentPolicy<DerivedPolicy>* agent_policy)
        : conditioned_kernel_{type_traits::GetDeepestLevelCopy(action_conditioned_kernel)},
          agent_policy_{agent_policy} {
    }

    inline void ResetPolicy(AbstractAgentPolicy<DerivedPolicy>* agent_policy) {
        agent_policy_ = agent_policy;
    }

    inline size_t GetSpaceDim() const {
        return conditioned_kernel_.GetSpaceDim();
    }

private:
    template <class S1, class S2, class RandomDeviceT>
    void EvolveImpl(const Particle<S1>& from, Particle<S2>* output, RandomDeviceT* rd) const {
        size_t action_num = agent_policy_->React(from);
        conditioned_kernel_.EvolveConditionally(from, output, action_num, rd);
    }

    template <class S1, class S2>
    FloatT GetTransDensityImpl(const Particle<S1>& from, const Particle<S2>& to) const {
        size_t action_num = agent_policy_->React(from);
        return conditioned_kernel_.GetTransDensityConditionally(from, to, action_num);
    }

    template <class S>
    HintT CalculateHintImpl(const Particle<S>& from) const {
        return agent_policy_->React(from);
    }

    template <class S1, class S2>
    FloatT GetTransDensityWithHintImpl(const Particle<S1>& from, const Particle<S2>& to,
                                       HintT* hint) const {
        return conditioned_kernel_.GetTransDensityConditionally(from, to, *hint);
    }

    type_traits::DeepestCRTPType<AbstractConditionedKernel<T, std::false_type>> conditioned_kernel_;
    AbstractAgentPolicy<DerivedPolicy>* agent_policy_{nullptr};
};
