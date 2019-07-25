#pragma once

#include "types.hpp"
#include "particle.hpp"
#include "exceptions.hpp"
#include "agent_policy.hpp"
#include "util.hpp"
#include "type_traits.hpp"

#include "abstract_kernel.hpp"

#include <array>
#include <memory>
#include <exception>
#include <functional>

template <class... Kernels>
class ActionConditionedKernel final
    : public EnableClone<ActionConditionedKernel<Kernels...>, InheritFrom<ConditionedRNGKernel>> {
public:
    ActionConditionedKernel(Kernels&&... args)
        : fixed_action_kernels_{std::forward<Kernels>(args)...} {
    }

    inline size_t GetSpaceDim() const override {
        size_t result = std::get<0>(fixed_action_kernels_).GetSpaceDim();
// Need to add checks
#ifndef NDEBUG
#endif
        return result;
    }

    static inline size_t GetDim() {
        return sizeof...(Kernels);
    }

private:
    void EvolveConditionallyImpl(TypeErasedParticleRef from, TypeErasedParticlePtr output,
                                 size_t action_number, std::mt19937* rd) const override {
        EvolveHelper helper{from, output, rd};
        CallOnTupleIx(std::move(helper), fixed_action_kernels_, action_number);
    }

    FloatT GetTransDensityConditionallyImpl(TypeErasedParticleRef from, TypeErasedParticleRef to,
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

        inline void operator()(const RNGKernel& kernel) {
            kernel.Evolve(from, to, rd);
        }
    };

    struct DensityHelper {
        TypeErasedParticleRef from;
        TypeErasedParticleRef to;
        FloatT& result;

        inline void operator()(const RNGKernel& kernel) {
            result = kernel.GetTransDensity(from, to);
        }
    };

    std::tuple<Kernels...> fixed_action_kernels_;
};

class HintableKernel : public EnableCloneInterface<HintableKernel, InheritFrom<RNGKernel>> {
    using BaseT = EnableCloneInterface<HintableKernel, InheritFrom<RNGKernel>>;

public:
    HintableKernel() = default;

    HintableKernel(std::mt19937* rd) : BaseT(rd) {
    }

    using HintT = size_t;

    virtual HintT CalculateHint(TypeErasedParticleRef from) const = 0;
    virtual FloatT GetTransDensityWithHint(TypeErasedParticleRef from, TypeErasedParticleRef to,
                                           HintT* hint) const = 0;
};

class MDPKernel final : public EnableClone<MDPKernel, InheritFrom<HintableKernel>> {
public:
    template <class... Kernels>
    MDPKernel(const ActionConditionedKernel<Kernels...>& action_conditioned_kernel,
              AbstractAgentPolicy* agent_policy)
        : conditioned_kernel_{action_conditioned_kernel.Clone()}, agent_policy_{agent_policy} {
    }

    inline void ResetPolicy(AbstractAgentPolicy* agent_policy) {
        agent_policy_ = agent_policy;
    }

    inline size_t GetSpaceDim() const override {
        return conditioned_kernel_->GetSpaceDim();
    }

    HintT CalculateHint(TypeErasedParticleRef from) const override {
        return agent_policy_->React(from);
    }

    FloatT GetTransDensityWithHint(TypeErasedParticleRef from, TypeErasedParticleRef to,
                                   HintT* hint) const override {
        return conditioned_kernel_->GetTransDensityConditionally(from, to, *hint);
    }

private:
    void EvolveImpl(TypeErasedParticleRef from, TypeErasedParticlePtr output,
                    std::mt19937* rd) const override {
        size_t action_num = agent_policy_->React(from);
        conditioned_kernel_->EvolveConditionally(from, output, action_num, rd);
    }

    FloatT GetTransDensityImpl(TypeErasedParticleRef from,
                               TypeErasedParticleRef to) const override {
        size_t action_num = agent_policy_->React(from);
        return conditioned_kernel_->GetTransDensityConditionally(from, to, action_num);
    }

    std::unique_ptr<ICondKernel> conditioned_kernel_;
    AbstractAgentPolicy* agent_policy_{nullptr};
};
