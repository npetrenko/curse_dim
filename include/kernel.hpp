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

class IActionConditionedKernel : public EnableCloneInterface<IActionConditionedKernel, InheritFromVirtual<ICondKernel>> {
    using BaseT = EnableCloneInterface<IActionConditionedKernel, InheritFromVirtual<ICondKernel>>;
public:
    using BaseT::BaseT;
    virtual size_t GetNumActions() const = 0;
};

template <class... Kernels>
class ActionConditionedKernel final
    : public EnableClone<ActionConditionedKernel<Kernels...>, InheritFromVirtual<IActionConditionedKernel>> {
public:
    ActionConditionedKernel(const ActionConditionedKernel&) = default;
    ActionConditionedKernel(ActionConditionedKernel&&) = default;

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

    inline size_t GetNumActions() const override {
        return sizeof...(Kernels);
    }

    void EvolveConditionally(TypeErasedParticleRef from, TypeErasedParticlePtr output,
                                 size_t action_number) const override {
        EvolveHelper<false> helper{from, output, nullptr};
        CallOnTupleIx(std::move(helper), fixed_action_kernels_, action_number);
    }

    void EvolveConditionally(TypeErasedParticleRef from, TypeErasedParticlePtr output,
                                 size_t action_number, std::mt19937* rd) const override {
        EvolveHelper<true> helper{from, output, rd};
        CallOnTupleIx(std::move(helper), fixed_action_kernels_, action_number);
    }

    FloatT GetTransDensityConditionally(TypeErasedParticleRef from, TypeErasedParticleRef to,
                                        size_t action_number) const override {
        FloatT result;
        DensityHelper helper{from, to, result};
        CallOnTupleIx(std::move(helper), fixed_action_kernels_, action_number);
        return result;
    }

private:
    template <bool needs_rd>
    struct EvolveHelper {
        TypeErasedParticleRef from;
        TypeErasedParticlePtr to;
        std::mt19937* rd;

        inline void operator()(const RNGKernel& kernel) {
            if constexpr (needs_rd) {
                kernel.Evolve(from, to, rd);
            } else {
		kernel.Evolve(from, to);
	    }
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

class IHintableKernel : public EnableCloneInterface<IHintableKernel, InheritFromVirtual<IKernel>> {
public:
    using HintT = size_t;

    virtual HintT CalculateHint(TypeErasedParticleRef from) const = 0;
    virtual FloatT GetTransDensityWithHint(TypeErasedParticleRef from, TypeErasedParticleRef to,
                                           HintT* hint) const = 0;
};

template <class BaseKernel>
class HintableKernel : public EnableCloneInterface<HintableKernel<BaseKernel>, InheritFromVirtual<IHintableKernel, BaseKernel>> {
    using BaseT = EnableCloneInterface<HintableKernel<BaseKernel>, InheritFrom<IHintableKernel, BaseKernel>>;

public:
    HintableKernel() = default;

    HintableKernel(std::mt19937* rd) : BaseKernel(rd) {
    }
};

class MDPKernel final : public EnableClone<MDPKernel, InheritFrom<HintableKernel<IKernel>>> {
public:
    MDPKernel(const IActionConditionedKernel& action_conditioned_kernel,
              IAgentPolicy* agent_policy)
        : conditioned_kernel_{action_conditioned_kernel.Clone()}, agent_policy_{agent_policy} {
    }

    MDPKernel(const MDPKernel& other) {
	conditioned_kernel_ = other.conditioned_kernel_->Clone();
	agent_policy_ = other.agent_policy_;
    }

    MDPKernel(MDPKernel&&) = default;

    MDPKernel& operator=(MDPKernel&&) = default;

    MDPKernel& operator=(const MDPKernel& other) {
	conditioned_kernel_ = other.conditioned_kernel_->Clone();
	agent_policy_ = other.agent_policy_;
	return *this;
    }

    inline void ResetPolicy(IAgentPolicy* agent_policy) {
        agent_policy_ = agent_policy;
    }

    inline size_t GetSpaceDim() const override {
        return conditioned_kernel_->GetSpaceDim();
    }

    inline HintT CalculateHint(TypeErasedParticleRef from) const override {
        return agent_policy_->React(from);
    }

    inline FloatT GetTransDensityWithHint(TypeErasedParticleRef from, TypeErasedParticleRef to,
                                   HintT* hint) const override {
        return conditioned_kernel_->GetTransDensityConditionally(from, to, *hint);
    }

    inline void Evolve(TypeErasedParticleRef from, TypeErasedParticlePtr output,
                    std::mt19937* rd) const override {
        size_t action_num = agent_policy_->React(from);
        conditioned_kernel_->EvolveConditionally(from, output, action_num, rd);
    }

    inline void Evolve(TypeErasedParticleRef from, TypeErasedParticlePtr output) const override {
        size_t action_num = agent_policy_->React(from);
        conditioned_kernel_->EvolveConditionally(from, output, action_num);
    }

    inline FloatT GetTransDensity(TypeErasedParticleRef from,
                                  TypeErasedParticleRef to) const override {
        size_t action_num = agent_policy_->React(from);
        return conditioned_kernel_->GetTransDensityConditionally(from, to, action_num);
    }

private:

    std::unique_ptr<const IActionConditionedKernel> conditioned_kernel_;
    IAgentPolicy* agent_policy_{nullptr};
};
