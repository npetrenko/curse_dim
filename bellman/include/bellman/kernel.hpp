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

class IActionConditionedKernel
    : public EnableCloneInterface<IActionConditionedKernel, InheritFrom<ICondKernel>> {
    using BaseT = EnableCloneInterface<IActionConditionedKernel, InheritFrom<ICondKernel>>;

public:
    using BaseT::BaseT;
    virtual size_t GetNumActions() const = 0;
};

template <class... Kernels>
class ActionConditionedKernel final : public EnableClone<ActionConditionedKernel<Kernels...>,
                                                         InheritFrom<IActionConditionedKernel>> {
public:
    ActionConditionedKernel(const ActionConditionedKernel&) = default;
    ActionConditionedKernel(ActionConditionedKernel&&) = default;

    ActionConditionedKernel(Kernels&&... args)
        : fixed_action_kernels_{std::forward<Kernels>(args)...} {
    }

    inline size_t GetSpaceDim() const override {
        size_t result = std::get<0>(fixed_action_kernels_).GetSpaceDim();
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

        template <class KerT>
        inline void operator()(const KerT& kernel) {
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

        template <class KerT>
        inline void operator()(const KerT& kernel) {
            result = kernel.GetTransDensity(from, to);
        }
    };

    std::tuple<Kernels...> fixed_action_kernels_;
};

class IHintableKernel : public EnableCloneInterface<IHintableKernel, InheritFrom<IKernel>> {
public:
    using HintT = size_t;

    virtual HintT CalculateHint(TypeErasedParticleRef from) const = 0;
    virtual FloatT GetTransDensityWithHint(TypeErasedParticleRef from, TypeErasedParticleRef to,
                                           HintT* hint) const = 0;
    virtual void EvolveWithHint(TypeErasedParticleRef from, TypeErasedParticlePtr to, std::mt19937* rd, HintT* hint) const = 0;
    virtual void EvolveWithHint(TypeErasedParticleRef from, TypeErasedParticlePtr to, HintT* hint) const = 0;
};

class IMDPKernel : public EnableCloneInterface<IMDPKernel, InheritFrom<IHintableKernel>> {
public:
    virtual void ResetPolicy(IAgentPolicy* policy) = 0;
};

template <class ACKernel, bool is_abstract = std::is_abstract_v<ACKernel>>
struct _ImplKernelHolder;

template <class ACKernel>
struct _ImplKernelHolder<ACKernel, false> {
    static constexpr bool holds_by_value = true;

    const ACKernel& Get() const {
        return kernel_;
    }

    ACKernel kernel_;
};

template <class ACKernel>
struct _ImplKernelHolder<ACKernel, true> {
    static constexpr bool holds_by_value = false;

    _ImplKernelHolder(const ACKernel& ker) : kernel_(ker.Clone()) {
    }

    _ImplKernelHolder(const _ImplKernelHolder& other) : kernel_(other.kernel_->Clone()) {
    }

    _ImplKernelHolder(_ImplKernelHolder&& other) = default;

    _ImplKernelHolder& operator=(const _ImplKernelHolder& other) {
        kernel_ = other.kernel_.Clone();
    }

    _ImplKernelHolder& operator=(_ImplKernelHolder&& other) = default;

    const ACKernel& Get() const {
        return *kernel_;
    }

    std::unique_ptr<ACKernel> kernel_;
};

template <class ACKernel>
using KernelHolder = _ImplKernelHolder<ACKernel>;

template <class ACKernel>
class MDPKernel final : public EnableClone<MDPKernel<ACKernel>, InheritFrom<IMDPKernel>> {
private:
    using KerHolder = KernelHolder<ACKernel>;

public:
    static constexpr bool holds_kernel_by_value = KerHolder::holds_by_value;

    using HintT = IMDPKernel::HintT;

    MDPKernel(const ACKernel& action_conditioned_kernel, IAgentPolicy* agent_policy)
        : conditioned_kernel_holder_{action_conditioned_kernel}, agent_policy_{agent_policy} {
    }

    MDPKernel(const MDPKernel& other)
        : conditioned_kernel_holder_(other.conditioned_kernel_holder_) {
        agent_policy_ = other.agent_policy_;
    }

    MDPKernel(MDPKernel&&) = default;

    MDPKernel& operator=(MDPKernel&&) = default;

    MDPKernel& operator=(const MDPKernel& other) {
        conditioned_kernel_holder_ = other.conditioned_kernel_holder_;
        agent_policy_ = other.agent_policy_;
        return *this;
    }

    inline void ResetPolicy(IAgentPolicy* agent_policy) override {
        agent_policy_ = agent_policy;
    }

    inline size_t GetSpaceDim() const override {
        return conditioned_kernel_holder_.Get().GetSpaceDim();
    }

    inline HintT CalculateHint(TypeErasedParticleRef from) const override {
        return agent_policy_->React(from);
    }

    inline FloatT GetTransDensityWithHint(TypeErasedParticleRef from, TypeErasedParticleRef to,
                                          HintT* hint) const override {
        return conditioned_kernel_holder_.Get().GetTransDensityConditionally(from, to, *hint);
    }

    inline void EvolveWithHint(TypeErasedParticleRef from, TypeErasedParticlePtr to, std::mt19937* rd, HintT* hint) const override {
	conditioned_kernel_holder_.Get().EvolveConditionally(from, to, *hint, rd);
    }

    inline void EvolveWithHint(TypeErasedParticleRef from, TypeErasedParticlePtr to, HintT* hint) const override {
	conditioned_kernel_holder_.Get().EvolveConditionally(from, to, *hint);
    }

    inline void Evolve(TypeErasedParticleRef from, TypeErasedParticlePtr output,
                       std::mt19937* rd) const override {
        size_t action_num = agent_policy_->React(from);
        conditioned_kernel_holder_.Get().EvolveConditionally(from, output, action_num, rd);
    }

    inline void Evolve(TypeErasedParticleRef from, TypeErasedParticlePtr output) const override {
        size_t action_num = agent_policy_->React(from);
        conditioned_kernel_holder_.Get().EvolveConditionally(from, output, action_num);
    }

    inline FloatT GetTransDensity(TypeErasedParticleRef from,
                                  TypeErasedParticleRef to) const override {
        size_t action_num = agent_policy_->React(from);
        return conditioned_kernel_holder_.Get().GetTransDensityConditionally(from, to, action_num);
    }

private:
    KerHolder conditioned_kernel_holder_;
    IAgentPolicy* agent_policy_{nullptr};
};
