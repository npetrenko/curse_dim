#pragma once

#include <src/types.hpp>
#include <src/particle.hpp>
#include <src/exceptions.hpp>
#include <src/agent_policy.hpp>
#include <src/util.hpp>

#include <array>
#include <memory>
#include <exception>

struct NullType {};

template <class DerivedT>
class AbstractKernel : public CRTPDerivedCaster<DerivedT> {
public:
    template <class S1, class S2, class RandomDeviceT = NullType>
    inline void Evolve(const Particle<S1>& from, Particle<S2>* output,
                       RandomDeviceT* rd = nullptr) const {
        (void)rd;
        assert(from.GetDim() == output->GetDim());
        if constexpr (std::is_same_v<NullType, RandomDeviceT>) {
            this->GetDerived()->EvolveImpl(from, output);
        } else {
            this->GetDerived()->EvolveImpl(from, output, rd);
        }
    }

    template <class S1, class S2>
    inline FloatT GetTransDensity(const Particle<S1>& from, const Particle<S2>& to) const {
        assert(from.GetDim() == to.GetDim());
        return this->GetDerived()->GetTransDensityImpl(from, to);
    }

    inline size_t GetSpaceDim() const {
        return this->GetDerived()->GetSpaceDim();
    }
};

template <class DerivedT>
class AbstractConditionedKernel : public CRTPDerivedCaster<DerivedT> {
public:
    template <class S1, class S2, class RandomDeviceT = NullType>
    inline void EvolveConditionally(const Particle<S1>& from, Particle<S2>* output,
                                    size_t condition, RandomDeviceT* rd = nullptr) const {
        (void)rd;
        if constexpr (std::is_same_v<NullType, RandomDeviceT>) {
            this->GetDerived()->EvolveConditionallyImpl(from, output, condition);
        } else {
            this->GetDerived()->EvolveConditionallyImpl(from, output, condition, rd);
        }
    }

    template <class S1, class S2>
    inline FloatT GetTransDensityConditionally(const Particle<S1>& from, const Particle<S2>& to,
                                               size_t condition) const {
        return this->GetDerived()->GetTransDensityConditionallyImpl(from, to, condition);
    }

    inline size_t GetSpaceDim() const {
        return this->GetDerived()->GetSpaceDim();
    }
};

template <class... T>
class ActionConditionedKernel final
    : public AbstractConditionedKernel<ActionConditionedKernel<T...>> {
    friend class AbstractConditionedKernel<ActionConditionedKernel<T...>>;

public:
    ActionConditionedKernel(T&&... args) : fixed_action_kernels_(std::move(args)...) {
    }

    ActionConditionedKernel(const T&... args) : fixed_action_kernels_(args...) {
    }

    inline size_t GetSpaceDim() const {
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
    template <class S1, class S2, class RandomDeviceT = NullType>
    void EvolveConditionallyImpl(const Particle<S1>& from, Particle<S2>* output,
                                 size_t action_number, RandomDeviceT* rd = nullptr) const {
        EvolveHelper<S1, S2, RandomDeviceT> helper{from, output, rd};
        CallOnTupleIx(std::move(helper), fixed_action_kernels_, action_number);
    }

    template <class S1, class S2>
    FloatT GetTransDensityConditionallyImpl(const Particle<S1>& from, const Particle<S2>& to,
                                            size_t action_number) const {
        FloatT result;
        DensityHelper<S1, S2> helper{from, to, result};
        CallOnTupleIx(std::move(helper), fixed_action_kernels_, action_number);
        return result;
    }

    template <class S1, class S2, class RandomDeviceT>
    struct EvolveHelper {
        const Particle<S1>& from;
        Particle<S2>* to;
        RandomDeviceT* rd;

        template <class Ker>
        inline void operator()(const Ker& kernel) {
            if constexpr (std::is_same_v<NullType, RandomDeviceT>) {
                kernel.Evolve(from, to);
            } else {
                kernel.Evolve(from, to, rd);
            }
        }
    };

    template <class S1, class S2>
    struct DensityHelper {
        const Particle<S1>& from;
        const Particle<S2>& to;
        FloatT& result;

        template <class Ker>
        inline void operator()(const Ker& kernel) {
            result = kernel.GetTransDensity(from, to);
        }
    };

    std::tuple<T...> fixed_action_kernels_;
};

template <class DerivedT>
class HintableKernel : public AbstractKernel<DerivedT> {
public:
    template <class S>
    auto CalculateHint(const Particle<S>& from) const {
        return this->GetDerived()->CalculateHintImpl(from);
    }

    template <class S1, class S2, class HintT>
    FloatT GetTransDensityWithHint(const Particle<S1>& from, const Particle<S2>& to,
                                   HintT* hint) const {
        return this->GetDerived()->GetTransDensityWithHintImpl(from, to, hint);
    }
};

template <class DerivedPolicy, class... T>
class MDPKernel final : public HintableKernel<MDPKernel<DerivedPolicy, T...>> {
    using ThisT = MDPKernel<DerivedPolicy, T...>;
    friend class HintableKernel<ThisT>;
    friend class AbstractKernel<ThisT>;

public:
    using HintT = size_t;

    MDPKernel(const ActionConditionedKernel<T...>& action_conditioned_kernel,
              AbstractAgentPolicy<DerivedPolicy>* agent_policy)
        : conditioned_kernel_{action_conditioned_kernel}, agent_policy_{agent_policy} {
    }

    inline void ResetPolicy(AbstractAgentPolicy<DerivedPolicy>* agent_policy) {
        agent_policy_ = agent_policy;
    }

    inline size_t GetSpaceDim() const {
        return conditioned_kernel_.GetSpaceDim();
    }

private:
    template <class S1, class S2, class RandomDeviceT = NullType>
    void EvolveImpl(const Particle<S1>& from, Particle<S2>* output,
                    RandomDeviceT* rd = nullptr) const {
        (void)rd;
        size_t action_num = agent_policy_->React(from);
        if constexpr (std::is_same_v<RandomDeviceT, NullType>) {
            conditioned_kernel_.EvolveConditionally(from, output, action_num);
        } else {
            conditioned_kernel_.EvolveConditionally(from, output, action_num, rd);
        }
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

    const ActionConditionedKernel<T...>& conditioned_kernel_;
    AbstractAgentPolicy<DerivedPolicy>* agent_policy_{nullptr};
};
