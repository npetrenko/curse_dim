#pragma once

#include <src/types.hpp>
#include <src/particle.hpp>
#include <src/exceptions.hpp>
#include <src/agent_policy.hpp>

#include <array>
#include <memory>

template <class BaseT>
class AbstractKernel {
public:
    void Evolve(const Particle& from, Particle* output);
    virtual FloatT GetTransDensity(const Particle& from, const Particle& to) = 0;
    virtual ~AbstractKernel() = default;
};

class AbstractConditionedKernel {
public:
    virtual void EvolveConditionally(const Particle& from, Particle* output, size_t condition) = 0;
    virtual void GetTransDensityConditionally(const Particle& from, const Particle& to,
                                              size_t condition) = 0;
    virtual ~AbstractConditionedKernel() = default;
};

template <size_t num_kernels>
class ActionConditionedKernel final : public AbstractConditionedKernel {
public:
    using KernelArrayT = std::array<std::unique_ptr<AbstractKernel>, num_kernels>;

    ActionConditionedKernel(KernelArrayT&& fixed_action_kernels)
        : fixed_action_kernels_(std::move(fixed_action_kernels)) {
    }

    void EvolveConditionally(const Particle& from, Particle* output,
                             size_t action_number) override {
        auto& kernel = fixed_action_kernels_[action_number];
        kernel.Evolve(from, output);
    }

private:
    KernelArrayT fixed_action_kernels_;
};

template <size_t num_kernels>
class MDPKernel final : public AbstractKernel {
public:
    MDPKernel(ActionConditionedKernel<num_kernels>&& action_conditioned_kernel,
              AbstractAgentPolicy* agent_policy)
        : conditioned_kernel_(std::move(action_conditioned_kernel)), agent_policy_(agent_policy) {
    }

    void Evolve(const Particle& from, Particle* output) override {
        size_t action_num = agent_policy_->React(from);
        conditioned_kernel_.EvolveConditionally(from, output, action_num);
    }

    FloatT GetTransDensity(const Particle& from, const Particle& to) override {
        size_t action_num = agent_policy_->React(from);
        return conditioned_kernel_.GetTransDensityConditionally(from, to, action_num);
    }

private:
    ActionConditionedKernel<num_kernels> conditioned_kernel_;
    AbstractAgentPolicy* agent_policy_{nullptr};
};
