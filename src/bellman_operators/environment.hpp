#pragma once

#include "../kernel.hpp"
#include "../agent_policy.hpp"

struct EnvParams {
    using RewardFuncT = std::function<FloatT(TypeErasedParticleRef)>;
    EnvParams(const ConditionedRNGKernel& kernel, RewardFuncT reward, FloatT gamma)
        : ac_kernel{kernel.Clone()}, reward_function{std::move(reward)}, kGamma{gamma} {
    }

    EnvParams(const EnvParams& other) : reward_function(other.reward_function), kGamma(other.kGamma) {
	ac_kernel = other.ac_kernel->Clone();
    }

    EnvParams(EnvParams&&) = default;

    std::unique_ptr<ConditionedRNGKernel> ac_kernel;
    const RewardFuncT reward_function;
    const FloatT kGamma;
};
