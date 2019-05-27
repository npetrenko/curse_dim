#pragma once

#include <src/kernel.hpp>
#include <src/agent_policy.hpp>

template <class RewardFunctionT, class... T>
struct EnvParams {
    EnvParams(ActionConditionedKernel<T...> kernel, RewardFunctionT reward, FloatT gamma)
        : ac_kernel{std::move(kernel)}, reward_function{std::move(reward)}, kGamma{gamma} {
    }
    const ActionConditionedKernel<T...> ac_kernel;
    const RewardFunctionT reward_function;
    const FloatT kGamma;
};
