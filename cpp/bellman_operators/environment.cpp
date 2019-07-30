#include <include/bellman_operators/environment.hpp>

EnvParams::EnvParams(const EnvParams& other)
    : reward_function(other.reward_function), gamma(other.gamma) {
    ac_kernel = other.ac_kernel->Clone();
    mdp_kernel = other.mdp_kernel->Clone();
}

EnvParams& EnvParams::operator=(const EnvParams& other) {
    ac_kernel = other.ac_kernel->Clone();
    mdp_kernel = other.mdp_kernel->Clone();
    reward_function = other.reward_function;
    gamma = other.gamma;
    return *this;
}
