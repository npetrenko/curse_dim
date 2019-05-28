#pragma once

#include <src/bellman.hpp>
#include <src/kernel.hpp>
#include <src/particle.hpp>
#include <src/bellman_operators/qfunc.hpp>
#include <src/bellman_operators/environment.hpp>
#include <src/density_estimators/stationary_estimator.hpp>

#include <random>
#include <cassert>
#include <optional>

#ifndef NDEBUG
#include <fenv.h>
#endif

struct StationaryBellmanOperatorParams {
    size_t num_particles;
    FloatT density_ratio_threshold, init_radius;
    FloatT uniform_sampling_ratio;
    size_t num_burnin_iterations = 10;
};

template <class RandomDeviceT, class RewardFuncT, class EstimatorKernelT, class... T>
class StationaryBellmanOperator {
public:
    StationaryBellmanOperator(EnvParams<RewardFuncT, T...> env_params,
                              const StationaryBellmanOperatorParams& operator_params,
                              RandomDeviceT* random_device)
        : env_params_{std::move(env_params)},
          operator_params_{operator_params},
          random_device_{random_device},
          qfunc_primary_{operator_params_.num_particles, env_params_.ac_kernel.GetDim()},
          qfunc_secondary_{operator_params_.num_particles, env_params_.ac_kernel.GetDim()},
          additional_weights_{operator_params_.num_particles} {

        std::uniform_real_distribution<FloatT> distr{-operator_params_.init_radius,
                                                     operator_params_.init_radius};

        RandomVectorizingInitializer<MemoryView, decltype(distr), RandomDeviceT> initializer{
            env_params_.ac_kernel.GetSpaceDim(), random_device, distr};

        density_estimator_ = std::make_unique<StationaryDensityEstimator<EstimatorKernelT>>(
            nullptr, initializer, operator_params_.num_particles);

        UpdateParticleCluster(operator_params_.num_burnin_iterations);

        qfunc_primary_.SetParticleCluster(density_estimator_->GetCluster());
    }

    void MakeIteration() {
#ifndef NDEBUG
        feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif
        UpdateParticleCluster(4);

        auto& cluster = qfunc_primary_.GetParticleCluster();
        assert(additional_weights_.size() == cluster.size());
        assert(additional_weights_[0].size() == env_params_.ac_kernel.GetDim());

        GreedyPolicy policy{qfunc_primary_};

        auto& ac_kernel = env_params_.ac_kernel;

        for (size_t i = 0; i < cluster.size(); ++i) {
            for (size_t action_number = 0; action_number < ac_kernel.GetDim(); ++action_number) {
                qfunc_secondary_.ValueAtIndex(i, action_number) =
                    env_params_.reward_function(cluster[i], action_number);
            }
            for (size_t j = 0; j < cluster.size(); ++j) {
                for (size_t action_number = 0; action_number < ac_kernel.GetDim();
                     ++action_number) {
                    size_t reaction = policy.React(j);
                    FloatT density = ac_kernel.GetTransDensityConditionally(cluster[i], cluster[j],
                                                                            action_number) /
                                     density_estimator_->GetCluster().GetWeights()[j];

                    if (density > operator_params_.density_ratio_threshold) {
                        continue;
                    }

                    qfunc_secondary_.ValueAtIndex(i, action_number) +=
                        env_params_.kGamma * density * qfunc_primary_.ValueAtIndex(j, reaction) *
                        additional_weights_[i][action_number];
                }
            }
        }

        std::swap(qfunc_primary_, qfunc_secondary_);
        qfunc_primary_.SetParticleCluster(std::move(qfunc_secondary_.GetParticleCluster()));
    }

    DiscreteQFuncEst& GetQFunc() {
        return qfunc_primary_;
    }

    const DiscreteQFuncEst& GetQFunc() const {
        return qfunc_primary_;
    }

private:
    void UpdateParticleCluster(size_t num_iterations) {
        GreedyPolicy policy{qfunc_primary_};
        MDPKernel mdp_kernel{env_params_.ac_kernel, &policy};
        density_estimator_->ResetKernel(&mdp_kernel);

        density_estimator_->MakeIteration(num_iterations);

        const WeightedParticleCluster& invariant_distr = density_estimator_->GetCluster();

        // Computing qfunction on new sample
        {
            auto prev_sample_reweighing =
                [prev_sample = prev_sampling_distribution_.get()](size_t sample_index) {
                    if (!prev_sample) {
                        return 1.;
                    }
                    return 1. / prev_sample->GetWeights()[sample_index];
                };

            QFuncEstForGreedy new_particles_estimator{env_params_, qfunc_primary_,
                                                      // should be previous density here
                                                      prev_sample_reweighing};

            DiscreteQFuncEst new_estimate{operator_params_.num_particles,
                                          env_params_.ac_kernel.GetDim()};

            for (size_t state_ix = 0; state_ix < operator_params_.num_particles; ++state_ix) {
                for (size_t action_num = 0; action_num < env_params_.ac_kernel.GetDim();
                     ++action_num) {
                    new_estimate.ValueAtIndex(state_ix, action_num) =
                        new_estimate.ValueAtPoint(invariant_distr[state_ix], action_num);
                }
            }

            new_estimate.SetParticleCluster(std::move(qfunc_primary_.GetParticleCluster()));
            qfunc_primary_ = std::move(new_estimate);
        }

        // Recomputing weights on new sample
        for (size_t action_num = 0; action_num < env_params_.ac_kernel.GetDim(); ++action_num) {
            for (size_t target_ix = 0; target_ix < operator_params_.num_particles; ++target_ix) {
                FloatT sum = 0;
                for (size_t particle_ix = 0; particle_ix < invariant_distr.size(); ++particle_ix) {
                    const auto& particle = invariant_distr[particle_ix];
                    FloatT pmass = invariant_distr.GetWeights()[particle_ix];
                    sum +=
                        env_params_.ac_kernel.GetTransDensityConditionally(
                            qfunc_primary_.GetParticleCluster()[target_ix], particle, action_num) /
                        pmass;
                }
                additional_weights_[target_ix][action_num] = 1 / sum;
            }
        }

        mdp_kernel.ResetPolicy(nullptr);
        density_estimator_->ResetKernel(nullptr);
        prev_sampling_distribution_ = std::make_unique<WeightedParticleCluster>(invariant_distr);
    }

    EnvParams<RewardFuncT, T...> env_params_;
    StationaryBellmanOperatorParams operator_params_;
    RandomDeviceT* random_device_;
    DiscreteQFuncEst qfunc_primary_, qfunc_secondary_;
    std::unique_ptr<StationaryDensityEstimator<EstimatorKernelT>> density_estimator_{nullptr};
    std::vector<std::array<FloatT, sizeof...(T)>> additional_weights_;
    std::unique_ptr<WeightedParticleCluster> prev_sampling_distribution_{nullptr};
};

// I feel sorry for this.
template <class RandomDeviceT, class RewardFuncT, class... T>
StationaryBellmanOperator(EnvParams<RewardFuncT, T...> env_params,
                          const StationaryBellmanOperatorParams&, RandomDeviceT*)
    ->StationaryBellmanOperator<RandomDeviceT, RewardFuncT,
                                decltype(MDPKernel{
                                    env_params.ac_kernel,
                                    static_cast<GreedyPolicy<DiscreteQFuncEst>*>(nullptr)}),
                                T...>;
