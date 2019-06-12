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

#include <glog/logging.h>

#ifndef NDEBUG
#include <fenv.h>
#endif

#include <thread_pool/include/for_loop.hpp>

struct StationaryBellmanOperatorParams {
    size_t num_particles;
    FloatT density_ratio_threshold, init_radius;
    FloatT invariant_density_threshold;
    size_t num_burnin_iterations = 10;
};

struct PrevSampleReweighingHelper {
    const WeightedParticleCluster* prev_sample;
    std::optional<FloatT> default_density;
    FloatT operator()(size_t sample_index) const {
        if (!prev_sample) {
            return default_density.value();
        }
        FloatT dens = prev_sample->GetWeights()[sample_index];
        if (dens < 1e-2) {
            return 0;
        }
        return 1. / dens;
    }
};

template <class RandomDeviceT, class RewardFuncT, class EstimatorKernelT, class... T>
class StationaryBellmanOperator {
public:
    StationaryBellmanOperator(const EnvParams<RewardFuncT, T...>& env_params,
                              const StationaryBellmanOperatorParams& operator_params,
                              RandomDeviceT* random_device)
        : env_params_{env_params},
          operator_params_{operator_params},
          random_device_{random_device},
          qfunc_primary_{operator_params_.num_particles, env_params_.ac_kernel.GetDim()},
          qfunc_secondary_{operator_params_.num_particles, env_params_.ac_kernel.GetDim()},
          additional_weights_(operator_params_.num_particles) {

        assert(operator_params.init_radius > 0);
        std::uniform_real_distribution<FloatT> distr{-operator_params_.init_radius,
                                                     operator_params_.init_radius};

        RandomVectorizingInitializer<MemoryView, decltype(distr), RandomDeviceT> initializer{
            env_params_.ac_kernel.GetSpaceDim(), random_device, distr};

        density_estimator_ = std::make_unique<StationaryDensityEstimator<EstimatorKernelT, std::false_type>>(
            nullptr, initializer, operator_params_.num_particles);

        qfunc_primary_.SetParticleCluster(density_estimator_->GetCluster());
        qfunc_secondary_.SetParticleCluster(density_estimator_->GetCluster());
        {
            std::uniform_real_distribution<FloatT> q_init{-0.01, 0.01};
            qfunc_primary_.SetRandom(random_device, q_init);
            qfunc_secondary_.SetRandom(random_device, q_init);
        }

        LOG(INFO) << "Initializing particle cluster with " << operator_params_.num_burnin_iterations
                  << " iterations";
        UpdateParticleCluster(operator_params_.num_burnin_iterations);
        LOG(INFO) << "Cluster is initialized";
    }

    void MakeIteration() {
#ifndef NDEBUG
        feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif
        UpdateParticleCluster(1);

        const ParticleCluster& cluster = qfunc_primary_.GetParticleCluster();
        GreedyPolicy policy{qfunc_primary_};

        auto& ac_kernel = env_params_.ac_kernel;

        ParallelFor{0, operator_params_.num_particles, 255}([&](size_t from_ix) {
            if (density_estimator_->GetCluster().GetWeights()[from_ix] <
                operator_params_.invariant_density_threshold) {
                return;
            }
            for (size_t action_number = 0; action_number < ac_kernel.GetDim(); ++action_number) {
                qfunc_secondary_.ValueAtIndex(from_ix)[action_number] =
                    env_params_.reward_function(cluster[from_ix], action_number);
            }
            for (size_t to_ix = 0; to_ix < operator_params_.num_particles; ++to_ix) {
                for (size_t action_number = 0; action_number < ac_kernel.GetDim();
                     ++action_number) {
                    FloatT stationary_density =
                        density_estimator_->GetCluster().GetWeights()[to_ix];
                    if (stationary_density < operator_params_.invariant_density_threshold) {
                        continue;
                    }
                    size_t reaction = policy.React(to_ix);
                    FloatT weighted_density = ac_kernel.GetTransDensityConditionally(
                                                  cluster[from_ix], cluster[to_ix], action_number) /
                                              stationary_density;

                    // LOG(INFO) << weighted_density << " " << stationary_density << " "
                    //<< qfunc_primary_.ValueAtIndex(j)[reaction] << " "
                    //<< additional_weights_[i][action_number];

                    if (qfunc_primary_.ValueAtIndex(to_ix)[reaction] > 1e2) {
                        LOG(INFO) << "Bad qfunc value, terminating";
                        std::terminate();
                    }

                    if (weighted_density > operator_params_.density_ratio_threshold) {
                        continue;
                    }

                    qfunc_secondary_.ValueAtIndex(from_ix)[action_number] +=
                        env_params_.kGamma * weighted_density *
                        qfunc_primary_.ValueAtIndex(to_ix)[reaction] *
                        additional_weights_[from_ix][action_number] /
                        operator_params_.num_particles;
                }
            }
        });

        std::swap(qfunc_primary_, qfunc_secondary_);
    }

    DiscreteQFuncEst& GetQFunc() {
        return qfunc_primary_;
    }

    const DiscreteQFuncEst& GetQFunc() const {
        return qfunc_primary_;
    }

    const WeightedParticleCluster* GetSamplingDistribution() {
        return prev_sampling_distribution_.get();
    }

private:
    void UpdateParticleCluster(size_t num_iterations) {
#ifndef NDEBUG
        feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif
        PrevSampleReweighingHelper prev_sample_reweighing{
            prev_sampling_distribution_.get(),
            1 / pow(2 * operator_params_.init_radius, env_params_.ac_kernel.GetSpaceDim())};

        QFuncEstForGreedy current_qfunc_estimator{env_params_, qfunc_primary_,
                                                  // should be previous density here
                                                  prev_sample_reweighing};
        GreedyPolicy policy{current_qfunc_estimator};
        MDPKernel mdp_kernel{env_params_.ac_kernel, &policy};
        density_estimator_->ResetKernel(&mdp_kernel);

        LOG(INFO) << "Making stationary iterations";
        density_estimator_->template MakeIteration<true, RandomDeviceT>(num_iterations,
                                                                        random_device_);
        LOG(INFO) << "Finished";

        const WeightedParticleCluster& invariant_distr = density_estimator_->GetCluster();

        // Computing qfunction on new sample
        {
            DiscreteQFuncEst new_estimate{operator_params_.num_particles,
                                          env_params_.ac_kernel.GetDim()};

            ParallelFor{0, operator_params_.num_particles, 255}([&](size_t state_ix) {
                for (size_t action_num = 0; action_num < env_params_.ac_kernel.GetDim();
                     ++action_num) {
                    new_estimate.ValueAtIndex(state_ix)[action_num] =
                        current_qfunc_estimator.ValueAtPoint(invariant_distr[state_ix], action_num);
                }
            });

            new_estimate.SetParticleCluster(invariant_distr);
            qfunc_primary_ = std::move(new_estimate);
        }

        mdp_kernel.ResetPolicy(nullptr);
        density_estimator_->ResetKernel(nullptr);
        prev_sampling_distribution_ = std::make_unique<WeightedParticleCluster>(invariant_distr);

        RecomputeWeights();
    }

    void RecomputeWeights() {
        const WeightedParticleCluster& invariant_distr = density_estimator_->GetCluster();
        for (size_t action_num = 0; action_num < env_params_.ac_kernel.GetDim(); ++action_num) {
            ParallelFor{0, operator_params_.num_particles, 255}([&](size_t target_ix) {
                FloatT sum = 0;
                for (size_t particle_ix = 0; particle_ix < operator_params_.num_particles;
                     ++particle_ix) {
                    const auto& particle = invariant_distr[particle_ix];

                    // ensure that checks run only once
                    if (!target_ix && !action_num) {
                        assert(particle == qfunc_primary_.GetParticleCluster()[particle_ix]);
                    }

                    FloatT pmass = invariant_distr.GetWeights()[particle_ix];
                    if (pmass == 0) {
                        continue;
                    }
                    sum +=
                        env_params_.ac_kernel.GetTransDensityConditionally(
                            qfunc_primary_.GetParticleCluster()[target_ix], particle, action_num) /
                        (pmass * operator_params_.num_particles);
                }
                additional_weights_[target_ix][action_num] = sum ? (1 / sum) : 0;

                if ((sum > 10) || (sum && (sum < 0.1))) {
                    LOG(INFO) << "Strange additional weights: " << 1 / sum;
                    std::terminate();
                }
            });
        }
    }

    const EnvParams<RewardFuncT, T...> env_params_;
    const StationaryBellmanOperatorParams operator_params_;
    RandomDeviceT* const random_device_;
    DiscreteQFuncEst qfunc_primary_, qfunc_secondary_;
    std::unique_ptr<StationaryDensityEstimator<EstimatorKernelT, std::false_type>> density_estimator_{nullptr};
    std::vector<std::array<FloatT, sizeof...(T)>> additional_weights_;
    std::unique_ptr<WeightedParticleCluster> prev_sampling_distribution_{nullptr};
};

// I feel sorry for this.
template <class RandomDeviceT, class RewardFuncT, class... T>
StationaryBellmanOperator(EnvParams<RewardFuncT, T...> env_params,
                          const StationaryBellmanOperatorParams&, RandomDeviceT*)
    ->StationaryBellmanOperator<
        RandomDeviceT, RewardFuncT,
        decltype(MDPKernel{env_params.ac_kernel,
                           static_cast<GreedyPolicy<QFuncEstForGreedy<
                               RewardFuncT, PrevSampleReweighingHelper, T...>>*>(nullptr)}),
        T...>;
