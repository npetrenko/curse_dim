#pragma once

#include <src/bellman_operators/qfunc.hpp>
#include <src/density_estimators/stationary_estimator.hpp>

class AbstractBellmanOperator {
public:
    virtual void MakeIteration() = 0;
    virtual const DiscreteQFuncEst& GetQFunc() const = 0;
    virtual const AbstractWeightedParticleCluster& GetSamplingDistribution() const = 0;
    virtual ~AbstractBellmanOperator() = default;
};
