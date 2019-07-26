#pragma once

#include "../particle.hpp"
#include "../memory_view.hpp"

class AbstractWeightedParticleCluster : public ParticleCluster {
public:
    using ParticleCluster::ParticleCluster;
    explicit AbstractWeightedParticleCluster(ParticleCluster cluster)
        : ParticleCluster{std::move(cluster)} {
    }

    virtual ConstStridedMemoryView GetWeights() const = 0;
    virtual StridedMemoryView GetMutableWeights() = 0;
    virtual ~AbstractWeightedParticleCluster() = default;
};

class VectorWeightedParticleCluster final : public AbstractWeightedParticleCluster {
public:
    template <class S>
    VectorWeightedParticleCluster(size_t size,
                                  const AbstractInitializer<S, MemoryView>& initializer)
        : AbstractWeightedParticleCluster{size, initializer}, weights_(size) {
    }

    inline ConstMemoryView GetWeights() const override {
        return {weights_.data(), weights_.size()};
    }

    inline MemoryView GetMutableWeights() override {
        return {weights_.data(), weights_.size()};
    }

private:
    std::vector<FloatT> weights_;
};

class ConstantWeightedParticleCluster final : public AbstractWeightedParticleCluster {
public:
    template <class S>
    ConstantWeightedParticleCluster(size_t size,
                                    const AbstractInitializer<S, MemoryView>& initializer,
                                    FloatT weighing_constant)
        : AbstractWeightedParticleCluster{size, initializer},
          weighing_constant_{weighing_constant} {
    }

    ConstantWeightedParticleCluster(ParticleCluster cluster, FloatT weighing_constant)
        : AbstractWeightedParticleCluster{std::move(cluster)},
          weighing_constant_{weighing_constant} {
    }

    inline StridedMemoryView GetMutableWeights() override {
        throw std::runtime_error(
            "GetMutableWeights for ConstantWeightedParticleCluster makes no sense");
    }

    inline ConstStridedMemoryView GetWeights() const override {
        return {&weighing_constant_, this->size(), 0};
    }

private:
    mutable std::vector<FloatT> weights_;
    const FloatT weighing_constant_;
};
