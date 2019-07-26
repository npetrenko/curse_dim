#include <include/particle.hpp>

ParticleCluster::ParticleCluster(const ParticleCluster& other)
    : ParentT{other}, storage_{other.storage_} {
    ResetChildParticles(other.storage_);
}

ParticleCluster& ParticleCluster::operator=(const ParticleCluster& other) {
    if (&other == this) {
        return *this;
    }
    static_cast<ParentT&>(*this) = static_cast<const ParentT&>(other);
    storage_ = other.storage_;
    ResetChildParticles(other.storage_);
    return *this;
}

// Reallocating all allocated particles. Since particles can only be stored in-order, and they
// cannot be remove from the storage in the middle, this is indeed a correct code
void ParticleCluster::ResetChildParticles(const ParticleStorage& origin_storage) {
    const size_t ensure_size = origin_storage.size();
    FloatT* storage_ptr = &storage_[0];
    for (auto& particle : *this) {
        particle.data_ = MemoryView{storage_ptr, particle.data_.size()};
        storage_ptr += particle.data_.size();
    }
    assert(ensure_size == static_cast<size_t>(storage_ptr - &storage_[0]));
    (void)ensure_size;
}
