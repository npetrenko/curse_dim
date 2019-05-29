#pragma once

#include <vector>

#include <src/types.hpp>
#include <src/util.hpp>
#include <src/initializer.hpp>
#include <src/particle_storage.hpp>

class ParticleCluster;

template <class StorageT = ParticleStorage>
class Particle {
    friend class ParticleCluster;

public:
    template <class T>
    Particle(const AbstractInitializer<T, StorageT>& initializer)
        : data_{initializer.CreateStorage()} {
        initializer.Initialize(&data_);
    }

    template <class S>
    Particle(const Particle<S>& other) : data_{other.GetDim()} {
        static_assert(!std::is_same_v<StorageT, MemoryView>,
                      "One shouldn't explicitly copy-initialize memory view");
        std::copy(other.data_.begin(), other.data_.end(), data_.begin());
    }

    template <class S>
    Particle& operator=(const Particle<S>& other) {
        assert(GetDim() == other.GetDim());
        std::copy(other.data_.begin(), other.data_.end(), data_.begin());
    }

    inline size_t GetDim() const {
        return data_.size();
    }

    inline FloatT& operator[](size_t i) {
        return data_[i];
    }

    inline FloatT operator[](size_t i) const {
        return data_[i];
    }

    template <class T>
    bool operator==(const Particle<T>& other) const {
        if (GetDim() != other.GetDim()) {
            return false;
        }
        for (size_t i = 0; i < GetDim(); ++i) {
            if ((*this)[i] != other[i]) {
                return false;
            }
        }
        return true;
    }

    Particle& operator*=(FloatT val) {
        for (auto& x : data_) {
            x *= val;
        }
        return *this;
    }

    FloatT NormSquared() const {
        FloatT norm{0};
        for (auto& x : data_) {
            norm += x * x;
        }

        return norm;
    }

    template <class S>
    Particle& operator-=(const Particle<S>& other) {
        for (size_t i = 0; i < GetDim(); ++i) {
            (*this)[i] -= other[i];
        }

        return *this;
    }

    template <class S>
    Particle& operator+=(const Particle<S>& other) {
        for (size_t i = 0; i < GetDim(); ++i) {
            (*this)[i] += other[i];
        }

        return *this;
    }

    template <class S>
    friend std::ostream& operator<<(std::ostream& stream, const Particle<S>& part);

    using iterator = typename StorageT::iterator;
    inline iterator begin() {
        return data_.begin();
    }

    inline iterator end() {
        return data_.end();
    }

private:
    StorageT data_;
};

template <class StorageT>
std::ostream& operator<<(std::ostream& stream, const Particle<StorageT>& part) {
    stream << "Particle" << part.data_;
    return stream;
}

class ParticleCluster : private std::vector<Particle<MemoryView>> {
public:
    template <class DerivedT>
    ParticleCluster(size_t size, const AbstractInitializer<DerivedT, MemoryView>& initializer)
        : storage_(size * initializer.GetDim()) {
        initializer.SetStorage(&storage_);
        this->reserve(size);
        for (size_t i = 0; i < size; ++i) {
            this->emplace_back(initializer);
        }
    }

    using ParentT = std::vector<Particle<MemoryView>>;
    ParticleCluster(const ParticleCluster& other) : ParentT{other}, storage_{other.storage_} {
        ResetChildParticles(other.storage_.GetCurrentSize());
    }

    ParticleCluster(ParticleCluster&&) = default;

    ParticleCluster& operator=(const ParticleCluster& other) {
	if (&other == this) {
	    return *this;
	}
        static_cast<ParentT&>(*this) = static_cast<const ParentT&>(other);
        storage_ = other.storage_;
        ResetChildParticles(other.storage_);
        return *this;
    }

    ParticleCluster& operator=(ParticleCluster&&) = default;

    using ParticleT = Particle<MemoryView>;
    inline ParticleT& operator[](size_t i) {
        return static_cast<ParentT&>(*this)[i];
    }

    inline const ParticleT& operator[](size_t i) const {
        return static_cast<const ParentT&>(*this)[i];
    }

    using iterator = ParentT::iterator;
    using const_iterator = ParentT::const_iterator;

    inline iterator begin() {
        return ParentT::begin();
    }

    inline iterator end() {
        return ParentT::end();
    }

    inline const_iterator begin() const {
        return ParentT::cbegin();
    }

    inline const_iterator end() const {
        return ParentT::cend();
    }

    inline size_t size() const {
        return ParentT::size();
    }

private:
    // Reallocating all allocated particles. Since particles can only be stored in-order, and they
    // cannot be remove from the storage in the middle, this is indeed a correct code
    void ResetChildParticles(const ParticleStorage& origin_storage) {
        const size_t ensure_size = origin_storage.size();
        FloatT* storage_ptr = &storage_[0];
        for (auto& particle : *this) {
            particle.data_ = MemoryView{storage_ptr, particle.data_.size()};
            storage_ptr += particle.data_.size();
        }
        assert(ensure_size == static_cast<size_t>(storage_ptr - &storage_[0]));
	(void)ensure_size;
    }
    ParticleStorage storage_;
};
