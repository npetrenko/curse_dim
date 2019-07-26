#pragma once

#include <vector>
#include <iterator>

#include "types.hpp"
#include "util.hpp"
#include "initializer.hpp"
#include "particle_storage.hpp"


class ParticleCluster;

template <class StorageT = ParticleStorage>
class Particle {
    friend class ParticleCluster;

public:
    using reference = typename std::iterator_traits<typename StorageT::iterator>::reference;

    template <bool is_const>
    Particle(MemoryViewTemplate<is_const> view) : data_{view} {
    }

    template <class T>
    Particle(const AbstractInitializer<T, StorageT>& initializer);

    template <class S>
    Particle(const Particle<S>& other);

    template <class S>
    Particle& operator=(const Particle<S>& other) {
        assert(GetDim() == other.GetDim());
        std::copy(other.data_.begin(), other.data_.end(), data_.begin());
    }

    inline size_t GetDim() const {
        return data_.size();
    }

    inline reference operator[](size_t i) {
        return data_[i];
    }

    inline FloatT operator[](size_t i) const {
        return data_[i];
    }

    template <class T>
    bool operator==(const Particle<T>& other) const;

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
    using const_iterator = typename StorageT::const_iterator;

    inline iterator begin() {
        return data_.begin();
    }

    inline iterator end() {
        return data_.end();
    }

    inline const_iterator begin() const {
        return data_.begin();
    }

    inline const_iterator end() const {
        return data_.end();
    }

private:
    StorageT data_;
};

template <bool is_const>
Particle(MemoryViewTemplate<is_const>)->Particle<MemoryViewTemplate<is_const>>;

class TypeErasedParticleRef : public Particle<ConstMemoryView> {
    using ParentT = Particle<ConstMemoryView>;
public:
    template <class StorageT>
    TypeErasedParticleRef(const Particle<StorageT>& part) : ParentT(ConstMemoryView(&*part.begin(), part.GetDim())) {
    }
};

class TypeErasedParticlePtr {
public:
    template <class StorageT>
    TypeErasedParticlePtr(Particle<StorageT>* part) : data_(MemoryView(&*part->begin(), part->GetDim())) {
    }

    Particle<MemoryView>& operator*() const {
	return data_;
    }

    Particle<MemoryView>* operator->() const {
	return &data_;
    }

private:
    mutable Particle<MemoryView> data_;
};

class ParticleCluster : private std::vector<Particle<MemoryView>> {
    using ParentT = std::vector<Particle<MemoryView>>;
    using ParticleT = Particle<MemoryView>;
    using iterator = ParentT::iterator;
    using const_iterator = ParentT::const_iterator;

public:
    template <class DerivedT>
    ParticleCluster(size_t size, const AbstractInitializer<DerivedT, MemoryView>& initializer);
    ParticleCluster(const ParticleCluster& other);
    ParticleCluster(ParticleCluster&&) = default;
    ParticleCluster& operator=(const ParticleCluster& other);
    ParticleCluster& operator=(ParticleCluster&&) = default;

    inline ParticleT& operator[](size_t i) {
        return static_cast<ParentT&>(*this)[i];
    }

    inline const ParticleT& operator[](size_t i) const {
        return static_cast<const ParentT&>(*this)[i];
    }

    inline iterator begin() {
        return ParentT::begin();
    }

    inline iterator end() {
        return ParentT::end();
    }

    inline const_iterator begin() const {
        return ParentT::begin();
    }

    inline const_iterator end() const {
        return ParentT::end();
    }

    inline size_t size() const {
        return ParentT::size();
    }

    inline ParticleStorage& GetStorage() {
        return storage_;
    }

private:
    // Makes all particles point to the right ParticleStorage
    void ResetChildParticles(const ParticleStorage& origin_storage);
    ParticleStorage storage_;
};

template <class StorageT>
std::ostream& operator<<(std::ostream& stream, const Particle<StorageT>& part) {
    stream << "Particle" << part.data_;
    return stream;
}

/////////////////////////// Implementation////////////////////////////

template <class StorageT>
template <class T>
Particle<StorageT>::Particle(const AbstractInitializer<T, StorageT>& initializer)
    : data_{initializer.CreateStorage()} {
    initializer.Initialize(&data_);
}

template <class StorageT>
template <class S>
Particle<StorageT>::Particle(const Particle<S>& other) : data_{other.GetDim()} {
    static_assert(!std::is_same_v<StorageT, MemoryView>,
                  "One shouldn't explicitly copy-initialize memory view");
    std::copy(other.data_.begin(), other.data_.end(), data_.begin());
}

template <class StorageT>
template <class T>
bool Particle<StorageT>::operator==(const Particle<T>& other) const {
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

template <class DerivedT>
ParticleCluster::ParticleCluster(size_t size,
                                 const AbstractInitializer<DerivedT, MemoryView>& initializer)
    : storage_(size * initializer.GetDim()) {
    initializer.SetStorage(&storage_);
    this->reserve(size);
    for (size_t i = 0; i < size; ++i) {
        this->emplace_back(initializer);
    }
}
