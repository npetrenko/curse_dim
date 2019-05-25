#pragma once

#include <vector>

#include <src/types.hpp>
#include <src/util.hpp>
#include <src/initializer.hpp>
#include <src/particle_storage.hpp>

template <class StorageT = ParticleStorage>
class Particle {
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

    size_t GetDim() const {
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
    Particle<ParticleStorage> operator-(const Particle<S>& other) {
        Particle<ParticleStorage> copy{*this};
        copy -= other;
        return copy;
    }

    template <class S>
    friend std::ostream& operator<<(std::ostream& stream, const Particle<S>& part);

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

    using ParticleT = Particle<MemoryView>;
    inline ParticleT& operator[](size_t i) {
        return static_cast<ParentT&>(*this)[i];
    }

    inline const ParticleT& operator[](size_t i) const {
        return static_cast<const ParentT&>(*this)[i];
    }

    using ParentT = std::vector<Particle<MemoryView>>;
    using iterator = ParentT::iterator;
    using const_iterator = ParentT::const_iterator;

    inline iterator begin() {
        return ParentT::begin();
    }

    inline iterator end() {
        return ParentT::end();
    }

    inline const_iterator cbegin() const {
        return ParentT::cbegin();
    }

    inline const_iterator end() const {
        return ParentT::cend();
    }

    inline size_t size() const {
        return ParentT::size();
    }

private:
    ParticleStorage storage_;
};
