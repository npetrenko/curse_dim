#pragma once

#include <vector>
#include <src/types.hpp>
#include <iostream>
#include <src/util.hpp>
#include <src/particle_storage.hpp>

template <class StorageT = ParticleStorage>
class Particle {
public:
    template <class T>
    Particle(const AbstractInitializer<T, StorageT>& initializer) : data_{initializer.CreateStorage()} {
        initializer.Initialize(&data_);
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

    template<class S>
    friend std::ostream& operator<<(std::ostream& stream, const Particle<S>& part);

private:
    StorageT data_;
};

template <class StorageT>
std::ostream& operator<<(std::ostream& stream, const Particle<StorageT>& part) {
    stream << "Particle" << part.data_;
    return stream;
}
