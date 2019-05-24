#pragma once

#include <vector>
#include <src/types.hpp>
#include <iostream>
#include <src/util.hpp>

class Particle {
public:
    template <class InitT>
    Particle(const InitT& initializer) {
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

    friend std::ostream& operator<<(std::ostream& stream, const Particle& part);

private:
    std::vector<FloatT> data_;
};

std::ostream& operator<<(std::ostream& stream, const Particle& part) {
    stream << "Particle" << part.data_;
    return stream;
}
