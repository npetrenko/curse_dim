#pragma once

#include <vector>
#include <src/types.hpp>

class Particle {
public:
    template<class InitT>
    Particle(InitT& initializer) {
	initializer.Initialize(&data_);
    }

    size_t GetDim() const {
	return data_.size();
    }

private:
    std::vector<FloatT> data_; 
};
