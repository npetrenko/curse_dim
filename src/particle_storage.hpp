#pragma once

#include "types.hpp"
#include "exceptions.hpp"
#include "memory_view.hpp"

#include <vector>
#include <type_traits>
#include <cassert>

class ParticleStorage : private std::vector<FloatT> {
public:
    using iterator = std::vector<FloatT>::iterator;
    using const_iterator = std::vector<FloatT>::const_iterator;

    ParticleStorage(size_t max_size) {
        std::vector<FloatT>::resize(max_size);
        current_pos_ = 0;
    }

    inline void resize(size_t size) {
        assert(size <= this->size());
        std::vector<FloatT>::resize(size);
    }

    inline MemoryView AllocateForParticle(size_t size) {
        if (current_pos_ + size > this->size()) {
            throw OutOfStorage{};
        }
        size_t old_pos = current_pos_;
        current_pos_ += size;

        return {this->data() + old_pos, size};
    }

    inline FloatT& operator[](size_t pos) {
        return *(this->begin() + pos);
    }

    inline FloatT operator[](size_t pos) const {
        return *(this->begin() + pos);
    }

    inline iterator begin() {
        return std::vector<FloatT>::begin();
    }

    inline iterator end() {
        return std::vector<FloatT>::end();
    }

    inline const_iterator begin() const {
        return std::vector<FloatT>::cbegin();
    }

    inline const_iterator end() const {
        return std::vector<FloatT>::cend();
    }

    inline size_t size() const {
        return std::vector<FloatT>::size();
    }

    inline void Clear() {
        current_pos_ = 0;
    }

    inline size_t GetCurrentSize() const {
        return current_pos_;
    }

private:
    size_t current_pos_;
};
