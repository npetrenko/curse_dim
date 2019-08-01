#pragma once

#include "types.hpp"
#include "exceptions.hpp"
#include "memory_view.hpp"

#include <vector>
#include <array>
#include <type_traits>
#include <cassert>

template <class BaseContainer>
class ParticleStorageTemplate : private BaseContainer {
public:
    using iterator = typename BaseContainer::iterator;
    using const_iterator = typename BaseContainer::const_iterator;

    ParticleStorageTemplate() = default;

    ParticleStorageTemplate(size_t max_size) {
        BaseContainer::resize(max_size);
        current_pos_ = 0;
    }

    inline void resize(size_t size) {
        assert(size <= this->size());
        BaseContainer::resize(size);
    }

    inline MemoryView AllocateForParticle(size_t size) {
        if (current_pos_ + size > this->size()) {
            throw OutOfStorage{};
        }
        size_t old_pos = current_pos_;
        current_pos_ += size;

        return {this->data() + old_pos, size};
    }

    using BaseContainer::operator[];
    using BaseContainer::begin;
    using BaseContainer::end;
    using BaseContainer::size;

    inline void Clear() {
        current_pos_ = 0;
    }

    inline size_t GetCurrentSize() const {
        return current_pos_;
    }

private:
    size_t current_pos_{0};
};

class ParticleStorage : public ParticleStorageTemplate<std::vector<FloatT>> {
public:
    using ParticleStorageTemplate::ParticleStorageTemplate;
};

template <size_t size>
class StackParticleStorage  : public ParticleStorageTemplate<std::array<FloatT, size>> {
public:
    using ParticleStorageTemplate<std::array<FloatT, size>>::ParticleStorageTemplate;
};
