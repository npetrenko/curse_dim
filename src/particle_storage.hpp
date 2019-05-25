#pragma once

#include <vector>
#include <src/types.hpp>
#include <src/exceptions.hpp>
#include <type_traits>
#include <cassert>

template <class T, bool is_const>
class ContiguousIterator {
public:
    using finT = std::conditional_t<is_const, const T*, T*>;
    using retT = std::conditional_t<is_const, const T&, T&>;
    using value_type = T;

    ContiguousIterator(finT data) noexcept : data_(data) {
    }

    inline retT operator*() const {
        return *data_;
    }

    inline ContiguousIterator& operator++() {
        ++data_;
        return *this;
    }

    inline ContiguousIterator operator++(int) {
        ContiguousIterator prev = *this;
        ++data_;
        return prev;
    }

    template <class K, bool ic>
    inline bool operator!=(ContiguousIterator<K, ic> other) {
        return other.data_ != data_;
    }

private:
    finT data_;
};

class MemoryView {
public:
    using iterator = ContiguousIterator<FloatT, false>;

    MemoryView(FloatT* data, size_t size) noexcept : data_(data), size_(size) {
    }

    inline FloatT& operator[](size_t pos) {
	return data_[pos];
    }

    inline FloatT operator[](size_t pos) const {
	return data_[pos];
    }

    inline size_t size() const {
        return size_;
    }

    inline iterator begin() const {
        return {data_};
    }

    inline iterator end() const {
        return {data_ + size_};
    }

private:
    FloatT* data_;
    size_t size_;
};

class ParticleStorage : private std::vector<FloatT> {
public:
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

private:
    size_t current_pos_;
};
