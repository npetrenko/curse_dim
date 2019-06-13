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

    inline ContiguousIterator operator+(int64_t val) const {
        return {data_ + val};
    }

    template <class K, bool ic>
    inline bool operator!=(ContiguousIterator<K, ic> other) {
        return other.data_ != data_;
    }

private:
    finT data_;
};

template <bool is_const>
class MemoryViewTemplate {
public:
    using iterator = ContiguousIterator<FloatT, is_const>;
    using const_iterator = ContiguousIterator<FloatT, true>;
    using DataPtrT = typename iterator::finT;
    using ReferenceT = typename iterator::retT;

    MemoryViewTemplate(DataPtrT data, size_t size) noexcept : data_(data), size_(size) {
    }

    inline ReferenceT operator[](size_t pos) const {
        return data_[pos];
    }

    inline size_t size() const {
        return size_;
    }

    inline const_iterator begin() const {
        return {data_};
    }

    inline const_iterator end() const {
        return {data_ + size_};
    }

    inline iterator begin() {
        return {data_};
    }

    inline iterator end() {
        return {data_ + size_};
    }

    DataPtrT GetDataPtr() const {
        return data_;
    }

private:
    DataPtrT data_;
    size_t size_;
};

class MemoryView : public MemoryViewTemplate<false> {
    using MemoryViewTemplate<false>::MemoryViewTemplate;

public:
    MemoryView(MemoryViewTemplate<false> origin) : MemoryViewTemplate<false>{origin} {
    }
};

class ConstMemoryView : public MemoryViewTemplate<true> {
    using MemoryViewTemplate<true>::MemoryViewTemplate;

public:
    ConstMemoryView(MemoryViewTemplate<true> origin) : MemoryViewTemplate<true>{origin} {
    }
};

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
