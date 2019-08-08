#pragma once

#include <utility>
#include <type_traits>
#include <memory>

template <class>
class PimplPtr;

template <class T, class... Args>
PimplPtr<T> MakePimplPtr(Args&&... args);

template <class T>
class PimplPtr : private std::unique_ptr<T, void (*)(T*)> {
    static_assert(std::is_class_v<T>, "PimplPtr is only intented for use with classes");

    using BaseT = std::unique_ptr<T, void (*)(T*)>;
    using DestCallerT = void (*)(T*);

    template <class S, class... Args>
    friend PimplPtr<S> MakePimplPtr(Args&&... args);

public:
    PimplPtr() = default;
    PimplPtr(PimplPtr&&) = default;
    PimplPtr& operator=(PimplPtr&&) = default;

    PimplPtr(const PimplPtr&) = delete;
    PimplPtr& operator=(const PimplPtr&) = delete;

    ~PimplPtr() = default;

    const T* operator->() const {
        return BaseT::operator->();
    }

    T* operator->() {
        return BaseT::operator->();
    }

    T& operator*() {
        return BaseT::operator*();
    }

    const T& operator*() const {
        return BaseT::operator*();
    }

    T* Release() const {
        return BaseT::release();
    }

    T* Get() {
        return BaseT::get();
    }

    const T* Get() const {
        return BaseT::get();
    }

    // no Reset here, on purpose -- how would one create a deleter using a pointer to an incomplete
    // type?
private:
    PimplPtr(T* ptr, DestCallerT&& dest_caller) noexcept : BaseT(ptr, dest_caller) {
    }
};

template <class T, class... Args>
PimplPtr<T> MakePimplPtr(Args&&... args) {
    return PimplPtr<T>{new T(std::forward<Args>(args)...), [](T* ptr) { delete ptr; }};
}
