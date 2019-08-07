#pragma once

#include <optional>
#include <string_view>

#include "exceptions.hpp"

template <class ValueType>
class BuilderOption : private std::optional<ValueType> {
    using BaseT = std::optional<ValueType>;

public:
    BuilderOption(const char* param_name) noexcept : param_name_(param_name) {
    }

    using BaseT::operator=;
    using BaseT::operator bool;

    const ValueType& Value() const {
        if (*this) {
            return **this;
        } else {
            throw BuilderNotInitialized(param_name_);
        }
    }

    ValueType& Value() {
        if (*this) {
            return **this;
        } else {
            throw BuilderNotInitialized(param_name_);
        }
    }

private:
    std::string_view param_name_;
};
