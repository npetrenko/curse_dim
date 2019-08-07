#pragma once

#include <string_view>
#include <string>
#include <exception>

class NotImplementedError : public std::exception {
    inline const char* what() const noexcept override {
        return "Not implemented";
    }
};

class OutOfStorage : public std::exception {};

class CloneException : public std::exception {
    inline const char* what() const noexcept override {
        return "Clone implementation has not been overriden";
    }
};

template <class>
class BuilderOption;

class BuilderNotInitialized : public std::exception {
    template <class>
    friend class BuilderOption;

private:
    inline BuilderNotInitialized(std::string_view param_name) {
        constexpr std::string_view is_not_set = " is not set";
        err_msg.append(param_name);
        err_msg.append(is_not_set);
    }

public:
    inline const char* what() const noexcept override {
        return err_msg.c_str();
    }

private:
    std::string err_msg = "BuilderNotInitialized: ";
};
