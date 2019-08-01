#pragma once

#include <string>
#include <exception>

class NotImplementedError : public std::exception {
    inline const char* what() const noexcept override {
	return "Not implemented";
    }
};

class OutOfStorage : public std::exception {
};

class BuilderNotInitialized : public std::exception {
    inline const char* what() const noexcept override {
	return "BuilderNotInitialized: some of the used parameters were not set";
    }
};

class CloneException : public std::exception {
    inline const char* what() const noexcept override {
	return "Clone implementation has not been overriden";
    }
};
