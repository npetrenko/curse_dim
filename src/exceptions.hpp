#include <string>
#include <exception>

class NotImplementedError : public std::exception {
    const char* what() const noexcept override {
	return "Not implemented";
    }
};
