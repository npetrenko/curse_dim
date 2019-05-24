#include <string>
#include <exception>

class NotImplementedError : public std::exception {
    const char* what() const override {
	return "Not implemented";
    }
};
