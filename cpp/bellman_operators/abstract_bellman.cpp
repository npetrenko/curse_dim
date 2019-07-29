#include <include/bellman_operators/abstract_bellman.hpp>

AbstractBellmanOperator::AbstractBellmanOperator(Params&& params) : kParams(std::move(params)) {
}
