#pragma once

#include "named_value.hpp"

using FloatT = double;

using ParticleDim = NamedValue<std::size_t, struct _ParticleDimTag>;
using NumParticles = NamedValue<std::size_t, struct _NumParticlesTag>;
using NumActions = NamedValue<std::size_t, struct _NumActionsTag>;
