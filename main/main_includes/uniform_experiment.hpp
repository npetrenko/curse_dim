#include <main/main_includes/experiment.hpp>

namespace UniformExperiment {
std::unique_ptr<AbstractExperiment> Make(AbstractExperiment::Builder builder);
}
