#include <curse_dim/experiment_printer.hpp>

std::ostream& Printer::PrintHeader(std::ostream& stream) {
    stream << "ExperimentName, Score, NumParticles, NumPendulums, NumIterations, Gamma, QFDur, "
              "SimDur\n";
    return stream;
}

std::ostream& Printer::PrintBody(std::ostream& stream, const AbstractExperiment& experiment) {
    auto print_ms = [](std::ostream& stream, std::chrono::milliseconds ms) {
        stream << ms.count() << "ms";
    };

    stream << experiment.GetName() << ", " << experiment.GetScore() << ", "
           << experiment.GetNumParticles() << ", " << experiment.GetNumPendulums() << ", "
           << experiment.GetNumIterations() << ", " << experiment.GetGamma();

    stream << ", ";
    print_ms(stream, experiment.GetQFuncEstDuration());
    stream << ", ";
    print_ms(stream, experiment.GetSimDuration());
    stream << "\n";
    return stream;
}
