#include <curse_dim/experiment.hpp>
#include <curse_dim/uniform_experiment.hpp>
#include <curse_dim/stationary_experiment.hpp>
#include <curse_dim/experiment_printer.hpp>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <fstream>

DEFINE_string(output_file, "/tmp/output.csv", "File to output experiment results to");

int main(int argc, char** argv) {
    google::LogToStderr();
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::mt19937 rd{1234};
    Printer printer;

    std::ofstream output_file;
    output_file.open(FLAGS_output_file.c_str(), std::ios::trunc);

    printer.PrintHeader(output_file);
    output_file.close();
    for (const size_t kNumPendulums : {1}) {
        for (size_t kNumParticles = 128; kNumParticles < 2048; kNumParticles += 128) {
            AbstractExperiment::Builder builder;
            builder.SetNumParticles(kNumParticles)
                .SetNumPendulums(kNumPendulums)
                .SetRandomDevice(std::mt19937(rd));
            std::unique_ptr uniform_experiment = UniformExperiment::Make(builder);
            std::unique_ptr stationary_experiment = StationaryExperiment::Make(builder);

            for (size_t iter = 1; iter < 40; ++iter) {
                VLOG(2) << "kNumPendulums: " << kNumPendulums
                        << ", kNumParticles: " << kNumParticles << " iter: " << iter;
                uniform_experiment->MakeIteration(AbstractExperiment::IterType::kSingle);
                stationary_experiment->MakeIteration(AbstractExperiment::IterType::kSingle);

                uniform_experiment->Score();
                stationary_experiment->Score();

		output_file.open(FLAGS_output_file.c_str(), std::ios::app);
                printer.PrintBody(output_file, *uniform_experiment);
                printer.PrintBody(output_file, *stationary_experiment);
		output_file.close();
            }
        }
    }
    return 0;
}
