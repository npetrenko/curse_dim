#include <curse_dim/experiment.hpp>
#include <curse_dim/uniform_experiment.hpp>
#include <curse_dim/stationary_experiment.hpp>
#include <curse_dim/random_experiment.hpp>
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
        for (size_t kNumParticles = 128; kNumParticles <= 2048*2; kNumParticles += 512) {
            AbstractExperiment::Builder builder;
            builder.SetNumParticles(kNumParticles)
                .SetNumPendulums(kNumPendulums)
                .SetRandomDevice(std::mt19937(rd));
            std::unique_ptr uniform_experiment = UniformExperiment::Make(builder);
            std::unique_ptr stationary_experiment = StationaryExperiment::Make(builder);
            std::unique_ptr random_experiment = RandomExperiment::Make(builder);

            std::vector<AbstractExperiment*> active_experiments{uniform_experiment.get(),
                                                                random_experiment.get()};

            auto run_for_all = [&](auto func) {
                for (auto* exp : active_experiments) {
                    func(exp);
                }
            };

            auto make_iter = [](auto* exp) {
                exp->MakeIteration(AbstractExperiment::IterType::kSingle);
            };
            auto make_score = [](auto* exp) { exp->Score(); };
            auto make_print = [&](auto* exp) { printer.PrintBody(output_file, *exp); };

            for (size_t iter = 1; iter < 40; ++iter) {
                VLOG(2) << "kNumPendulums: " << kNumPendulums
                        << ", kNumParticles: " << kNumParticles << " iter: " << iter;
                run_for_all(make_iter);
                run_for_all(make_score);

                output_file.open(FLAGS_output_file.c_str(), std::ios::app);
                run_for_all(make_print);
                output_file.close();
            }
        }
    }
    return 0;
}
