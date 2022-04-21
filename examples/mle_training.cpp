#include <iostream>
#include <vector>

#include "examples/example_graph.h"
#include "grante/BeliefPropagation.h"
#include "grante/Factor.h"
#include "grante/FactorGraph.h"
#include "grante/FactorGraphModel.h"
#include "grante/FactorType.h"
#include "grante/GibbsInference.h"
#include "grante/MaximumLikelihood.h"

void print_vect(const std::vector<double> &v) {
    for (const auto &x : v) std::cout << x << " ";
    std::cout << std::endl;
}

int main() {
    const std::vector<std::string> var_names = {"f1", "f2", "f3"};
    const std::vector<std::vector<int>> test_scopes = {{0, 1}, {1, 2}};
    auto graph = ExampleGraph(var_names, test_scopes);

    graph.fg->Print();
    std::cout << std::endl;

    // Sample from the factor graph
    Grante::GibbsInference ginf(graph.fg.get());
    ginf.SetSamplingParameters(1000, 10, 5000);
    std::vector<std::vector<unsigned int>> states;
    unsigned int sample_count = 5000;
    ginf.Sample(states, sample_count);

    // Print samples
    // for (const auto &sample: states){
    // 	for (const auto &x: sample)
    // 		std::cout << x << " ";
    // 	std::cout << std::endl;
    // }

    // Reset model parameters
    std::cout << "Resetting model parameters..." << std::endl;
    std::cout << "Before reset:" << std::endl;
    for (const auto phi : graph.fg->Factors()) {
        auto ft = phi->Type();
        print_vect(ft->Weights());
    }
    for (const auto phi : graph.fg->Factors()) {
        const auto name = phi->Type()->Name();
        auto ft = graph.model->FindFactorType(name);
        std::fill(ft->Weights().begin(), ft->Weights().end(), 0.0);
    }
    std::cout << "--------------------------------" << std::endl;
    std::cout << "After reset:" << std::endl;
    for (const auto phi : graph.fg->Factors()) {
        auto ft = phi->Type();
        print_vect(ft->Weights());
    }

    // Learn model parameters
    std::vector<Grante::ParameterEstimationMethod::labeled_instance_type> training_data;
    std::vector<Grante::InferenceMethod *> inference_methods;
    for (unsigned int si = 0; si < states.size(); ++si) {
        training_data.push_back(Grante::ParameterEstimationMethod::labeled_instance_type(
            graph.fg.get(), new Grante::FactorGraphObservation(states[si])));
        inference_methods.push_back(
            new Grante::BeliefPropagation(graph.fg.get(), Grante::BeliefPropagation::Sequential));
    }
    Grante::MaximumLikelihood mle(graph.model.get());
    mle.SetupTrainingData(training_data, inference_methods);
    mle.Train(1.0e-8);

    std::cout << "After reset:" << std::endl;
    for (const auto phi : graph.fg->Factors()) {
        auto ft = phi->Type();
        print_vect(ft->Weights());
    }
    return 0;
}