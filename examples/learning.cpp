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
#include "grante/MaximumPseudolikelihood.h"
#include "grante/NaivePiecewiseTraining.h"
#include "grante/StructuredSVM.h"
#include "grante/NormalPrior.h"

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
  ginf.SetSamplingParameters(1000, 10, 1000);
  std::vector<std::vector<unsigned int>> states;
  unsigned int sample_count = 1000;
  ginf.Sample(states, sample_count);

  // Reset model parameters
  auto model = InstantiateModel(var_names, test_scopes, true);
  
  std::cout << "Before training:" << std::endl;
  for (const auto ft : model.model->FactorTypes()) {
    print_vect(ft->Weights());
  }

  // Learn model parameters
  std::vector<Grante::ParameterEstimationMethod::labeled_instance_type> training_data;
  std::vector<Grante::InferenceMethod *> inference_methods;
  std::vector<std::shared_ptr<Grante::FactorGraph>> graphs;
  // std::vector<std::StructuredHammingLoss> losses;
  for (unsigned int si = 0; si < states.size(); ++si) {
    std::vector<unsigned int> data;
    for (unsigned int i = 0; i < test_scopes.size(); ++i) {
      data.push_back(states[si][var_names.size() + i]);
    }
    auto fg = InstantiateFactorGraphFromData(model, data);
    std::vector<unsigned int> obs;
    for (unsigned int i = 0; i < var_names.size(); ++i) {
      obs.push_back(states[si][i]);
    }
    auto fg_obs = new Grante::FactorGraphObservation(obs);
    // losses.push_back(std::StructuredHammingLoss(fg_obs));
    training_data.push_back(
        Grante::ParameterEstimationMethod::labeled_instance_type(fg.get(), fg_obs));
    inference_methods.push_back(
        new Grante::BeliefPropagation(fg.get(), Grante::BeliefPropagation::Sequential));
    graphs.push_back(fg);
  }
  // Grante::NaivePiecewiseTraining trainer(model.model.get());
  // Grante::MaximumLikelihood trainer(model.model.get());
  // Grante::MaximumPseudolikelihood trainer(model.model.get());
  // trainer.SetupTrainingData(training_data, inference_methods);
  
  Grante::StructuredSVM trainer(model.model.get(), 0.01, "bmrm");
  // for (const auto ft : model.model->FactorTypes()) {
  //   trainer.AddPrior(ft->Name(), new Grante::NormalPrior(10.0, ft->Weights().size()));
  // }
  trainer.SetupTrainingData(training_data, inference_methods);
  
  trainer.Train(1.0e-8, 10);

  std::cout << "Ground Truth:" << std::endl;
  for (const auto ft : graph.model->FactorTypes()) {
    print_vect(ft->Weights());
  }
  std::cout << "Learned:" << std::endl;
  for (const auto ft : model.model->FactorTypes()) {
    print_vect(ft->Weights());
  }
  return 0;
}