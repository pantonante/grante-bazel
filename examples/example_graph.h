#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "grante/Factor.h"
#include "grante/FactorGraph.h"
#include "grante/FactorGraphModel.h"
#include "grante/FactorType.h"

double ToEnergy(double w) {
  const auto epsilon = 1.0e-8;
  if (w <= epsilon)
    return -std::log(epsilon);
  else
    return -std::log(w);
}

struct ModeledFactorGraph {
  std::shared_ptr<Grante::FactorGraphModel> model;
  std::shared_ptr<Grante::FactorGraph> fg;
  std::vector<std::string> var_names;
  std::vector<std::vector<int>> test_scopes;
};

ModeledFactorGraph ExampleGraph(const std::vector<std::string> &var_names,
                                const std::vector<std::vector<int>> &test_scopes) {
  // --- Create the factor graph model ---
  std::shared_ptr<Grante::FactorGraphModel> model =
      std::make_shared<Grante::FactorGraphModel>();
  // Initialize FactorType for Priors
  for (unsigned int vi = 0; vi < var_names.size(); ++vi) {
    std::vector<double> w = {0.8, 0.2};
    std::transform(w.begin(), w.end(), w.begin(), &ToEnergy);
    Grante::FactorType *factortype =
        new Grante::FactorType("pi_" + std::to_string(vi), {2}, w);
    model->AddFactorType(factortype);
  }
  // Initialize FactorType for Tests
  for (unsigned int ti = 0; ti < test_scopes.size(); ++ti) {
    const auto scope = test_scopes[ti];
    const std::vector<unsigned int> card(scope.size() + 1, 2);
    std::vector<double> w = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0};
    std::transform(w.begin(), w.end(), w.begin(), &ToEnergy);
    Grante::FactorType *factortype =
        new Grante::FactorType("t_" + std::to_string(ti), card, w);
    model->AddFactorType(factortype);
  }
  // --- Create factor graph ---
  std::vector<double> data;
  std::vector<unsigned int> var_cards(var_names.size() + test_scopes.size(), 2);
  std::shared_ptr<Grante::FactorGraph> fg =
      std::make_shared<Grante::FactorGraph>(model.get(), var_cards);
  // Add priors
  for (unsigned int vi = 0; vi < var_names.size(); ++vi) {
    auto *ft = model->FindFactorType("pi_" + std::to_string(vi));
    fg->AddFactor(new Grante::Factor(ft, {vi}, data));
  }
  // Add tests
  for (unsigned int ti = 0; ti < test_scopes.size(); ++ti) {
    const auto scope = test_scopes[ti];
    Grante::FactorType *ft = model->FindFactorType("t_" + std::to_string(ti));
    std::vector<unsigned int> var_index(scope.size() + 1);
    for (unsigned int i = 0; i < scope.size(); ++i) var_index[i] = scope[i];
    var_index[scope.size()] = var_names.size() + ti;
    fg->AddFactor(new Grante::Factor(ft, var_index, data));
  }
  fg->ForwardMap();
  return ModeledFactorGraph{model, fg, var_names, test_scopes};
}

ModeledFactorGraph InstantiateModel(const std::vector<std::string> &var_names,
                                    const std::vector<std::vector<int>> &test_scopes,
                                    bool randomize = false) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  std::shared_ptr<Grante::FactorGraphModel> model =
      std::make_shared<Grante::FactorGraphModel>();
  // Initialize FactorType for Priors
  for (unsigned int vi = 0; vi < var_names.size(); ++vi) {
    std::vector<double> w = {0.8, 0.2};
    if (randomize) {
      for (auto &v : w) v = dis(gen);
    }
    std::transform(w.begin(), w.end(), w.begin(), &ToEnergy);
    Grante::FactorType *factortype =
        new Grante::FactorType("pi_" + std::to_string(vi), {2}, w);
    model->AddFactorType(factortype);
  }
  // Initialize FactorType for Tests
  for (unsigned int ti = 0; ti < test_scopes.size(); ++ti) {
    const auto scope = test_scopes[ti];
    const std::vector<unsigned int> card(scope.size(), 2);
    std::vector<double> w = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0};
    if (randomize) {
      for (auto &v : w) v = dis(gen);
    }
    std::transform(w.begin(), w.end(), w.begin(), &ToEnergy);
    Grante::FactorType *factortype =
        new Grante::FactorType("t_" + std::to_string(ti), card, w);
    model->AddFactorType(factortype);
  }
  return ModeledFactorGraph{model, nullptr, var_names, test_scopes};
}

std::shared_ptr<Grante::FactorGraph> InstantiateFactorGraphFromData(
    ModeledFactorGraph &graph, const std::vector<unsigned int> &data) {
  const auto num_vars = graph.var_names.size();
  std::vector<unsigned int> var_cards(num_vars, 2);
  std::shared_ptr<Grante::FactorGraph> fg =
      std::make_shared<Grante::FactorGraph>(graph.model.get(), var_cards);
  // Add priors
  for (unsigned int vi = 0; vi < graph.var_names.size(); ++vi) {
    auto *ft = graph.model->FindFactorType("pi_" + std::to_string(vi));
    std::vector<double> factor_data;
    fg->AddFactor(new Grante::Factor(ft, {vi}, factor_data));
  }
  // Add tests
  for (unsigned int ti = 0; ti < graph.test_scopes.size(); ++ti) {
    const auto scope = graph.test_scopes[ti];
    Grante::FactorType *ft = graph.model->FindFactorType("t_" + std::to_string(ti));
    std::vector<unsigned int> var_index(scope.size());
    for (unsigned int i = 0; i < scope.size(); ++i) var_index[i] = scope[i];
    std::vector<double> factor_data = {0.0, 0.0};
    factor_data[data[ti]] = 1.0;
    fg->AddFactor(new Grante::Factor(ft, var_index, factor_data));
  }
  fg->ForwardMap();
  return fg;
}