#include <iostream>
#include <vector>

#include "grante/BeliefPropagation.h"
#include "grante/BruteForceExactInference.h"
#include "grante/Conditioning.h"
#include "grante/Factor.h"
#include "grante/FactorConditioningTable.h"
#include "grante/FactorGraph.h"
#include "grante/FactorGraphModel.h"
#include "grante/FactorGraphPartialObservation.h"
#include "grante/FactorType.h"

std::vector<double> encode_test_outcome(int outcome){
  if (outcome == 0){
    return {1,0};
  }
  else if (outcome == 1){
    return {0,1};
  }
  else {
    throw std::runtime_error("Invalid outcome");
  }
}

Grante::FactorGraph BakeFactorGraph(Grante::FactorGraphModel* model, int t0, int t1){
  Grante::FactorGraph fg(model, {2, 2, 2});
  auto ft = model->FindFactorType("test");
  // Add factors
  Grante::Factor* phi_1 = new Grante::Factor(ft, {0, 1}, encode_test_outcome(t0));
  Grante::Factor* phi_2 = new Grante::Factor(ft, {1, 2}, encode_test_outcome(t1));
  fg.AddFactor(phi_1);
  fg.AddFactor(phi_2);
  fg.ForwardMap();
  return fg;
}

int main() {
  Grante::FactorGraphModel model;

  // Create one simple pairwise factor type
  std::vector<double> w = {-1, -0, -0, -1, -0, -1, -1, -1};
  Grante::FactorType* factortype = new Grante::FactorType("test", {2, 2}, w);
  model.AddFactorType(factortype);

  //  Test it
  for (int i = 0; i < 2; i++){
    for (int j = 0; j < 2; j++){
      auto fg = BakeFactorGraph(&model, i, j); 
      Grante::BruteForceExactInference bfexact(&fg);
      std::vector<unsigned int> est_state;
      bfexact.MinimizeEnergy(est_state);
      std::cout << "Test:" << i << "," << j << std::endl;
      for (std::size_t i = 0; i < est_state.size(); ++i)
        std::cout << "f_" << i << " = " << est_state[i] << std::endl;
    }
  }
 
  return 0;
}