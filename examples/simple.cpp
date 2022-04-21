#include <cmath>
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

unsigned int LinearIndexToVariableState(size_t ei, size_t var_index) { return (ei / (1u << var_index)) % 2; }

int main() {
    Grante::FactorGraphModel model;

    // Create one simple pairwise factor type
    const unsigned int num_failure_modes = 3;
    const unsigned int num_tests = 2;
    const unsigned int num_vars = num_failure_modes + num_tests;
    const std::vector<unsigned int> var_card(num_vars, 2);
    std::vector<double> w = {-1, -0, -0, -1, -0, -1, -1, -1};
    Grante::FactorType* test_ft = new Grante::FactorType("test", {2, 2, 2}, w);
    //  std::vector<double> max_card_w = {1, 1, 1, 1, 1, 1, 1, 0};
    std::vector<double> max_card_w = {-1, -1, -1, -0, -1, -0, -0, -0};
    Grante::FactorType* card_ft = new Grante::FactorType("MaxCardinality", {2, 2, 2}, max_card_w);
    model.AddFactorType(test_ft);
    model.AddFactorType(card_ft);

    // Create a factor graph from the model: 3 binary variables (f1,f2,t)
    Grante::FactorGraph fg(&model, var_card);

    // Add factors
    std::vector<double> data;
    Grante::Factor* phi_1 = new Grante::Factor(test_ft, {3, 0, 1}, data);
    Grante::Factor* phi_2 = new Grante::Factor(test_ft, {4, 1, 2}, data);
    Grante::Factor* phi_3 = new Grante::Factor(card_ft, {0, 1, 2}, data);
    fg.AddFactor(phi_1);
    fg.AddFactor(phi_2);
    fg.AddFactor(phi_3);
    fg.ForwardMap();
    fg.Print();

    for (unsigned int ei = 0; ei < (1u << num_vars); ++ei) {
        std::cout << "[" << ei << "] : ";
        std::vector<unsigned int> state;
        for (unsigned int vi = 0; vi < num_vars; ++vi) {
            auto s = LinearIndexToVariableState(ei, vi);
            std::cout << s << " ";
            state.push_back(s);
        }
        std::cout << "-> " << fg.EvaluateEnergy(state) << std::endl;
    }

    // Conditioning
    std::cout << "Conditioning ...." << std::endl;
    std::vector<unsigned int> cond_var_set = {3,4};
    std::vector<unsigned int> cond_var_state(2,1);
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 2; ++j) {
            std::cout << "TEST: t1=" << i << ", t2=" << j << std::endl;
            cond_var_state[0] = i;
            cond_var_state[1] = j;
            Grante::FactorGraphPartialObservation partial_obs(cond_var_set, cond_var_state);
            std::vector<unsigned int> var_new_to_orig; //var_new_to_orig[new_factor_index] = original_factor_index
            Grante::FactorConditioningTable conditioning_table;
            Grante::FactorGraph* fg_cond =
                Grante::Conditioning::ConditionFactorGraph(&conditioning_table, &fg, &partial_obs, var_new_to_orig);
            fg_cond->ForwardMap();

            Grante::BruteForceExactInference inference(fg_cond);
            std::vector<unsigned int> est_state;
            inference.MinimizeEnergy(est_state);
            std::cout << "Result: "<<std::endl;
            for (std::size_t i = 0; i < est_state.size(); ++i)
                std::cout << "var_" << i << " = " << est_state[i] << std::endl;
        }
    }
    return 0;
}