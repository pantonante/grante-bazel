
#include "grante/Conditioning.h"

#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "grante/BeliefPropagation.h"
#include "grante/Factor.h"
#include "grante/FactorConditioningTable.h"
#include "grante/FactorGraph.h"
#include "grante/FactorGraphModel.h"
#include "grante/FactorGraphPartialObservation.h"
#include "grante/FactorType.h"
#include "grante/TreeInference.h"
#include "gtest/gtest.h"

TEST(Conditioning, Simple) {
    Grante::FactorGraphModel model;

    // Create one simple pairwise factor type
    std::vector<unsigned int> card;
    card.push_back(2);
    card.push_back(2);
    std::vector<double> w;
    w.push_back(0.0);  // (0,0)
    w.push_back(0.3);  // (1,0)
    w.push_back(0.2);  // (0,1)
    w.push_back(0.0);  // (1,1)
    Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
    model.AddFactorType(factortype);

    std::vector<unsigned int> card1;
    card1.push_back(2);
    std::vector<double> w1;
    w1.push_back(0.1);
    w1.push_back(0.7);
    Grante::FactorType* factortype1a = new Grante::FactorType("unary1", card1, w1);
    model.AddFactorType(factortype1a);

    w1[0] = 0.3;
    w1[1] = 0.6;
    Grante::FactorType* factortype1b = new Grante::FactorType("unary2", card1, w1);
    model.AddFactorType(factortype1b);

    // Create a factor graph from the model: 2 binary variables
    std::vector<unsigned int> vc;
    vc.push_back(2);
    vc.push_back(2);
    Grante::FactorGraph fg(&model, vc);

    // Add factors
    const Grante::FactorType* pt2 = model.FindFactorType("pairwise");
    const Grante::FactorType* pt1a = model.FindFactorType("unary1");
    const Grante::FactorType* pt1b = model.FindFactorType("unary2");
    std::vector<double> data;
    std::vector<unsigned int> var_index(2);
    var_index[0] = 0;
    var_index[1] = 1;
    Grante::Factor* fac1 = new Grante::Factor(pt2, var_index, data);
    fg.AddFactor(fac1);

    std::vector<unsigned int> var_index1(1);
    var_index1[0] = 0;
    Grante::Factor* fac1a = new Grante::Factor(pt1a, var_index1, data);
    fg.AddFactor(fac1a);
    var_index1[0] = 1;
    Grante::Factor* fac1b = new Grante::Factor(pt1b, var_index1, data);
    fg.AddFactor(fac1b);

    // Compute the forward map
    fg.ForwardMap();

    std::vector<unsigned int> state(2);
    state[0] = 0;
    state[1] = 0;
    ASSERT_THAT(fg.EvaluateEnergy(state), testing::DoubleNear(0.4, 1.0e-5));
    state[0] = 0;
    state[1] = 1;
    ASSERT_THAT(fg.EvaluateEnergy(state), testing::DoubleNear(0.9, 1.0e-5));
    state[0] = 1;
    state[1] = 0;
    ASSERT_THAT(fg.EvaluateEnergy(state), testing::DoubleNear(1.3, 1.0e-5));
    state[0] = 1;
    state[1] = 1;
    ASSERT_THAT(fg.EvaluateEnergy(state), testing::DoubleNear(1.3, 1.0e-5));

    // Condition the factor graph
    Grante::FactorConditioningTable ftab;
    std::vector<unsigned int> cond_var_set;
    std::vector<unsigned int> cond_var_state;
    // Condition on state[1] = 0
    cond_var_set.push_back(1);
    cond_var_state.push_back(0);
    std::vector<unsigned int> var_new_to_orig;

    // Test conditioned energies
    Grante::FactorGraphPartialObservation pobs(cond_var_set, cond_var_state);
    Grante::FactorGraph* fg_cond = Grante::Conditioning::ConditionFactorGraph(&ftab, &fg, &pobs, var_new_to_orig);
    fg_cond->ForwardMap();
    // fg_cond->Print();

    // Perform inference
    Grante::TreeInference tinf_uncond(&fg);
    Grante::TreeInference tinf_cond(fg_cond);
    tinf_uncond.PerformInference();
    tinf_cond.PerformInference();
    double logZ_uncond = tinf_uncond.LogPartitionFunction();
    ASSERT_THAT(logZ_uncond, testing::DoubleNear(0.48363, 1.0e-5));
    double logZ_cond = tinf_cond.LogPartitionFunction();
    ASSERT_THAT(logZ_cond, testing::DoubleNear(0.24115, 1.0e-5));

    std::vector<unsigned int> state_cond(1);
    state_cond[0] = 0;
    ASSERT_THAT(std::exp(-fg_cond->EvaluateEnergy(state_cond)) / std::exp(logZ_cond),
                testing::DoubleNear(0.71095, 1.0e-5));
    state_cond[0] = 1;
    ASSERT_THAT(std::exp(-fg_cond->EvaluateEnergy(state_cond)) / std::exp(logZ_cond),
                testing::DoubleNear(0.28905, 1.0e-5));

    state_cond[0] = 0;
    double energy_0 = fg_cond->EvaluateEnergy(state_cond);
    ASSERT_THAT(energy_0, testing::DoubleNear(0.4 - 0.3, 1.0e-5));

    state_cond[0] = 1;
    double energy_1 = fg_cond->EvaluateEnergy(state_cond);
    ASSERT_THAT(energy_1, testing::DoubleNear(1.3 - 0.3, 1.0e-5));

    // TODO: compare probabilities, not energies.  Energies change with
    // constant bias, probabilities should not change
    delete fg_cond;
}
