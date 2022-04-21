
#include "grante/FactorGraph.h"

#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "grante/BeliefPropagation.h"
#include "grante/Conditioning.h"
#include "grante/Factor.h"
#include "grante/FactorConditioningTable.h"
#include "grante/FactorGraphModel.h"
#include "grante/FactorGraphPartialObservation.h"
#include "grante/FactorType.h"
#include "grante/GibbsInference.h"
#include "grante/TreeInference.h"
#include "gtest/gtest.h"

TEST(FactorGraph, Simple) {
    Grante::FactorGraphModel model;

    // Create one simple pairwise factor type
    std::vector<unsigned int> card;
    card.push_back(2);
    card.push_back(2);
    std::vector<double> w;
    w.push_back(1.0);
    w.push_back(0.2);
    w.push_back(-0.2);
    w.push_back(1.0);
    Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
    model.AddFactorType(factortype);

    // Create a factor graph from the model: 3 binary variables
    std::vector<unsigned int> vc;
    vc.push_back(2);
    vc.push_back(2);
    vc.push_back(2);
    Grante::FactorGraph fg(&model, vc);

    // Add factors
    const Grante::FactorType* pt = model.FindFactorType("pairwise");
    std::vector<double> data;
    std::vector<unsigned int> var_index(2);
    var_index[0] = 0;
    var_index[1] = 1;
    Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
    fg.AddFactor(fac1);
    var_index[0] = 1;
    var_index[1] = 2;
    Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
    fg.AddFactor(fac2);
    var_index[0] = 0;
    var_index[1] = 2;
    fg.AddFactor(new Grante::Factor(pt, var_index, data));

    // Compute the forward map
    fg.ForwardMap();

    // Compute the backward map for some marginal vector
    std::vector<double> marg(4);
    marg[0] = 0.25;
    marg[1] = 0.4;
    marg[2] = 0.1;
    marg[3] = 0.25;
    std::vector<double> pargrad(4, 0.0);
    fac1->BackwardMap(marg, pargrad);
    for (unsigned int pi = 0; pi < pargrad.size(); ++pi) {
        ASSERT_THAT(marg[pi], testing::DoubleNear(pargrad[pi], 1.0e-7));
    }
}
