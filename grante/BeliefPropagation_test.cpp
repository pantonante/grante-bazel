#include "grante/BeliefPropagation.h"

#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "grante/Factor.h"
#include "grante/FactorGraph.h"
#include "grante/FactorGraphModel.h"
#include "grante/FactorType.h"
#include "grante/GibbsInference.h"
#include "gtest/gtest.h"

TEST(BeliefPropagation, EnergyMinimization) {
    // std::uniform_int_distribution<int> uniform_dist(0, 1);
    std::uniform_real_distribution<double> randu(0, 1);
    for (unsigned int random_start = 0; random_start < 100; ++random_start) {
        std::default_random_engine e1(random_start);

        Grante::FactorGraphModel model;

        // Create one simple parametrized, data-independent pairwise factor type
        std::vector<unsigned int> card;
        card.push_back(2);
        std::vector<double> w;

        Grante::FactorType* factortype_u = new Grante::FactorType("unary", card, w);
        model.AddFactorType(factortype_u);

        card.push_back(2);
        Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
        model.AddFactorType(factortype);

        // Create a N-by-N grid-structured model
        unsigned int N = 2;

        // Create a factor graph for the model
        std::vector<unsigned int> vc(N * N, 2);
        Grante::FactorGraph fg(&model, vc);

        // Add unary factors
        Grante::FactorType* pt_u = model.FindFactorType("unary");
        std::vector<double> data_u(2);
        std::vector<unsigned int> var_index_u(1);
        for (unsigned int y = 0; y < N; ++y) {
            for (unsigned int x = 0; x < N; ++x) {
                var_index_u[0] = y * N + x;
                for (unsigned int di = 0; di < data_u.size(); ++di) data_u[di] = randu(e1);
                Grante::Factor* fac = new Grante::Factor(pt_u, var_index_u, data_u);
                fg.AddFactor(fac);
            }
        }

        // Add pairwise factors
        Grante::FactorType* pt = model.FindFactorType("pairwise");
        std::vector<double> data(4);
        std::vector<unsigned int> var_index(2);
        for (unsigned int y = 0; y < N; ++y) {
            for (unsigned int x = 1; x < N; ++x) {
                // Horizontal edge
                var_index[0] = y * N + x - 1;
                var_index[1] = y * N + x;

                for (unsigned int di = 0; di < data.size(); ++di) data[di] = randu(e1);
                Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
                fg.AddFactor(fac);
            }
        }
        for (unsigned int y = 1; y < N; ++y) {
            unsigned int x = 0;

            // Vertical edge
            var_index[0] = (y - 1) * N + x;
            var_index[1] = y * N + x;

            for (unsigned int di = 0; di < data.size(); ++di) data[di] = randu(e1);
            Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
            fg.AddFactor(fac);
        }

        // fg is now a N-by-N tree-structured factor graph.  Minimize energy.
        fg.ForwardMap();
        Grante::BeliefPropagation bpinf(&fg);
        std::vector<unsigned int> bpinf_state;
        double bpinf_energy = bpinf.MinimizeEnergy(bpinf_state);

        // Find minimum energy state by exhaustive search
        std::vector<unsigned int> test_var(4);
        std::vector<unsigned int> min_var(4);
        double min_var_energy = std::numeric_limits<double>::infinity();
        for (unsigned int v0 = 0; v0 < 2; ++v0) {
            test_var[0] = v0;
            for (unsigned int v1 = 0; v1 < 2; ++v1) {
                test_var[1] = v1;
                for (unsigned int v2 = 0; v2 < 2; ++v2) {
                    test_var[2] = v2;
                    for (unsigned int v3 = 0; v3 < 2; ++v3) {
                        test_var[3] = v3;

                        double orig_e = fg.EvaluateEnergy(test_var);
                        if (orig_e < min_var_energy) {
                            min_var = test_var;
                            min_var_energy = orig_e;
                        }
                    }
                }
            }
        }
        ASSERT_THAT(bpinf_energy, testing::DoubleNear(min_var_energy, 1.0e-6));
    }
}

TEST(BeliefPropagation, Simple) {
    Grante::FactorGraphModel model;

    // Create one simple pairwise factor type
    std::vector<unsigned int> card;
    card.push_back(2);
    card.push_back(2);
    std::vector<double> w;
    w.push_back(0.0);
    w.push_back(0.3);
    w.push_back(0.2);
    w.push_back(0.0);
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

    // Test inference results
    Grante::BeliefPropagation bpinf(&fg, Grante::BeliefPropagation::Sequential);
    bpinf.PerformInference();

    double log_z = bpinf.LogPartitionFunction();
    std::cout << "log_z " << log_z << std::endl;
    ASSERT_THAT(log_z, testing::DoubleNear(0.4836311, 1.0e-3));

    const std::vector<double>& m_fac0 = bpinf.Marginal(0);
    ASSERT_THAT(m_fac0.size(), testing::Eq(4));
    ASSERT_THAT(m_fac0[0], testing::DoubleNear(0.4132795, 1e-3));
    ASSERT_THAT(m_fac0[1], testing::DoubleNear(0.1680269, 1e-3));
    ASSERT_THAT(m_fac0[2], testing::DoubleNear(0.2506666, 1e-3));
    ASSERT_THAT(m_fac0[3], testing::DoubleNear(0.1680269, 1e-3));

    const std::vector<double>& m_fac1 = bpinf.Marginal(1);
    ASSERT_THAT(m_fac1.size(), testing::Eq(2));
    ASSERT_THAT(m_fac1[0], testing::DoubleNear(0.6639461, 1e-3));
    ASSERT_THAT(m_fac1[1], testing::DoubleNear(0.3360538, 1e-3));

    const std::vector<double>& m_fac2 = bpinf.Marginal(2);
    ASSERT_THAT(m_fac2.size(), testing::Eq(2));
    ASSERT_THAT(m_fac2[0], testing::DoubleNear(0.5813064, 1e-3));
    ASSERT_THAT(m_fac2[1], testing::DoubleNear(0.4186935, 1e-3));

    // Check Gibbs sampler
    Grante::GibbsInference ginf(&fg);
    ginf.SetSamplingParameters(10000, 10, 100000);
    ginf.PerformInference();
    std::cout << "Gibbs log_z " << ginf.LogPartitionFunction() << std::endl;

    const std::vector<double>& g_fac0 = ginf.Marginal(0);
    ASSERT_THAT(g_fac0.size(), testing::Eq(4));
    ASSERT_THAT(g_fac0[0], testing::DoubleNear(0.4132795, 1e-2));
    ASSERT_THAT(g_fac0[1], testing::DoubleNear(0.1680269, 1e-2));
    ASSERT_THAT(g_fac0[2], testing::DoubleNear(0.2506666, 1e-2));
    ASSERT_THAT(g_fac0[3], testing::DoubleNear(0.1680269, 1e-2));

    const std::vector<double>& g_fac1 = ginf.Marginal(1);
    ASSERT_THAT(g_fac1.size(), testing::Eq(2));
    ASSERT_THAT(g_fac1[0], testing::DoubleNear(0.6639461, 1e-2));
    ASSERT_THAT(g_fac1[1], testing::DoubleNear(0.3360538, 1e-2));

    const std::vector<double>& g_fac2 = ginf.Marginal(2);
    ASSERT_THAT(g_fac2.size(), testing::Eq(2));
    ASSERT_THAT(g_fac2[0], testing::DoubleNear(0.5813064, 1e-2));
    ASSERT_THAT(g_fac2[1], testing::DoubleNear(0.4186935, 1e-2));
}