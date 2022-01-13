#include <vector>
#include <iostream>

#include "grante/FactorGraph.h"
#include "grante/FactorType.h"
#include "grante/Factor.h"
#include "grante/FactorGraphModel.h"
#include "grante/BeliefPropagation.h"

int main() {
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	const std::vector<unsigned int> card = {2,2};
	std::vector<double> w = {1.0, 0.2, -0.2, 1.0};
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a factor graph from the model: 3 binary variables
	std::vector<unsigned int> vc = {2,2,2};
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
	fg.ForwardMap();
	fg.Print();

	Grante::BeliefPropagation bpinf(&fg, Grante::BeliefPropagation::Sequential);
	bpinf.PerformInference();
	for (unsigned int vi = 0; vi < fg.Cardinalities().size(); ++vi) {
		std::cout << "Variable " << vi << ": " << std::endl;
		auto marginal = bpinf.Marginal(vi); 
		for (const auto &val: marginal)
			std::cout << val <<std::endl;
	}

  return 0;
}