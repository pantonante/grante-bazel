#include <vector>
#include <iostream>

#include "grante/FactorGraph.h"
#include "grante/FactorType.h"
#include "grante/Factor.h"
#include "grante/FactorGraphModel.h"
#include "grante/BeliefPropagation.h"
#include "grante/GibbsInference.h"
#include "grante/MaximumLikelihood.h"

void print_vect(const std::vector<double> &v)
{
	for (const auto &x : v)
		std::cout << x << " ";
	std::cout << std::endl;
}

int main()
{
	const std::vector<std::string> var_names = {"f1", "f2", "f3"};
	const std::vector<std::vector<int>> test_scopes = {{0, 1}, {1, 2}};

	// Create diagnostic factor graph
	Grante::FactorGraphModel model;
	// Initialize FactorType for Priors
	for (unsigned int vi = 0; vi < var_names.size(); ++vi)
	{
		std::vector<double> w = {0.8, 0.2};
		Grante::FactorType *factortype = new Grante::FactorType(
			"pi_" + std::to_string(vi), {2}, w);
		model.AddFactorType(factortype);
	}
	// Initialize FactorType for Tests
	for (unsigned int ti = 0; ti < test_scopes.size(); ++ti)
	{
		const auto scope = test_scopes[ti];
		const std::vector<unsigned int> card(scope.size() + 1, 2);
		std::vector<double> w = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0};
		Grante::FactorType *factortype = new Grante::FactorType(
			"t_" + std::to_string(ti), card, w);
		model.AddFactorType(factortype);
	}
	std::vector<double> data;
	std::vector<unsigned int> var_cards(var_names.size() + test_scopes.size(), 2);
	Grante::FactorGraph fg(&model, var_cards);
	// Add priors
	for (unsigned int vi = 0; vi < var_names.size(); ++vi)
	{
		auto *ft = model.FindFactorType("pi_" + std::to_string(vi));
		fg.AddFactor(new Grante::Factor(ft, {vi}, data));
	}
	// Add tests
	for (unsigned int ti = 0; ti < test_scopes.size(); ++ti)
	{
		const auto scope = test_scopes[ti];
		Grante::FactorType *ft = model.FindFactorType("t_" + std::to_string(ti));
		std::vector<unsigned int> var_index(scope.size() + 1);
		for (unsigned int i = 0; i < scope.size(); ++i)
			var_index[i] = scope[i];
		var_index[scope.size()] = var_names.size() + ti;
		fg.AddFactor(new Grante::Factor(ft, var_index, data));
	}
	fg.ForwardMap();

	fg.Print();
	std::cout << std::endl;

	// Sample from the factor graph
	Grante::GibbsInference ginf(&fg);
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
	for (const auto phi : fg.Factors())
	{
		auto ft = phi->Type();
		print_vect(ft->Weights());
	}
	for (const auto phi : fg.Factors())
	{
		const auto name = phi->Type()->Name();
		auto ft = model.FindFactorType(name);
		std::fill(ft->Weights().begin(), ft->Weights().end(), 0.0);
	}
	std::cout << "--------------------------------" << std::endl;
	std::cout << "After reset:" << std::endl;
	for (const auto phi : fg.Factors())
	{
		auto ft = phi->Type();
		print_vect(ft->Weights());
	}

	// Learn model parameters
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod *> inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si)
	{
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg, new Grante::FactorGraphObservation(states[si])));

		inference_methods.push_back(
			new Grante::BeliefPropagation(&fg, Grante::BeliefPropagation::Sequential));
	}
	Grante::MaximumLikelihood mle(&model);
	mle.SetupTrainingData(training_data, inference_methods);
	mle.Train(1.0e-8);

	std::cout << "After reset:" << std::endl;
	for (const auto phi : fg.Factors())
	{
		auto ft = phi->Type();
		print_vect(ft->Weights());
	}
	return 0;
}