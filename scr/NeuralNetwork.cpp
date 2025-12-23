#include <iostream>
#include <vector>
#include <random>
#include <omp.h>


#include "NeuralNetwork.h"
#include "EMath.h"

using namespace std;

/*
 * Constructor: Network Initialization & Memory Allocation.
 * --------------------------------------------------------
 * Purpose: Pre-allocates contiguous memory for layers, weights, and biases
 * based on the provided topology vector.
 *
 * Parameters:
 * - _structure: A vector defining the neuron count per layer (e.g., {784, 128, 10}).
 * * Memory Layout:
 * - Weight[i]: Matrix representing synapses from Layer[i] to Layer[i+1].
 * - Bias[i]: Offset values for neurons in Layer[i].
 * - Layer[i]: Buffers for activations (A-values).

2x2 matrik I define like
[
	[   ],
	[   ],
	[   ]
]

*/
NNET::nnet::nnet(vector<int> _structure)
{
	structure = _structure; //... this is needed lol fuck!
	if (_structure.size() < 2) throw std::runtime_error("ERROR: Invalid Model Structure!");

	//resize them to the structure size
	Weight.resize(_structure.size());
	Bias.resize(_structure.size());
	Layer.resize(_structure.size());
	Error.resize(_structure[_structure.size() - 1],0);

	//init them with zeros
#pragma omp parallel for
	for (int i = 0; i < _structure.size(); i++) 
	{
		if (i > 0) Bias[i].resize(_structure[i] , 0); //bias start at 1 cuz the index 0 is not used
		if (i + 1 < _structure.size()) Weight[i].resize(_structure[i], vector<float>(_structure[i+1], 0));
		Layer[i].resize(_structure[i]);
	}

}

/**
 * @brief Performs stochastic initialization of the network parameters using a Normal Distribution.
 * * This function populates the Weight and Bias tensors with values from a Gaussian distribution
 * (Mean = 0.0, StdDev = 1.0). This break in symmetry is critical for backpropagation to function
 * correctly during the initial training epochs.
 * * @param _nnet A reference to the nnet object to be initialized. Pass-by-reference is utilized
 * to prevent high-overhead memory copies of large weight matrices.
 * * @performance_note Utilizes OpenMP multi-threading (#pragma omp parallel for) to parallelize
 * the nested loops. This optimizes initialization speed for deep architectures by distributing
 * the Gaussian generation across available CPU cores.
 * * @return int Returns 0 upon successful allocation and initialization.
 */
int NNET::Random_Initialise(nnet& _nnet)
{
	std::default_random_engine gen;
	std::normal_distribution<float> normal_distribution(0.0, 1.0);

#pragma omp parallel for
	for (auto& l : _nnet.Bias)
	{
		for (auto& b : l)
		{
			b = normal_distribution(gen);
		}
	}

#pragma omp parallel for
	for (auto& l : _nnet.Weight)
	{
		for (auto& r : l)
		{
			for (auto& e : r)
			{
				e = normal_distribution(gen);
			}
		}
	}

	return 0;

}

/*
* This is the feedpropagation of the model 
* 
*/
int NNET::Feed_Propagation(nnet& _nnet)
{
	for (int _layer = 1; _layer < _nnet.Layer.size();_layer++)
	{
		for (int _e = 0; _e < _nnet.Layer[_layer].size(); _e++)
		{
			_nnet.Layer[_layer][_e] = _nnet.Bias[_layer][_e];
			for (int _pl = 0; _pl < _nnet.Layer[_layer - 1].size(); _pl++)
			{
				_nnet.Layer[_layer][_e] += _nnet.Weight[_layer - 1][_pl][_e] * _nnet.Layer[_layer - 1][_pl];
			}
			_nnet.Layer[_layer][_e] = 1.0f / (1.0f + expf(-_nnet.Layer[_layer][_e]));
		}

	}

	return 0;
}

/*
* This calculate error of model
*/
float NNET::Calculate_Error(nnet& _nnet, std::vector<float>& result)
{
	float total_cost = 0;
	if (_nnet.Layer[_nnet.Layer.size() - 1].size() != result.size()) std::abort;
	for (int i = 0; i < _nnet.Error.size(); i++)
	{
		float diff = (result[i] - _nnet.Layer[_nnet.Layer.size() - 1][i]);
		_nnet.Error[i] = 0.5f * diff * diff;
		total_cost += _nnet.Error[i];
	}
	return total_cost;
}