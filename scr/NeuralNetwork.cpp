#include <iostream>
#include <vector>
#include <random>
#include <omp.h>


#include "NeuralNetwork.h"
#include "EMath.h"
#include "Run.h"

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
 * @brief Performs a complete structural reconfiguration of the Neural Network manifold.
 * * This method executes a full teardown and re-allocation of all internal tensor buffers,
 * including weights, biases, layer activations, and gradient accumulation buffers.
 * It is primarily used during model restoration (FMANAGER) or when switching
 * architectures (e.g., from XOR to MNIST) to ensure memory integrity and zero
 * cross-contamination between sessions.
 * * @param _structure A vector defining the neuron count for each layer (e.g., {784, 128, 10}).
 * @throws std::runtime_error If the topology contains fewer than two layers.
 * * @note Computational Complexity: O(N) where N is the total number of weights.
 * @note Optimization: Utilizes OpenMP parallelization for accelerated memory mapping
 * on high-dimensional layers.
 */
void NNET::nnet::reinit(vector<int> _structure)
{
	//structure of Neural Network, Input Weight Bias Layer
	Weight.clear();
	Bias.clear();
	Layer.clear();
	structure.clear();
	Error.clear();
	Delta.clear();
	Weight_g.clear();
	Bias_g.clear();

	structure = _structure; //... this is needed lol fuck!
	if (_structure.size() < 2) throw std::runtime_error("ERROR: Invalid Model Structure!");

	//resize them to the structure size
	Weight.resize(_structure.size());
	Bias.resize(_structure.size());
	Layer.resize(_structure.size());
	Error.resize(_structure[_structure.size() - 1], 0);

	//init them with zeros
#pragma omp parallel for
	for (int i = 0; i < _structure.size(); i++)
	{
		if (i > 0) Bias[i].resize(_structure[i], 0); //bias start at 1 cuz the index 0 is not used
		if (i + 1 < _structure.size()) Weight[i].resize(_structure[i], vector<float>(_structure[i + 1], 0));
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

/**
 * @brief Performs a Forward Propagation pass through the network.
 * * This kernel iterates through each layer, calculating neuron activations by
 * performing a weighted sum of the previous layer's output plus a bias term,
 * followed by a Sigmoid activation.
 * * @note Optimized for cache locality by traversing the previous layer's
 * activations in a linear fashion.
 * * @param _nnet Reference to the network structure containing weights and layers.
 * @return int Returns 0 upon successful execution.
 */
int NNET::Feed_Propagation(nnet& _nnet)
{
	for (int _layer = 1; _layer < _nnet.Layer.size();_layer++)
	{
#pragma omp parallel for
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

/**
 * @brief Calculates the Mean Squared Error (MSE) for the current output layer.
 * * This function performs a point-wise comparison between the network's terminal
 * activations and the target labels. It uses the 0.5 * (target - output)^2
 * convention to simplify the derivative calculation during backpropagation.
 * * @param _nnet Reference to the neural network structure.
 * @param result The ground truth (target) vector for the current sample.
 * @return float The aggregate cost (loss) for the output layer.
 */
float NNET::Calculate_Error(nnet& _nnet, std::vector<float>& result)
{
	float total_cost = 0;
	if (_nnet.Layer[_nnet.Layer.size() - 1].size() != result.size()) std::abort;
#pragma omp parallel for
	for (int i = 0; i < _nnet.Error.size(); i++)
	{
		float diff = (result[i] - _nnet.Layer[_nnet.Layer.size() - 1][i]);
		_nnet.Error[i] = 0.5f * diff * diff;
		total_cost += _nnet.Error[i];
	}
	return total_cost;
}

/*
* this init the weight and bias accumulation
*/
float NNET::Init_Gradient_Accumulation(nnet& _nnet)
{
	// 1. Match the Weight Gradients
	_nnet.Weight_g.resize(_nnet.Weight.size());
	for (size_t l = 0; l < _nnet.Weight.size(); l++) {
		_nnet.Weight_g[l].resize(_nnet.Weight[l].size());
		for (size_t i = 0; i < _nnet.Weight[l].size(); i++) {
			_nnet.Weight_g[l][i].assign(_nnet.Weight[l][i].size(), 0.0f);
		}
	}

	// 2. Match the Bias Gradients
	_nnet.Bias_g.resize(_nnet.Bias.size());
	for (size_t l = 0; l < _nnet.Bias.size(); l++) {
		_nnet.Bias_g[l].assign(_nnet.Bias[l].size(), 0.0f);
	}
	return 0.0f;
}

/*
* 
* BACK_PROPAGATION KERNEL
* Purpose: Calculate error signals (Deltas) and accumulate weight gradients.
* Strategy: Virtual Transpose for memory efficiency; zero-allocation buffer reuse.
*/
float NNET::Back_Propagation(nnet& _nnet, std::vector<float>& result)
{
	// this is really clean lol
	//vector<float> delta_l = (_nnet.Last_Layer() - result) ^ (_nnet.Last_Layer() ^ (1 - _nnet.Last_Layer()));
	_nnet.Delta.resize(_nnet.Layer.size());
	_nnet.Delta[_nnet.Delta.size()-1].resize(result.size());
	vector<float>& delta_l = _nnet.Delta[_nnet.Delta.size() - 1];
#pragma omp parallel for
	for (int i = 0; i < delta_l.size(); i++)
	{
		delta_l[i] = (_nnet.Last_Layer()[i] - result[i]) * (_nnet.Last_Layer()[i] * (1 - _nnet.Last_Layer()[i]));
	}

	//the real propagation
	for (int l = _nnet.Layer.size() - 1; l > 0; l--)
	{
		//accumulate weight n bias
#pragma omp parallel for
		for (int i = 0; i < delta_l.size(); i++)
		{
			for (int o = 0; o < _nnet.Layer[l-1].size(); o++)
			{
				_nnet.Weight_g[l-1][o][i] +=  _nnet.Layer[l - 1][o] * delta_l[i];
			}
		}
		//PROFILE_NS("adding",
		//	_nnet.Bias_g[l] += delta_l;
		//);
		/*
		* old:
		[PROFILE] adding took: 300 ns
		[PROFILE] adding took: 79200 ns
		[PROFILE] adding took: 28900 ns
		* new:
		[PROFILE] adding took: 500 ns
		[PROFILE] adding took: 2800 ns
		[PROFILE] adding took: 4600 ns
		[PROFILE] adding took: 8900 ns
		* 
		*/
			for (int i = 0; i < _nnet.Bias_g[l].size();i++)
			{
				_nnet.Bias_g[l][i] += delta_l[i];
			}
		
		//update delta_l
		if (l > 1) {
			vector<float>& new_delta_l = _nnet.Delta[l - 1];
			new_delta_l.resize(_nnet.Layer[l - 1].size());
			std::fill(new_delta_l.begin(), new_delta_l.end(), 0.0f);
#pragma omp parallel for
			for (int o = 0; o < _nnet.Weight[l - 1].size(); o++)
			{
				for (int i = 0; i < _nnet.Weight[l - 1][o].size(); i++)
				{
					new_delta_l[o] += _nnet.Weight[l - 1][o][i] * delta_l[i];
				}
				
				new_delta_l[o] *= _nnet.Layer[l - 1][o] * (1.0f - _nnet.Layer[l - 1][o]);
			}
			delta_l = new_delta_l;
		}

	}
	return 0;
}

/**
 * @brief Performs a Stochastic Gradient Descent (SGD) update on the model parameters.
 * * This kernel scales the accumulated gradients by the normalized learning rate
 * and applies them to the weight/bias tensors. It utilizes recursive in-place
 * operator overloads to achieve a zero-allocation update cycle, maximizing
 * cache locality and minimizing heap fragmentation.
 *
 * @param _nnet The network structure containing parameter and gradient tensors.
 * @param learning_rate The scalar step-size for optimization.
 * @param batch_size The number of samples processed in the current accumulation.
 * @return 0 upon successful parameter synchronization.
 */
float NNET::Update_Model(NNET::nnet& _nnet, float& learning_rate, int& batch_size)
{
	float lr = learning_rate / batch_size;

	//update weight and bias
	for (int i = 0; i < _nnet.Layer.size(); i++)
	{
		//update bias
		_nnet.Bias_g[i] *= -lr;
		_nnet.Weight_g[i] *= -lr;

		_nnet.Bias[i] += _nnet.Bias_g[i];
		_nnet.Weight[i] += _nnet.Weight_g[i];
	}

	//claer the gradint accumulation
	ZeroOut(_nnet.Bias_g);
	ZeroOut(_nnet.Weight_g);
	return 0;
}

void NNET::Clear_Layer(NNET::nnet& _nnet)
{
	ZeroOut(_nnet.Layer);
}