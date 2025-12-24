#include <iostream>
#include<string>
#include <vector>
using namespace std;
#include<fstream>

#include "NeuralNetwork.h"
#include "FileManager.h"

/*
* This load file from the directory and load it to the model
*/
int FMANAGER::LoadFile(NNET::nnet& _nnet,const std::string& path)
{
	std::ifstream file(path, std::ios::binary);

	if (!file.is_open()) std::abort;

	size_t arch_size;
	file.read(reinterpret_cast<char*>(&arch_size), sizeof(arch_size));

	std::vector<int> file_arch(arch_size);
	file.read(reinterpret_cast<char*>(file_arch.data()), arch_size * sizeof(int));

	if (file_arch != _nnet.structure)
	{
		std::cout << "[FMANAGER] Architecture mismatch! Re-initializing model to match file..." << std::endl;
		_nnet.reinit(file_arch);
	}

	for (auto& layer : _nnet.Weight)
	{
		for (auto& neuron_weights : layer)
		{
			file.read(reinterpret_cast<char*>(neuron_weights.data()),
				neuron_weights.size() * sizeof(float));
		}
	}

	for (auto& bias_layer : _nnet.Bias)
	{
		file.read(reinterpret_cast<char*>(bias_layer.data()),
			bias_layer.size() * sizeof(float));
	}

	file.close();


	return 0;
}

/*
* This create and save the neural network
*/
int FMANAGER::NewFile(const NNET::nnet& _nnet)
{
	string filename = "model_v1.nnet";
	FMANAGER::SaveFile(_nnet,filename);
	return 0;
}

int FMANAGER::SaveFile(const NNET::nnet& _nnet, string& file_path)
{
	ofstream output(file_path, std::ios::binary);

	if (output.is_open())
	{

		size_t arch_size = _nnet.structure.size();
		output.write(reinterpret_cast<const char*>(&arch_size), sizeof(arch_size));
		output.write(reinterpret_cast<const char*>(_nnet.structure.data()),
			arch_size * sizeof(int));

		for (auto& layer : _nnet.Weight) {
			for (auto& neuron : layer) {
				output.write(reinterpret_cast<const char*>(neuron.data()),
					neuron.size() * sizeof(float));
			}
		}

		for (auto& bias_layer : _nnet.Bias) {
			output.write(reinterpret_cast<const char*>(bias_layer.data()),
				bias_layer.size() * sizeof(float));
		}

		output.close();

		//for (int e : _structure) output << e << " ";
		//output << "\n";
		//
		//for (auto& l : _nnet.Weight)
		//{
		//	for (auto& r : l)
		//	{
		//		for (auto& e : r)
		//		{
		//			output << e << " ";
		//		}
		//		output << "\n";
		//	}
		//}
		//
		//for (auto& l : _nnet.Bias)
		//{
		//	for (auto& b : l)
		//	{
		//		output << b << " ";
		//	}
		//	output << "\n";
		//}
		//output.close();
	}
	return 0;
}














