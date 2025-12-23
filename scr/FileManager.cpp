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
int FMANAGER::LoadFile(NNET::nnet& _nnet)
{
	return 0;
}

/*
* This create and save the neural network
*/
int FMANAGER::NewFile(const NNET::nnet& _nnet)
{
	string filename = "model_v1.nnet";
	FMANAGER::SaveFile(_nnet, _nnet.structure,filename);
	return 0;
}

int FMANAGER::SaveFile(const NNET::nnet& _nnet, const vector<int> _structure, string& file_path)
{
	ofstream output(file_path);

	if (output.is_open())
	{
		for (int e : _structure) output << e << " ";
		output << "\n";

		for (auto& l : _nnet.Weight)
		{
			for (auto& r : l)
			{
				for (auto& e : r)
				{
					output << e << " ";
				}
				output << "\n";
			}
		}

		for (auto& l : _nnet.Bias)
		{
			for (auto& b : l)
			{
				output << b << " ";
			}
			output << "\n";
		}
		output.close();
	}
	return 0;
}














