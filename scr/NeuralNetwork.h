#ifndef NNET_H
#define NNET_H

#pragma once
#include <iostream>
#include <vector>
using namespace std;


namespace NNET
{
	class nnet
	{
	public:
		//structure of Neural Network, Input Weight Bias Layer
		vector<vector<vector<float>>> Weight;
		vector<vector<float>> Bias;
		vector<vector<float>> Layer;

		vector<int> structure;

		vector<float> Error;

		/*
		This Initialise the neural network.
		This create and initialise the weight bias and layer of the network from user input and save the model
		or
		load the network from exsiting file
		*/
		nnet(vector<int> _structure);

		int input(vector<float>& inp)
		{
			if (inp.size() == Layer[0].size()) Layer[0] = inp;
			return 0;
		}
	};

	/*
	* This create a model which contain random numbers
	*/
	int Random_Initialise(nnet& _nnet);

	/*
	* This is the feedpropagation of the model
	*/
	int Feed_Propagation(nnet& _nnet);

	float Calculate_Error(nnet& _nnet, std::vector<float>& result);



}



#endif