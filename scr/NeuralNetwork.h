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

		//this for gradient accumulation
		vector<vector<float>> Delta;
		vector<vector<vector<float>>> Weight_g;
		vector<vector<float>> Bias_g;

		/*
		This Initialise the neural network.
		This create and initialise the weight bias and layer of the network from user input and save the model
		or
		load the network from exsiting file
		*/
		nnet(vector<int> _structure);

		void reinit(vector<int> _structure);

		int input(vector<float>& inp)
		{
			if (inp.size() == Layer[0].size()) Layer[0] = inp;
			return 0;
		}

		std::vector<float>& Last_Layer()
		{
			return Layer[Layer.size() - 1];
		}

		int MNISTResult()
		{
			int predicted_digit = 0;
			float max_activation = Last_Layer()[0];
			for (int i = 1; i < 10; i++) {
				if (Last_Layer()[i] > max_activation) {
					max_activation = Last_Layer()[i];
					predicted_digit = i;
				}
			}
			return predicted_digit;
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

	float Init_Gradient_Accumulation(nnet& _nnet);

	float Back_Propagation(nnet& _nnet, std::vector<float>& result);

	float Update_Model(NNET::nnet& _nnet, float& learning_rate, int& batch_size);

	void Clear_Layer(NNET::nnet& _nnet);


}



#endif