#ifndef F_MANAGER
#define F_MANAGER

#pragma once
#include <iostream>
#include<fstream>
#include <vector>
using namespace std;



namespace FMANAGER
{
	/*
	* This load file from the directory and load it to the model
	*/
	int LoadFile(NNET::nnet& _nnet, const std::string& path);

	/*
	* This create and save the neural network
	*/
	int NewFile(const NNET::nnet& _nnet);

	int SaveFile(const NNET::nnet& _nnet, string& file_path);
}

namespace MNIST
{
#ifndef RINT
#define RINT
	int ReverseInt(int i);
#endif // 

	void LoadImages(std::string filename, std::vector<std::vector<float>>& dataset);

	void LoadLabels(std::string filename, std::vector<std::vector<float>>& labels);
}
#endif