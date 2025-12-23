#ifndef F_MANAGER
#define F_MANAGER

#pragma once
#include <iostream>
#include <vector>
using namespace std;



namespace FMANAGER
{
	/*
	* This load file from the directory and load it to the model
	*/
	int LoadFile(NNET::nnet& _nnet);

	/*
	* This create and save the neural network
	*/
	int NewFile(const NNET::nnet& _nnet);

	/*
	* This save file
	*/
	int SaveFile(const NNET::nnet& _nnet, const vector<int> _structure, string& file_path);

}

#endif