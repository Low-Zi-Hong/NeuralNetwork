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

int MNIST::ReverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void MNIST::LoadImages(std::string filename, std::vector<std::vector<float>>& dataset) {
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open()) {
		std::cout << "[SYSTEM] Loading MNIST Images..." << std::endl;
		int magic_number = 0, num_images = 0, rows = 0, cols = 0;
		file.read((char*)&magic_number, 4);
		file.read((char*)&num_images, 4);
		file.read((char*)&rows, 4);
		file.read((char*)&cols, 4);

		num_images = ReverseInt(num_images);
		rows = ReverseInt(rows);
		cols = ReverseInt(cols);

		dataset.resize(num_images, std::vector<float>(rows * cols));

		for (int i = 0; i < num_images; i++) {
			for (int p = 0; p < rows * cols; p++) {
				unsigned char pixel = 0;
				file.read((char*)&pixel, 1);
				// NORMALIZATION: 0-255 becomes 0.0-1.0
				dataset[i][p] = (float)pixel / 255.0f;
			}

			// --- PROGRESS BAR LOGIC ---
			if (i % 1000 == 0 || i == num_images - 1) {
				float progress = (float)(i + 1) / num_images * 100;
				int barWidth = 30;

				std::cout << "\r[";
				int pos = barWidth * (progress / 100.0);
				for (int b = 0; b < barWidth; ++b) {
					if (b < pos) std::cout << "=";
					else if (b == pos) std::cout << ">";
					else std::cout << " ";
				}
				std::cout << "] " << (int)progress << "% (" << i + 1 << "/" << num_images << ")" << std::flush;
			}
		}
		std::cout << std::endl; // Break line when finished
		
	}
}


void MNIST::LoadLabels(std::string filename, std::vector<std::vector<float>>& labels) {
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open()) {
		std::cout << "[SYSTEM] Loading MNIST Labels..." << std::endl;
		int magic_number = 0, num_items = 0;
		file.read((char*)&magic_number, 4);
		file.read((char*)&num_items, 4);

		num_items = MNIST::ReverseInt(num_items);

		// 1. Resize the container: 60,000 samples, each having 10 output neurons
		labels.resize(num_items, std::vector<float>(10, 0.0f));

		for (int i = 0; i < num_items; i++) {
			unsigned char label = 0;
			file.read((char*)&label, 1);

			// 2. ONE-HOT ENCODING
			// If the label is '3', set the 3rd index to 1.0. 
			// Everything else remains 0.0.
			if (label < 10) {
				labels[i][label] = 1.0f;
			}

			// --- PROGRESS BAR LOGIC ---
			if (i % 1000 == 0 || i == num_items - 1) {
				float progress = (float)(i + 1) / num_items * 100;
				int barWidth = 30;

				std::cout << "\r[";
				int pos = barWidth * (progress / 100.0);
				for (int b = 0; b < barWidth; ++b) {
					if (b < pos) std::cout << "=";
					else if (b == pos) std::cout << ">";
					else std::cout << " ";
				}
				std::cout << "] " << (int)progress << "% (" << i + 1 << "/" << num_items << ")" << std::flush;
			}

		}
		std::cout << std::endl<< "[SYSTEM] Successfully loaded " << num_items << " labels." << std::endl;
	}
	else {
		std::cout << "[ERROR] Could not open label file!" << std::endl;
	}
}


void MNIST::ProcessImgLabel(
	std::vector < std::vector<float>>& dataset,	
std::vector < std::vector<float>>& dataans_raw,
	std::vector<std::vector<std::vector<float>>>& b_dataset, 
	std::vector<std::vector<std::vector<float>>>& b_dataans, 
	std::vector < std::vector<float>>& val_data, 
	std::vector < std::vector<float>>& val_ans,
	int batchSize
)
{


	std::vector < std::vector<float>> train_data, train_ans;


	int split_point = 50000;

	train_data.reserve(split_point);
	train_ans.reserve(split_point);

	val_data.reserve(60000 - split_point);
	val_ans.reserve(60000 - split_point);


	for (int i = 0; i < 60000; i++)
	{
		if (i < split_point)
		{
			train_data.push_back(dataset[i]);
			train_ans.push_back(dataans_raw[i]);
		}
		else
		{
			val_data.push_back(dataset[i]);
			val_ans.push_back(dataans_raw[i]);
		}
	}

	dataset.clear(); dataset.shrink_to_fit();
	dataans_raw.clear(); dataans_raw.shrink_to_fit();

	//split to mini batch
	int numBatches = train_data.size() / batchSize;

	b_dataset.resize(numBatches);
	b_dataans.resize(numBatches);
	for (int i = 0; i < train_data.size(); i++)
	{
		int batchIndex = i / batchSize;
		int sampleIndex = i % batchSize;

		if (batchIndex >= numBatches) break;

		if (sampleIndex == 0) b_dataset[batchIndex].resize(batchSize);
		if (sampleIndex == 0) b_dataans[batchIndex].resize(batchSize);

		b_dataset[batchIndex][sampleIndex] = train_data[i];
		b_dataans[batchIndex][sampleIndex] = train_ans[i];
	}

	train_data.clear(); train_data.shrink_to_fit();
	train_ans.clear(); train_ans.shrink_to_fit();
}







