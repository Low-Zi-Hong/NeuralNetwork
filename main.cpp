// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
using namespace std;

//debugging
#include <chrono>

#include "scr/NeuralNetwork.h"
#include "scr/FileManager.h"



#define nnet_structure {2,3,3,1} //{input, [hidden layer], output}

int main()
{
    std::cout << "Hello World!\n";

    //init model
    NNET::nnet NeuralNetwork(vector<int>(nnet_structure));



    //here decide to create a new model or load one from file
    bool load_else_create = false;
    if (load_else_create)
    {
        //FMANAGER::LoadFile();
    }
    else
    {
//        auto start = chrono::high_resolution_clock::now();
        NNET::Random_Initialise(NeuralNetwork);
//        auto end = chrono::high_resolution_clock::now();
//        std::cout << chrono::duration_cast<chrono::microseconds>(end - start).count();
        FMANAGER::NewFile(NeuralNetwork);
    }

    /*
    * input data set
    * 
    * 
    * training
    * {
    *   feedpropagation
    *   print result
    *   backpropagation 
    * }
    * 
    * save model
    * 
    */

    //generate dataset XOR one later do the MNIST dataset
    std::vector<std::vector<float>> dataset;
    dataset.reserve(1000);
    for (int i = 0; i < 1000; i++)
    {
        auto rannd = [](double x) {return (rand() % 2 == 0) ? 1.0 : 0.0; };
        dataset.push_back({ (float)rannd(0),(float)rannd(0) });
    }

    //split to mini batch
    int batchSize = 10;
    int numBatches = dataset.size() / batchSize;
    std::vector<std::vector<std::vector<float>>> b_dataset;
    b_dataset.resize(numBatches);
    for (int i = 0; i < dataset.size(); i++)
    {
        int batchIndex = i / batchSize;     
        int sampleIndex = i % batchSize;    

        if (sampleIndex == 0) b_dataset[batchIndex].resize(batchSize);

        b_dataset[batchIndex][sampleIndex] = dataset[i];
    }

    //generate result/ans
    std::vector<std::vector<std::vector<float>>> b_dataans(b_dataset.size());
    // 1. Resize the top-level container to match your batch count
    b_dataans.resize(numBatches);

    for (int b = 0; b < numBatches; b++) {
        // 2. Resize each batch to hold the correct number of answers
        b_dataans[b].resize(batchSize);

        for (int s = 0; s < batchSize; s++) {
            // 3. Resize each answer vector (XOR has 1 output)
            b_dataans[b][s].resize(1);

            // 4. Logic: XOR is 1 if inputs are different, 0 if same
            float in1 = b_dataset[b][s][0];
            float in2 = b_dataset[b][s][1];

            // The XOR "Truth": (in1 != in2)
            b_dataans[b][s][0] = (in1 != in2) ? 1.0f : 0.0f;
        }
    }




    int approach = 1;

    for (int ap = 0; ap < approach; ap++)
    {
        for (int batch = 0; batch < b_dataset.size(); batch++)
        {
            //gradient = 0;
#pragma omp parallel for
            for (int iteration = 0; iteration < b_dataset[batch].size(); iteration++)
            {
                //std::cout << iteration[0] << " ";

                //loading input
                NeuralNetwork.input(b_dataset[batch][iteration]);

                //running the model
                NNET::Feed_Propagation(NeuralNetwork);

                //calculate Error
                std::cout <<"[cost]: " << NNET::Calculate_Error(NeuralNetwork, b_dataans[batch][iteration]) << " ";

                //auto c_gradient = NNET::BackPropagation(NeuralNetwork, result);

                //gradient = gradient + c_gradient;

                //feed prop
                //back prop
                //accumulate gradient
            }
            //NNET::UpdateWeight(NeuralNetwork, gradient);
            std::cout << "\n";
        }
    }




    /*
    //debug purpose
    for (const auto& w : NeuralNetwork.Weight)
    {
        for (const auto& i : w)
        {
            for (const auto& o : i)
            {
                std::cout << o << " ";
            }
            std::cout << "\n";
            
        }
        std::cout << "\n\n\n";
    }

    //feed propagation

*/


}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
