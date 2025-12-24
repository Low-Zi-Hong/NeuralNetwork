// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
using namespace std;

//debugging
#include <chrono>

#include "scr/NeuralNetwork.h"
#include "scr/FileManager.h"
#include"scr/UI.h"
#include "scr/XOR.h"

#define nnet_structure {2,30,30,30,4} //{input, [hidden layer], output}

// ======= XOR Version of Model =======
#define XOR_Version
//comment if dont want


int approach = 10;
int batchSize = 10;


int main()
{
    std::cout << "Hello World!\n";

    //init model
    NNET::nnet NeuralNetwork(vector<int>(nnet_structure));

    //UI
    int choice = 0;
    std::string filename = "model_v1.nnet";

    std::cout << "========================================" << std::endl;
    std::cout << "   LZH Neural Network :D c++       " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "1. Create New Model (Random Weights)" << std::endl;
    std::cout << "2. Load Existing Model (FMANAGER)" << std::endl;
    std::cout << "Choice: ";
    std::cin >> choice;

    //here decide to create a new model or load one from file
    if (choice == 2)
    {
        FMANAGER::LoadFile(NeuralNetwork,filename);
        Print_Weights_Sample(NeuralNetwork, 1);
    }
    else
    {
        std::cout << "[SYSTEM] Initializing new model with random Gaussian weights..." << std::endl;
        NNET::Random_Initialise(NeuralNetwork);
        FMANAGER::NewFile(NeuralNetwork);
    }


    //generate dataset XOR one later do the MNIST dataset
    std::vector<std::vector<float>> dataset;
#ifdef XOR_Version
    XOR::init(dataset);
#endif // 


    //split to mini batch
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

#ifdef XOR_Version
    XOR::generateAns;
#endif // XOR_Version




    NNET::Init_Gradient_Accumulation(NeuralNetwork);
    for (int ap = 0; ap < approach; ap++)
    {
        std::cout << std::endl;
        for (int batch = 0; batch < b_dataset.size(); batch++)
        {
            float batch_total_cost = 0;
            int correct_hits = 0;
            //gradient = 0;
            for (int iteration = 0; iteration < b_dataset[batch].size(); iteration++)
            {
                //std::cout << iteration[0] << " ";

                //loading input
                NeuralNetwork.input(b_dataset[batch][iteration]);

                //running the model
                NNET::Feed_Propagation(NeuralNetwork);

                //calculate Error
                std::vector<float> result = { b_dataans[batch][iteration][0],0,0,0};
                float current_cost = NNET::Calculate_Error(NeuralNetwork, result);
                batch_total_cost += current_cost;

                // Accuracy Logic: Check if prediction matches target (threshold 0.5)
                float prediction = NeuralNetwork.Last_Layer()[0];
                float target = b_dataans[batch][iteration][0];
                if ((prediction >= 0.5f && target == 1.0f) || (prediction < 0.5f && target == 0.0f)) {
                    correct_hits++;
                }

                NNET::Back_Propagation(NeuralNetwork, result);

                NNET::Clear_Layer(NeuralNetwork);

                //feed prop done
                //back prop done
                //accumulate gradient done
            }

            float learning_rate = 1.0f;
            NNET::Update_Model(NeuralNetwork, learning_rate,batchSize);

            // 5. Live Dashboard Call
            float avg_cost = batch_total_cost / batchSize;
            float accuracy = (float)correct_hits / batchSize;
            Display_Progress(ap, batch, b_dataset.size(), avg_cost, accuracy);
        }
    }

    std::cout << "\n[SYSTEM] Epoch " << approach << " Complete. Model Saved." << std::endl;
    FMANAGER::SaveFile(NeuralNetwork, filename);



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
