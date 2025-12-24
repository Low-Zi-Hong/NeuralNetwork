// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
using namespace std;

//debugging
#include <chrono>
#define PROFILE_SCOPE(name, code_block) \
    { \
        auto start = std::chrono::high_resolution_clock::now(); \
        code_block \
        auto end = std::chrono::high_resolution_clock::now(); \
        std::chrono::duration<float> duration = end - start; \
        std::cout << "[PROFILE] " << name << " took: " << duration.count() << " seconds." << std::endl; \
    }
#define PROFILE_MS(name, code_block) \
    { \
        auto start = std::chrono::high_resolution_clock::now(); \
        code_block \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
        std::cout << "[PROFILE] " << name << " took: " << duration.count() << " us" \
                  << " (" << (float)duration.count() / 1000.0f << " ms)" << std::endl; \
    }

#include "scr/NeuralNetwork.h"
#include "scr/FileManager.h"
#include"scr/UI.h"
#include "scr/XOR.h"
#include "scr/Run.h"

#define nnet_structure {784,256,128,64,10} //{input, [hidden layer], output}


// ======= XOR Version of Model =======
//#define XOR_Version
//comment if dont want

#ifdef XOR_Version
    #define nnet_structure { 2,10,10,1 }
#endif


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
    std::cout << "3. Using The Model" << std::endl;
    std::cout << "Choice: ";
    std::cin >> choice;

    //here decide to create a new model or load one from file
    if (choice == 2)
    {
        FMANAGER::LoadFile(NeuralNetwork,filename);
        Print_Weights_Sample(NeuralNetwork, 1);
    }
    if (choice == 3)
    {
        FMANAGER::LoadFile(NeuralNetwork, filename);
        Print_Weights_Sample(NeuralNetwork, 1);

        if (NeuralNetwork.structure.empty()) std::abort;

        bool ui_active = true;
        while (ui_active)
        {
            std::vector<float> inp;

#ifdef XOR_Version
            float a, b;
            std::cout << "Enter Input A (0 or 1): "; std::cin >> a;
            if (a == -1) { ui_active = false; std::abort; break; }
            std::cout << "Enter Input B (0 or 1): "; std::cin >> b;
            inp = { a, b };
#else
            int image_index;
            std::cout << "Enter MNIST Image Index (0-9999): "; std::cin >> image_index;
            //user_input = MNIST::GetImage(image_index);
#endif

            NeuralNetwork.input(inp);
            NNET::Feed_Propagation(NeuralNetwork);

#ifdef XOR_Version
            std::cout << "Prediction: " << NeuralNetwork.Last_Layer()[0] << std::endl;
#else
            //int prediction = NNET::Get_Highest_Output(NeuralNetwork);
            //std::cout << "The Brain thinks this digit is: " << prediction << std::endl;
#endif
        }

    }
    else
    {
        std::cout << "[SYSTEM] Initializing new model with random Gaussian weights..." << std::endl;
        NNET::Random_Initialise(NeuralNetwork);
        FMANAGER::NewFile(NeuralNetwork);
    }

#ifdef XOR_Version
    //generate dataset XOR one later do the MNIST dataset
    std::vector<std::vector<float>> dataset;

    XOR::init(dataset);


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

    XOR::generateAns(batchSize,numBatches,b_dataans,b_dataset);


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
                std::vector<float> result = b_dataans[batch][iteration];
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
            NNET::Update_Model(NeuralNetwork, learning_rate, batchSize);

            // 5. Live Dashboard Call
            float avg_cost = batch_total_cost / batchSize;
            float accuracy = (float)correct_hits / batchSize;
            Display_Progress(ap, batch, b_dataset.size(), avg_cost, accuracy);
        }
    }

    std::cout << "\n[SYSTEM] Epoch " << approach << " Complete. Model Saved." << std::endl;
    FMANAGER::SaveFile(NeuralNetwork, filename);






#else

    std::vector < std::vector<float>> dataset;
    std::vector < std::vector<float>> dataans_raw;

    MNIST::LoadImages("train-images.idx3-ubyte", dataset);
    MNIST::LoadLabels("train-labels.idx1-ubyte", dataans_raw);

    std::vector < std::vector<float>> train_data, train_ans;
    std::vector < std::vector<float>> val_data, val_ans;

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
    std::vector<std::vector<std::vector<float>>> b_dataset;
    std::vector<std::vector<std::vector<float>>> b_dataans;
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
                std::vector<float> result = b_dataans[batch][iteration];
                float current_cost = NNET::Calculate_Error(NeuralNetwork, result);
                batch_total_cost += current_cost;

                // Accuracy Logic: Check if prediction matches target (threshold 0.5)
                int target_digit = 0;
                for (int i = 0; i < 10; i++)
                {
                    if (result[i] == 1.0f)
                    {
                        target_digit = i;
                    }
                }

                if (NeuralNetwork.MNISTResult() == target_digit) correct_hits++;

                NNET::Back_Propagation(NeuralNetwork, result);

                NNET::Clear_Layer(NeuralNetwork);

                //feed prop done
                //back prop done
                //accumulate gradient done
            }

            float learning_rate = 1.0f;
            NNET::Update_Model(NeuralNetwork, learning_rate, batchSize);

            // 5. Live Dashboard Call
            float avg_cost = batch_total_cost / batchSize;
            float accuracy = (float)correct_hits / batchSize;
            Display_Progress(ap, batch, b_dataset.size(), avg_cost, accuracy);
        }

        int val_correct = 0;
        float val_total_cost = 0;

        std::cout << "\n[SYSTEM] Running Validation Pass on 10,000 samples..." << std::endl;

        for (int i = 0; i < val_data.size(); i++) {
            // 1. Forward Pass Only (No Backprop here!)
            NeuralNetwork.input(val_data[i]);
            NNET::Feed_Propagation(NeuralNetwork);

            // 2. Calculate Cost (Optional, but good for logs)
            val_total_cost += NNET::Calculate_Error(NeuralNetwork, val_ans[i]);

            // 3. Winner Logic (ArgMax)
            int predicted = 0;
            float max_act = NeuralNetwork.Last_Layer()[0];
            for (int n = 1; n < 10; n++) {
                if (NeuralNetwork.Last_Layer()[n] > max_act) {
                    max_act = NeuralNetwork.Last_Layer()[n];
                    predicted = n;
                }
            }

            // 4. Check Answer
            int actual = 0;
            for (int n = 0; n < 10; n++) {
                if (val_ans[i][n] == 1.0f) {
                    actual = n;
                    break;
                }
            }

            if (predicted == actual) val_correct++;

            // 5. Clean up for next sample
            NNET::Clear_Layer(NeuralNetwork);
        }

        float final_val_acc = (float)val_correct / val_data.size();
        float final_val_loss = val_total_cost / val_data.size();

        std::cout << ">> [EPOCH " << ap << "] Validation Accuracy: " << (final_val_acc * 100.0f) << "%" << std::endl;
        std::cout << ">> [EPOCH " << ap << "] Validation Loss: " << final_val_loss << std::endl;


    }



    std::cout << "\n[SYSTEM] Epoch " << approach << " Complete. Model Saved." << std::endl;
    FMANAGER::SaveFile(NeuralNetwork, filename);



#endif // XOR_Version







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
