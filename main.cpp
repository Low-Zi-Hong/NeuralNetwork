// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
using namespace std;

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


int approach = 20;
int batchSize = 30;
float learning_rate = 1.0f;


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
    else if (choice == 3)
    {
#ifdef XOR_Version
        FMANAGER::LoadFile(NeuralNetwork, filename);
        Print_Weights_Sample(NeuralNetwork, 1);

        if (NeuralNetwork.structure.empty()) std::abort;

        bool ui_active = true;
        while (ui_active)
        {
            std::vector<float> inp;


            float a, b;
            std::cout << "Enter Input A (0 or 1): "; std::cin >> a;
            if (a == -1) { ui_active = false; std::abort; break; }
            std::cout << "Enter Input B (0 or 1): "; std::cin >> b;
            inp = { a, b };

            std::cout << "Prediction: " << NeuralNetwork.Last_Layer()[0] << std::endl;
    }
#else
            //int prediction = NNET::Get_Highest_Output(NeuralNetwork);
            //std::cout << "The Brain thinks this digit is: " << prediction << std::endl;
        // 1. Load the trained model
        FMANAGER::LoadFile(NeuralNetwork, filename);
        Print_Weights_Sample(NeuralNetwork, 1);

        if (NeuralNetwork.structure.empty()) {
            std::cerr << "Error: Model structure is empty. Load failed." << std::endl;
            std::abort();
        }

        // 2. Load the MNIST Test Dataset into memory once
        std::vector<std::vector<float>> test_images;
        std::vector<std::vector<float>> test_labels;

        std::cout << "Loading MNIST Test Data..." << std::endl;
        // Assuming these are your binary test files
        MNIST::LoadImages("train-images.idx3-ubyte", test_images);
        MNIST::LoadLabels("train-labels.idx1-ubyte", test_labels);

        bool ui_active = true;
        while (ui_active)
        {
            int image_index;
            std::cout << "\n------------------------------------------" << std::endl;
            std::cout << "Enter MNIST Image Index (0-9999) or -1 to exit: ";
            std::cin >> image_index;

            if (image_index == -1) {
                ui_active = false;
                std::abort;
                break;
            }

            if (image_index < 0 || image_index >= test_images.size()) {
                std::cout << "Invalid Index! Please try again." << std::endl;
                continue;
            }

            // 3. Extract the image vector and feed it to the model
            std::vector<float> inp = test_images[image_index];
            NeuralNetwork.input(inp);

            // 4. Run Inference (Forward Pass)
            PROFILE_MS("feed",
                NNET::Feed_Propagation(NeuralNetwork);
            );
            // 5. Interpret the Results
            int prediction = NeuralNetwork.MNISTResult(); // Your C++ index-max function

            // Find the actual label for comparison
            int actual_label = 0;
            for (int i = 0; i < 10; i++) if (test_labels[image_index][i] > 0.5f) actual_label = i;

            std::cout << "The Brain thinks this digit is: " << prediction << std::endl;
            std::cout << "The Actual Label is: " << actual_label << std::endl;

            if (prediction == actual_label)
                std::cout << "RESULT: CORRECT! [V]" << std::endl;
            else
                std::cout << "RESULT: WRONG! [X]" << std::endl;
        }

#endif
        

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


    std::vector<std::vector<std::vector<float>>> b_dataset;
    std::vector<std::vector<std::vector<float>>> b_dataans;

    std::vector < std::vector<float>> val_data, val_ans;


    MNIST::ProcessImgLabel(dataset, dataans_raw, b_dataset, b_dataans, val_data, val_ans, batchSize);

    float lr = learning_rate;
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
                //PROFILE_MS("Feed",
                    NNET::Feed_Propagation(NeuralNetwork);
                //);

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

                //PROFILE_MS("BP",
                    NNET::Back_Propagation(NeuralNetwork, result);
                //);

                NNET::Clear_Layer(NeuralNetwork);

                //feed prop done
                //back prop done
                //accumulate gradient done
            }



            NNET::Update_Model(NeuralNetwork, lr, batchSize);

            // 5. Live Dashboard Call
            float avg_cost = batch_total_cost / batchSize;
            float accuracy = (float)correct_hits / batchSize;
            Display_Progress(ap, batch, b_dataset.size(), avg_cost, accuracy);
            
            NNET::Updatelr(accuracy,lr, learning_rate);
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

        std::cout << "\n[SYSTEM] Epoch " << ap << " Complete. Model Saved." << std::endl;
        FMANAGER::SaveFile(NeuralNetwork, filename);
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
