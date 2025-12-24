#pragma once
#include<iostream>
#include<vector>
namespace XOR {

	void init(std::vector<std::vector<float>>& dataset)
	{
        dataset.reserve(10000);
        for (int i = 0; i < 10000; i++)
        {
            auto rannd = [](double x) {return (rand() % 2 == 0) ? 1.0 : 0.0; };
            dataset.push_back({ (float)rannd(0),(float)rannd(0) });
        }
	}

    void generateAns(int& batchSize,int& numBatches, std::vector< std::vector<std::vector<float>>>& b_dataans, std::vector< std::vector<std::vector<float>>>& b_dataset)
    {
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
    }




}