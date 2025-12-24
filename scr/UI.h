#pragma once
#include "NeuralNetwork.h"
#include<iostream>

//debug
#include <iomanip>
void Print_Weights_Sample(const NNET::nnet& model, int layer_idx) {
    if (layer_idx >= model.Weight.size()) return;

    std::cout << "Weights Sample for Layer " << layer_idx << " -> " << layer_idx + 1 << ":" << std::endl;

    // Print only the first 5 neurons' weights to avoid clutter
    for (int i = 0; i < std::min((int)model.Weight[layer_idx].size(), 5); ++i) {
        std::cout << "  Neuron [" << i << "]: ";
        for (int j = 0; j < std::min((int)model.Weight[layer_idx][i].size(), 5); ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                << model.Weight[layer_idx][i][j] << " ";
        }
        std::cout << "..." << std::endl;
    }
}

void Display_Progress(int epoch, int current_batch, int total_batches, float loss, float accuracy) {
    // Calculate percentage
    float progress = (float)(current_batch + 1) / total_batches * 100.0f;

    // Create a simple visual bar [##########----------]
    int barWidth = 20;
    int pos = (int)(barWidth * (progress / 100.0f));

    std::cout << "\r" // Return to start of line
        << "[Epoch " << std::setw(2) << epoch << "] "
        << "[" << std::string(pos, '#') << std::string(barWidth - pos, '-') << "] "
        << std::fixed << std::setprecision(1) << std::setw(5) << progress << "% "
        << "| Loss: " << std::setprecision(4) << loss << " "
        << "| Acc: " << std::setprecision(2) << (accuracy * 100.0f) << "% "
        << std::flush; // Force the console to update immediately
}