#pragma once
#include <iostream>
#include <chrono>
#define PROFILE_SCOPE(name, code_block) \
    { \
        auto start = std::chrono::high_resolution_clock::now(); \
        code_block \
        auto end = std::chrono::high_resolution_clock::now(); \
        std::chrono::duration<float> duration = end - start; \
        std::cout << "\r[PROFILE] " << name << " took: " << duration.count() << " seconds." << std::flush; \
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
#define PROFILE_NS(name, code_block) \
    { \
        auto start = std::chrono::high_resolution_clock::now(); \
        code_block \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start); \
        std::cout << "[PROFILE] " << name << " took: " << duration.count() << " ns" \
                  << std::endl; \
    }