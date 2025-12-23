#pragma once
#include <iostream>
using namespace std;
#include <vector>
#include <cmath>

namespace MATH {

	template <typename T>
	void Sigmoid(std::vector<T>& A)
	{
		for (auto& I : A)
		{
			I = 1.0 / (1.0 + std::exp(-I));
		}
	}



}
