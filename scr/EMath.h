#pragma once
#include <iostream>
using namespace std;
#include <vector>
#include <algorithm>
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



//vector operator
template <typename T>
inline vector<T> operator+(const vector<T>& a, const vector<T>& b)
{
	std::vector<T> result(a.size());
	if (a.size() != b.size()) std::abort;
	for (size_t i = 0; i < a.size(); i++)
	{
		result[i] = a[i] + b[i];
	}
	return result;
}

template <typename T>
inline vector<T> operator-(const float& a, const vector<T>& b)
{
	std::vector<T> result(a.size());
	if (a.size() != b.size()) std::abort;
	for (size_t i = 0; i < a.size(); i++)
	{
		result[i] = a - b[i];
	}
	return result;
}

template <typename T>
inline vector<T> operator-(const vector<T>& a, const vector<T>& b)
{
	std::vector<T> result(a.size());
	if (a.size() != b.size()) std::abort;
	for (size_t i = 0; i < a.size(); i++)
	{
		result[i] = a[i] - b[i];
	}
	return result;
}

template <typename T>
inline vector<T> operator*(const float a, const vector<T>& b)
{
	std::vector<T> result(b.size());
	for (size_t i = 0; i < b.size(); i++)
	{
		result[i] = a * b[i];
	}
	return result;
}




template <typename T>
inline vector<T> operator^(const vector<T>& a, const vector<T>& b)
{
	std::vector<T> result(a.size());
	if (a.size() != b.size()) std::abort;
	for (size_t i = 0; i < a.size(); i++)
	{
		result[i] = a[i] * b[i];
	}
	return result;
}


template <typename T>
inline void operator+=(vector<T>& a, const vector<T>& b)
{
	if (a.size() != b.size()) std::abort;
	for (size_t i = 0; i < a.size(); i++)
	{
		a[i] += b[i];
	}
}

template <typename T>
inline void operator*=(vector<T>& a, const float b)
{
	//if (a.size != b.size) std::abort;
	for (size_t i = 0; i < a.size(); i++)
	{
		a[i] *= b;
	}
}


template <typename T>
inline void ZeroOut(vector<T>& a);

inline void ZeroOut(float& f) { f = 0.0f; }

template <typename T>
inline void ZeroOut(vector<T>& a)
{
	for (auto& i : a)
	{
		ZeroOut(i);
	}
}

