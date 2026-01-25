//#define PythonLib

#ifdef PythonLib

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Emath.h"
#include "NeuralNetwork.h"
#include "FileManager.h"


namespace py = pybind11;

int add(int i, int j) { return i + j; }

/**
 * LZH Neural Engine - High Performance C++ Backend
 * -----------------------------------------------
 * This module implements the pybind11 bridge, exposing native C++ neural
 * kernels to the Python interpreter for rapid prototyping and telemetry.
 */
PYBIND11_MODULE(NnetLZH, m) {
	m.doc() = "pybind11 plugin";

	m.def("add", &add, "a fuction that add");

	py::class_<NNET::nnet>(m, "NeuralNet")
		.def(py::init<std::vector<int>>(),
			py::arg("structure"))

		.def("input", &NNET::nnet::input)
		.def("rawResult", &NNET::nnet::Last_Layer)
		.def("MNISTResult", &NNET::nnet::MNISTResult)

		//look inside model
		.def_readonly("layers", &NNET::nnet::Layer);

	
	m.def("RandomInitialise", &NNET::Random_Initialise, "Random Initialise the model");
	m.def("FeedPropagation", &NNET::Feed_Propagation, "Feed Propagate the model");
	m.def("CalculateError", &NNET::Calculate_Error);
	m.def("ClearLayer", &NNET::Clear_Layer);

	m.def("LoadModel", &FMANAGER::LoadFile, "Load model");
	m.def("SaveModel", &FMANAGER::SaveFile, "Save model");

	m.def("LoadImg", &MNIST::LoadImages, "Load MNIST image");
	m.def("LoadLabel", &MNIST::LoadLabels, "Load MNIST label");
	m.def("ProcessData", &MNIST::ProcessImgLabel, "Process MNIST Images and Labels");

}


#endif // PythonLib