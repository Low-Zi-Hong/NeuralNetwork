#ifdef PythonLib

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

int add(int i, int j) { return i + j; }

PYBIND11_MODULE(ADDINGINCPP, m) {
	m.doc() = "pybind11 plugin";

	m.def("add", &add, "a fuction that add");
}


#endif // PythonLib