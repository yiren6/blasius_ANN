import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import vtk
except ImportError:
    install('vtk')
    import vtk

def extract_vtu_data(filename):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()  # Necessary to load the data

    unstructured_grid = reader.GetOutput()
    data_dict = {}

    for i in range(unstructured_grid.GetPointData().GetNumberOfArrays()):
        array_name = unstructured_grid.GetPointData().GetArrayName(i)
        array = unstructured_grid.GetPointData().GetArray(array_name)
        python_list = [array.GetValue(j) for j in range(array.GetSize())]
        data_dict[array_name] = python_list

    return data_dict

if __name__ == "__main__":
    filename = "Inc_Laminar_Flat_Plate/flow.vtu"
    data = extract_vtu_data(filename)
    if data:
        print("Test passed: Data extracted successfully.")
    else:
        print("Test failed: No data extracted.")