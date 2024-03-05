import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import vtk
    from vtk.util import numpy_support
except ImportError:
    install('vtk')
    import vtk
    from vtk.util import numpy_support



def extract_vtu_data(filename):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()  # Necessary to load the data_dict

    unstructured_grid = reader.GetOutput()
    data_dict = {}

    # get number of points, number of dimensions, and coordinates
    coords = numpy_support.vtk_to_numpy(unstructured_grid.GetPoints().GetData())

    n_points, n_dim = np.shape(coords)
    data_dict['Coordinates'] = coords

    # get nodal coordinate connectivity 
    connectivity = numpy_support.vtk_to_numpy(unstructured_grid.GetCells().GetData())
    connectivity = connectivity.reshape((-1, 5)) # 4 nodes per cell
    connectivity = connectivity[:, 1:]

    # find the cell index at bottom right corner of [10,-4]
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(unstructured_grid) 
    ini_cell_idx = locator.FindCell([0.3048, 0, 0])

    # get nodal chain from connectivity
    ini_cell_nodes = coords[connectivity[ini_cell_idx]]
    ini_max_y = np.max(ini_cell_nodes[:, 1])
    max_y_idx = np.where(ini_cell_nodes[:, 1] == ini_max_y)[0][1]
    next_ini_y = ini_cell_nodes[max_y_idx][1]
    counter = 0
    termin_flag = False
    while (next_ini_y < 0.03) & (not termin_flag):
        if counter == 0:
            nodal_chain, next_layer_ini_idx, nxt_nodal_chain = _extract_right_nodes(connectivity, ini_cell_idx, coords)
            len_C_loop = np.size(nodal_chain)
            counter += 1
        else:
            ini_node_xyz = coords[next_layer_ini_idx]
            ini_cell_idx = locator.FindCell(ini_node_xyz+[0,1e-5,0]) # add a small val to ensure its the cell above last one
            cur_nodal_chain, next_layer_ini_idx, nxt_nodal_chain = _extract_right_nodes(connectivity, ini_cell_idx, coords)

            if np.size(cur_nodal_chain) != len_C_loop:
                termin_flag = True
                print(f"Chain length inconsistency for {counter}th layer!")
                break
            nodal_chain.extend(cur_nodal_chain)
            counter += 1

            # add last layer in center symmetry
            if next_layer_ini_idx == 9:
                nodal_chain.extend(nxt_nodal_chain)
                print(f"Arrived at center node at {counter}th layer!")
                termin_flag = True

    # check if any vertex is missing 
    all_nodes = np.arange(0, connectivity.max()+1)    
    is_all_nodes_contained = _check_elements(all_nodes, nodal_chain)
    if not is_all_nodes_contained:
        print("WARNING: NOT ALL NODES CONTAINED")

    # reshape nodal_chain to 2d
    len_layers = np.size(nodal_chain)//len_C_loop
    nodal_chain_shaped = np.array(nodal_chain).reshape(len_layers, -1)

    # get number of fields, their names, and field data_dict
    n_fields  = unstructured_grid.GetPointData().GetNumberOfArrays()

    field_names = []
    for i in range(n_fields):
        field_name = unstructured_grid.GetPointData().GetArrayName(i)
        field_names.append(field_name)

        if field_name in ['Pressure', 'Velocity', 'Temperature', 'Pressure_Coefficient', 'Density', 'Laminar_Viscosity', 'Heat_Capacity', 'Thermal_Conductivity', 'Skin_Friction_Coefficient','Heat_Flux','Y_Plus']:
            data_dict[field_name] = numpy_support.vtk_to_numpy(unstructured_grid.GetPointData().GetArray(field_name))
    
    # print current file dir and dimensions of nodal_chain_shaped
    print(f"Dimensions of nodal_chain_shaped: {np.shape(nodal_chain_shaped)}")

    return data_dict



def _check_elements(target_vector, second_vector):
    return all(element in second_vector for element in target_vector)



def _extract_right_nodes(element_list, initial_cell_idx, coords):
    # input: numpy array of 2D element list
    nodal_chain = []
    nxt_nodal_chain = [] # nodal chain reserved for next layer
    is_close_loop = False
    iter = 0
    while not is_close_loop:
        # 1st iter
        if iter == 0:
            cur_elem = element_list[initial_cell_idx]
            ini_first_node, second_node, third_node, last_node = cur_elem
            # in indexing the corner node is smallest 
            nodal_xyz = coords[[ini_first_node,second_node,third_node,last_node]]
            max_x = np.max(nodal_xyz[:,0])
            min_y = np.min(nodal_xyz[:,1])
            max_x_idx = np.where(nodal_xyz[:, 0] == max_x)[0]
            min_y_idx = np.where(nodal_xyz[:, 1] == min_y)[0]
            first_idx = np.intersect1d(max_x_idx, min_y_idx)[0]
            second_idx = min_y_idx[min_y_idx != first_idx][0]
            if (first_idx == 2) & (second_idx == 1):
                temp_val_1 = ini_first_node
                temp_val_2 = second_node
                ini_first_node = third_node
                second_node = last_node
                third_node = temp_val_1
                last_node = temp_val_2

            nodal_chain.append(ini_first_node)
            nodal_chain.append(last_node)
            adjacent_cell_pair = np.where(np.any(np.isin(element_list, third_node), axis=1) & np.any(np.isin(element_list, last_node), axis=1))
            adjacent_cell_pair = adjacent_cell_pair[0]
            if adjacent_cell_pair.size > 1:
                nxt_elem_idx = adjacent_cell_pair[adjacent_cell_pair != initial_cell_idx][-1]
            else:
                is_close_loop = True
                continue
            prev_elem_idx = -1
            iter += 1
            next_layer_ini_idx = second_node
            if next_layer_ini_idx == 9:
                # arrived at center node
                nxt_nodal_chain.append(next_layer_ini_idx)
                nxt_nodal_chain.append(third_node)
                
            continue
        # if lst node is initial first node, break
        if last_node == ini_first_node:
            is_close_loop = True
            continue
        # if next cell index is same as current, break
        if prev_elem_idx == nxt_elem_idx:
            is_close_loop = True
            continue

        # else
        prev_elem_idx = nxt_elem_idx
        prev_second_node = second_node
        prev_third_node = third_node
        prev_last_node = last_node

        cur_elem = element_list[nxt_elem_idx]
        first_node, second_node, third_node, last_node = cur_elem
        if (prev_last_node == first_node) & (prev_third_node == second_node):
            # 1-4  -- 1-4
            # 2-3  -- 2-3
            # condition, do not modify ordering
            pass

        if (prev_last_node == last_node) & (prev_third_node == first_node):
            # 1-4 -- 4-3
            # 2-3 -- 1-2
            # condition, change ordering
            temp_val_1 = third_node
            temp_val_2 = second_node
            second_node = first_node
            first_node = last_node
            last_node = temp_val_1
            third_node = temp_val_2

        if (prev_last_node == third_node) & (prev_third_node == last_node):
            # 1-4 -- 3-2
            # 2-3 -- 4-1
            # condition, change ordering
            temp_val_1 = first_node
            temp_val_2 = second_node
            first_node = third_node
            second_node = last_node
            third_node = temp_val_1
            last_node = temp_val_2    

        nodal_chain.append(last_node)
        nxt_nodal_chain.append(third_node)
        adjacent_cell_pair = np.where(np.any(np.isin(element_list, third_node), axis=1) & np.any(np.isin(element_list, last_node), axis=1))
        adjacent_cell_pair = adjacent_cell_pair[0]
        if adjacent_cell_pair.size > 1:
            nxt_elem_idx = adjacent_cell_pair[adjacent_cell_pair != prev_elem_idx][-1]
        else:
            is_close_loop = True
            continue
        iter += 1

    return nodal_chain, next_layer_ini_idx, nxt_nodal_chain


if __name__ == "__main__":
    filename = "Inc_Laminar_Flat_Plate/flow.vtu"
    data_dict = extract_vtu_data(filename)
    if data_dict:
        print("Test passed: data_dict extracted successfully.")
        # plot pressure field 
        plt.figure()
        plt.plot(data_dict['Coordinates'][:, 0], data_dict['Coordinates'][:, 1], 'k.')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Flat Plate Mesh')
        plt.show()

        # pressure field plot, plot x,y and data_dict['Pressure']
        plt.figure()
        plt.tricontourf(data_dict['Coordinates'][:, 0], data_dict['Coordinates'][:, 1], data_dict['Pressure'], 100, cmap='jet')
        plt.colorbar()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Pressure Field')
        plt.show()


    else:
        print("Test failed: No data_dict extracted.")