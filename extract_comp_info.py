import re
import glob
import os

def extract_farfield_conditions(output_folder_path, comp_path, verbose=False):
    current_path = os.getcwd()
    output_path = []
    # get the list of file name for output files 
    if comp_path is None:
        KeyError("comp_path is None")
    else:
        for vtu_file_name in comp_path:
            file_name_no_ext = os.path.splitext(vtu_file_name)[0]
            file_name_base = os.path.basename(file_name_no_ext)
            file_name_output = current_path + "/" + output_folder_path + "output" + file_name_base[9:] + ".0.o"
            output_path.append(file_name_output)

    # initialize a list to store dictorary of extracted values
    ff_list = []

    # loop through the list of file path
    for file_path in output_path:
        # Define a dictionary to hold the extracted values
        extracted_values = {
            "Static Pressure": None,
            "Density": None,
            "Velocity-X": None,
            "Total Energy": None
        }

        # Define the pattern to search for the values
        pattern = re.compile(r'(\w[\w\s-]+)\|\s+([\d.e+-]+)')

        # Open the file and read line by line
        with open(file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    key, value = match.groups()
                    key = key.strip()
                    # Check if the key is one of the fields we're interested in
                    if key in extracted_values:
                        extracted_values[key] = float(value)
        # add mach and back pressure to extracted values
        file_pattern = re.compile(r'_(\d+\.\d+)_(\d+\.\d+)\.o$')
        # Search for the pattern in the file path
        match = file_pattern.search(file_path)

        if match:
            # Extract the two numbers
            number1, number2 = match.groups()
            number1 = float(number1)
            number2 = float(number2)
            
        extracted_values["Mach"] = number1
        extracted_values["BackPres"] = number2          
        if verbose:
            # Print the extracted values
            for key, value in extracted_values.items():
                print(f"{key}: {value}")
        # append the extracted values to the list
        ff_list.append(extracted_values)        
    return ff_list

if __name__ == "__main__":
    extract_farfield_conditions("all_comp/output")
