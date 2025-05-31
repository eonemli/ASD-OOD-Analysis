import os
import numpy as np
from multiprocessing import Pool
from functools import partial

# Function to load and collect data from a file
def process_file(filepath,properties):
    # Load data from the .npz file
    data = np.load(filepath, allow_pickle=True)
    
    # Extract the identifier from the file name (without extension)
    identifier = os.path.basename(filepath).replace('.npz', '')
    
    # Create a dictionary with all properties, including the identifier
    result = {prop: data[prop] for prop in properties if prop != 'identifier'}
    result['identifier'] = identifier
    return result

# Function to process a folder
def process_folder(folder, parent_folder, properties=['score_norms', 'identifier']):
    folder_path = os.path.join(parent_folder, folder)
    results = {prop: [] for prop in properties}
    
    # List all files in the folder
    files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".npz")]
    
    # Use multiprocessing to process files in parallel
    with Pool() as pool:
        process_func = partial(process_file, properties=properties)
        data_list = pool.map(process_func, files)
    
    # Collect results
    for data in data_list:
        for prop in properties:
            results[prop].append(data[prop])
    
    # Convert lists to numpy arrays (except for 'identifier', which should remain a list)
    for prop in properties:
        if prop != 'identifier':
            results[prop] = np.array(results[prop])
    
    return results