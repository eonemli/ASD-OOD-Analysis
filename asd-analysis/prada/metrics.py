
import numpy as np



def get_data_mapped_counts_list(leaf_metrics):
    """
    Extracts the 'data_mapped_count' values from the leaf metrics dictionary and returns them as a list.
    
    :param leaf_metrics: Dictionary containing metrics for all leaf neurons
    :return: List of 'data_mapped_count' values
    """
    data_mapped_counts = [metrics["data_mapped_count"] for metrics in leaf_metrics.values() if "data_mapped_count" in metrics]
    return data_mapped_counts



def count_total_neurons(neuron):
    """
    Recursively count the total number of neurons in the GHSOM model starting from the given neuron.
    
    :param neuron: The root neuron (zero_unit) or any other neuron in the GHSOM hierarchy
    :return: Total number of neurons in the hierarchy
    """
    # Start counting with the current neuron
    total_neurons = 1

    # If the neuron has a child map, recursively count its child neurons
    if neuron.child_map and hasattr(neuron.child_map, 'neurons'):
        for pos, child_neuron in neuron.child_map.neurons.items():
            total_neurons += count_total_neurons(child_neuron)

    return total_neurons




def calculate_quantization_error(neuron):
    # Get the dataset mapped to this neuron (input_dataset)
    mapped_data = neuron.input_dataset
    if mapped_data is None or len(mapped_data) == 0:
        return np.nan  # If no data is mapped, return NaN
    
    # Get the distances between each data point and the neuron
    distances = neuron.activation(mapped_data)
    
    # Compute the average distance (quantization error)
    quantization_error = distances.mean()
    return quantization_error



# Function to check if two neurons are neighbors based on their positions in the grid
def are_neurons_neighbours(neuron1, neuron2):
    pos1 = neuron1.position
    pos2 = neuron2.position

    # Neurons are neighbors if their positions differ by 1 in one axis and 0 in the other axis
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

# Function to calculate the Euclidean distance between a data point and a neuron’s weight vector
def calculate_distance(data_point, neuron):
    weight_vector = neuron.weight_vector()  # Assuming the neuron has a function to get its weight vector
    return np.linalg.norm(data_point - weight_vector)

# Function to find the first and second BMUs for a given data point
def find_bmu1_and_bmu2(data_point, neuron_map):
    """
    Find the best matching units (BMU1 and BMU2) for a given data point in a map of neurons.
    
    :param data_point: The data point to find BMUs for
    :param neuron_map: The map of neurons (child_map or leaf neuron map)
    """
    bmus = []

    # Calculate distances between the data point and all neurons in the map
    for pos, neuron in neuron_map.neurons.items():
        distance = calculate_distance(data_point, neuron)
        bmus.append((neuron, distance))
    
    # Sort neurons by distance (smallest first)
    bmus = sorted(bmus, key=lambda x: x[1])

    # Return the first and second BMUs
    bmu1 = bmus[0][0]  # Neuron with the smallest distance
    bmu2 = bmus[1][0]  # Neuron with the second smallest distance
    return bmu1, bmu2


# Function to calculate topographic error for a specific neuron using its input dataset
def calculate_topographic_error_for_neuron(neuron, leaf_neuron_map):
    """
    Calculate topographic error for a specific neuron based on the leaf neuron map.
    
    :param neuron: The neuron for which to calculate the topographic error
    :param leaf_neuron_map: The map of all leaf neurons
    """
    # Get the dataset mapped to this neuron
    mapped_data = neuron.input_dataset
    if mapped_data is None or len(mapped_data) == 0:
        return np.nan  # If no data is mapped, return NaN

    topographic_error = 0
    for data_point in mapped_data:
        # Find BMU1 and BMU2 for each data point in the leaf neuron map
        bmu1, bmu2 = find_bmu1_and_bmu2(data_point, leaf_neuron_map)

        # Check if BMU2 is adjacent to BMU1
        if not are_neurons_neighbours(bmu1, bmu2):
            topographic_error += 1

    return topographic_error / len(mapped_data)


def calculate_neighborhood_distances_for_map(neuron_map):
    """
    Calculate the mean weight difference (Euclidean distance) between each neuron in the map and its neighbors.
    The result is an array with the same shape as the map, containing the neighborhood distances for each neuron.
    """
    # Get the map shape
    map_shape = neuron_map.map_shape()
    
    # Initialize an array to store the neighborhood distances, same shape as the map
    neighborhood_distances = np.zeros(map_shape)
    
    # Iterate over all neurons in the map
    for pos, neuron in neuron_map.neurons.items():
        # Get the position of the current neuron
        i, j = neuron.position

        # Find the neighboring positions (top, bottom, left, right, and diagonals)
        neighbors = [(i + di, j + dj) for di in [-1, 0, 1] for dj in [-1, 0, 1]
                     if (di != 0 or dj != 0)  # Exclude the neuron itself
                     and (0 <= i + di < map_shape[0])  # Ensure valid rows
                     and (0 <= j + dj < map_shape[1])]  # Ensure valid columns

        distances = []
        # Calculate distances between the current neuron and its valid neighbors
        for neighbor_pos in neighbors:
            if neighbor_pos in neuron_map.neurons:
                neighbor = neuron_map.neurons[neighbor_pos]
                # Calculate the Euclidean distance between weight vectors
                weight_diff = np.linalg.norm(neuron.weight_vector() - neighbor.weight_vector())
                distances.append(weight_diff)
        
        # Compute the mean of distances and store it in the array
        if distances:
            neighborhood_distances[i, j] = np.mean(distances)
        else:
            neighborhood_distances[i, j] = 0  # If no neighbors, set to 0
    
    return neighborhood_distances



def calculate_mean_inter_neuron_distance(child_map):
    neighborhood_distances_array = calculate_neighborhood_distances_for_map(child_map)
    mean_inter_neuron_distance = np.mean(neighborhood_distances_array)
    return mean_inter_neuron_distance




def is_leaf_neuron(neuron):
    """
    Check if the neuron is a leaf neuron (i.e., it does not have a child map or child neurons).
    A neuron is considered a leaf if it has no child map.
    """
    return neuron.child_map is None  # No child map means it's a leaf neuron


def calculate_metrics_for_leaf_neurons(neuron, leaf_neuron_map, position_list=None, metrics_dict=None):
    """
    Recursively find and calculate metrics for leaf neurons in the GHSOM map.
    
    :param neuron: The neuron to start with (root neuron)
    :param leaf_neuron_map: The map of leaf neurons (i.e., neurons without child maps)
    :param position_list: List of positions to track the full path in the hierarchy
    :param metrics_dict: A dictionary to store metrics for leaf neurons
    
    :return: Dictionary with metrics for all leaf neurons, using position list strings as keys
    """
    if metrics_dict is None:
        metrics_dict = {}
    if position_list is None:
        position_list = []

    # Add current neuron position to the position list
    current_position_list = position_list + [tuple(neuron.position)]
    position_key = ' -> '.join([str(pos) for pos in current_position_list])  # Create a string from the position list

    # Check if the neuron is a leaf neuron
    if is_leaf_neuron(neuron):
        # Get the number of data points mapped to this neuron
        mapped_data_count = len(neuron.input_dataset) if neuron.input_dataset is not None else 0
        
        # Store metrics for this leaf neuron using the position key
        metrics_dict[position_key] = {
            "quantization_error": calculate_quantization_error(neuron),
            "topographic_error": calculate_topographic_error_for_neuron(neuron, leaf_neuron_map),  # Use leaf neuron map
            "neuron_density": mapped_data_count,  # Number of data points mapped to this neuron
            "mean_inter_neuron_distance": calculate_mean_inter_neuron_distance(leaf_neuron_map),  # Use leaf neuron map
            "data_mapped_count": mapped_data_count  # Number of data points mapped
        }
    else:
        # If not a leaf neuron, recursively check its children
        if neuron.child_map and hasattr(neuron.child_map, 'neurons'):
            for pos, child_neuron in neuron.child_map.neurons.items():
                calculate_metrics_for_leaf_neurons(child_neuron, leaf_neuron_map, current_position_list, metrics_dict)

    return metrics_dict


def calculate_leaf_neuron_metrics(train_mapped_zero_unit):
    """
    Calculate metrics for the GHSOM model, focusing on leaf neurons.
    
    :param train_mapped_zero_unit: The root neuron of the GHSOM model with mapped training data
    
    :return: Dictionary of metrics for all leaf neurons, with position lists as keys
    """
    # Use the root neuron map or the appropriate map to find leaf neurons
    leaf_neuron_map = train_mapped_zero_unit.child_map
    metrics_dict = calculate_metrics_for_leaf_neurons(train_mapped_zero_unit, leaf_neuron_map)
    return metrics_dict


