import copy
import numpy as np
import pickle

def map_test_data_to_neurons(test_data, test_labels, test_identifiers, zero_unit):
    """
    Maps test data, labels, indices, and identifiers into an already trained GHSOM tree. At each level, the parent neuron's 
    test data is mapped into its children. Returns a new structure with the test data, test labels, test data indices, 
    and identifiers as the input dataset, input labels, input indices, and input identifiers for each neuron.
    The training data will be removed from the cloned structure.
    """

    # Helper function to remove input attributes from every neuron
    def remove_input_datasets_labels_indices_identifiers(neuron):
        neuron.input_dataset = None
        neuron.input_labels = None
        neuron.input_indices = None
        neuron.input_identifiers = None

        # Recursively remove input attributes from each child neuron
        if hasattr(neuron, 'child_map') and neuron.child_map is not None:
            for pos, child_neuron in neuron.child_map.neurons.items():
                remove_input_datasets_labels_indices_identifiers(child_neuron)

    # Helper function to recursively map data, labels, indices, and identifiers to neurons and their children
    def map_to_children(data, labels, indices, identifiers, neuron):
        data = np.array(data)
        labels = np.array(labels)
        indices = np.array(indices)
        identifiers = np.array(identifiers)

        # Check if the neuron is a leaf node (no children)
        if not hasattr(neuron, 'child_map') or neuron.child_map is None:
            # Leaf node: Assign the data, labels, indices, and identifiers
            neuron.input_dataset = data if len(data) > 0 else np.array([])
            neuron.input_labels = labels if len(labels) > 0 else np.array([])
            neuron.input_indices = indices if len(indices) > 0 else np.array([])
            neuron.input_identifiers = identifiers if len(identifiers) > 0 else np.array([])
            return neuron

        # Map data, labels, indices, and identifiers to child neurons
        winner_neurons, _ = neuron.child_map.winner_neuron(data)
        data_for_children = {pos: [] for pos in neuron.child_map.neurons.keys()}
        labels_for_children = {pos: [] for pos in neuron.child_map.neurons.keys()}
        indices_for_children = {pos: [] for pos in neuron.child_map.neurons.keys()}
        identifiers_for_children = {pos: [] for pos in neuron.child_map.neurons.keys()}

        # Assign data, labels, indices, and identifiers to the respective child neurons
        for data_point, label, index, identifier, winner in zip(data, labels, indices, identifiers, winner_neurons):
            data_for_children[winner.position].append(data_point)
            labels_for_children[winner.position].append(label)
            indices_for_children[winner.position].append(index)
            identifiers_for_children[winner.position].append(identifier)

        # Recursively map to child neurons
        for pos, child_neuron in neuron.child_map.neurons.items():
            if data_for_children[pos]:
                data_for_children[pos] = np.array(data_for_children[pos])
                labels_for_children[pos] = np.array(labels_for_children[pos])
                indices_for_children[pos] = np.array(indices_for_children[pos])
                identifiers_for_children[pos] = np.array(identifiers_for_children[pos])
                neuron.child_map.neurons[pos] = map_to_children(
                    data_for_children[pos], labels_for_children[pos], indices_for_children[pos], identifiers_for_children[pos], child_neuron
                )
            else:
                # No data mapped: set empty arrays
                child_neuron.input_dataset = np.array([])
                child_neuron.input_labels = np.array([])
                child_neuron.input_indices = np.array([])
                child_neuron.input_identifiers = np.array([])

        # Assign input attributes to the current neuron
        neuron.input_dataset = data
        neuron.input_labels = labels
        neuron.input_indices = indices
        neuron.input_identifiers = identifiers

        return neuron

    # Step 1: Create a deep copy of the entire GHSOM structure
    cloned_neuron_tree = copy.deepcopy(zero_unit)

    # Step 2: Remove input attributes from every neuron in the cloned tree
    remove_input_datasets_labels_indices_identifiers(cloned_neuron_tree)

    # Step 3: Convert the test_data, test_labels, test_identifiers, and their indices into NumPy arrays
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    test_identifiers = np.array(test_identifiers)
    test_indices = np.arange(len(test_data))  # Create an array of indices for the test data

    # Step 4: Start the process of mapping the data, labels, indices, and identifiers
    return map_to_children(test_data, test_labels, test_indices, test_identifiers, cloned_neuron_tree)






def load_and_map_zero_units(zu_dir ,file_name, log_scale_t1, log_scale_t2, inlier_data, train_labels, inlier_identifiers, test_data, test_labels, test_identifiers):
    """
    Load trained zero unit maps, map training and test data, and store the results for each t1 and t2 combination.
    
    :param log_scale: List of t1 and t2 values to use for loading zero units
    :param inlier_data: Training data to map to the zero units
    :param train_labels: Labels corresponding to the training data
    :param test_data: Test data to map to the zero units
    :param test_labels: Labels corresponding to the test data
    :return: Dictionary storing train_mapped_zero_unit and test_mapped_zero_unit for each (t1, t2) combination
    """
    mapped_units = {}  # Dictionary to store mapped units
    
    # Iterate over t1 and t2 values
    for i, t1 in enumerate(log_scale_t1):
        for j, t2 in enumerate(log_scale_t2):
            t1_r = round(t1, 3)
            t2_r = round(t2, 3)

            zero_unit_path = f'{zu_dir}/{t1_r}_{t2_r}_{file_name}'
            
            # Load zero_unit
            with open(zero_unit_path, 'rb') as f:
                zero_unit = pickle.load(f)
            print(f"zero_unit object loaded from: ==>    {zero_unit_path}")

            # Map the training data to the zero unit
            train_mapped_zero_unit = map_test_data_to_neurons(inlier_data, train_labels, inlier_identifiers, zero_unit)
            
            # Map the test data to the zero unit
            test_mapped_zero_unit = map_test_data_to_neurons(test_data, test_labels, test_identifiers, zero_unit)
            
            # Store the mapped units in the dictionary using (t1_r, t2_r) as the key
            mapped_units[(t1_r, t2_r)] = {
                "train_mapped_zero_unit": train_mapped_zero_unit,
                "test_mapped_zero_unit": test_mapped_zero_unit,
                "zero_unit": zero_unit
            }

    return mapped_units



