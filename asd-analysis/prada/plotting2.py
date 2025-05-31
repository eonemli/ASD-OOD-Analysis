from .metrics import calculate_neighborhood_distances_for_map

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as path_effects
from collections import Counter
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap



# Define markers and colors for test data categories
markers = {
    "Down Syndrome": "v",  # triangle
    "IBIS-ASD": "*",       # star
    "ABCD-ASD": "*",       # star
    "Conte": "^",       # triangle up
    "HR-Typical": "o",     # circle
    "Atypical": "D"        # diamond
}
colors = {
    "Down Syndrome": "#e41a1c",  # red
    "HR-Typical": "#7fc97f",     # green
    "IBIS-ASD": "#fdc076",       # orange
    "ABCD-ASD": "#fd76f4",       # pink
    "Conte": "#ffff99",          # yellow
    "Atypical": "#a6cee3"        # blue
}




# Truncate the colormap to focus on lighter tones
def truncate_colormap(cmap, minval=0.5, maxval=1.0, n=256):
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(maxval, minval, n))
    )
    return new_cmap

# Use a truncated colormap focusing on lighter tones of "Blues"
colormap_defined = truncate_colormap(plt.cm.Purples, minval=0.05, maxval=0.5)




def print_tree_structure(neuron, level=0):
    """
    Recursively prints the structure of the neuron tree, showing the number of data points 
    mapped to each neuron at each level.
    """
    # Indentation based on the level to visually represent the hierarchy
    indent = "  |  " * level
    
    # Check if the neuron has a child map
    if hasattr(neuron, 'child_map') and neuron.child_map is not None:
        # Print the current neuron's information
        print(f"{indent}Level: {level}, Position: {neuron.position}, Child map size: {neuron.child_map.map_shape()}, Number of data: {len(neuron.input_dataset) if neuron.input_dataset is not None else 0}")
        
        # Recursively print the structure for child neurons
        for i in range(neuron.child_map.map_shape()[0]):
            for j in range(neuron.child_map.map_shape()[1]):
                print_tree_structure(neuron.child_map.neurons[(i, j)], level + 1)
    else:
        # If the neuron is a leaf node, only print its information
        print(f"{indent}Level: {level}, Position: {neuron.position}, Number of data: {len(neuron.input_dataset) if neuron.input_dataset is not None else 0}")


# reach a specific neuron on the map by its position on the map.
def get_neuron_by_position_list(zero_unit, position_list):
    current_neuron = zero_unit
    for position in position_list:
        if hasattr(current_neuron, 'child_map') and current_neuron.child_map is not None:
            current_neuron = current_neuron.child_map.neurons.get(position)
        else:
            raise ValueError(f"Neuron at {current_neuron.position} does not have children.")
    return current_neuron


def plot_map_with_quad(neuron_map, dataset_id_to_name):
    """
    Plot the neuron map using quadmesh, where each square represents a neuron.
    The color intensity of the square represents the neighborhood distance.
    The number of data points, positions, and class distributions are displayed for each neuron.
    """
    # Get the neighborhood distances array
    neighborhood_distances = calculate_neighborhood_distances_for_map(neuron_map)
    
    # Get the map shape
    map_shape = neuron_map.map_shape()

    # Create a meshgrid for plotting, shifted by 0.5 to center the quads
    x, y = np.meshgrid(np.arange(map_shape[1]+1), np.arange(map_shape[0]+1))

    # Plot the neuron grid using quadmesh
    fig, ax = plt.subplots(figsize=(map_shape[1]*3, map_shape[0]*2.3))
    cmap = plt.cm.gist_gray  # Use grayscale color map
    c = ax.pcolormesh(x - 0.5, y - 0.5, neighborhood_distances, cmap=cmap, shading='auto', alpha=0.8, edgecolor='gray')

    # Loop through the neurons and display data
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            if (i, j) in neuron_map.neurons:
                neuron = neuron_map.neurons[(i, j)]
                if neuron.input_dataset is None:
                    num_data = 0
                else:
                    num_data = len(neuron.input_dataset)

                # Count the occurrences of each class label
                class_counts = {}
                if neuron.input_labels is not None:
                    for label in neuron.input_labels:
                        class_counts[label] = class_counts.get(label, 0) + 1

                # Get the dominant class (class with the most occurrences)
                dominant_class_label = max(class_counts, key=class_counts.get) if class_counts else None

                # Format the class information
                class_info = ""
                for label, count in class_counts.items():
                    class_name = dataset_id_to_name.get(label, "Unknown")
                    # Bold the dominant class
                    if label == dominant_class_label:
                        class_info += f"*{class_name}: {count}\n"
                    else:
                        class_info += f"{class_name}: {count}\n"

                # Add the text at the center of the square (data count + class info)
                ax.text(j, i, f"data total: {num_data}\n\n{class_info}\npos: ({i},{j})", 
                        ha='center', va='center', fontsize=10, color='red').set_path_effects([
                    path_effects.Stroke(linewidth=2, foreground='white'),  # The shadow (gray outline)
                    path_effects.Normal()  # The actual text
                ])

    # Set axis tick locators to force integer labels
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust tick positions to align with quad centers
    ax.set_xticks(np.arange(map_shape[1]), minor=False)
    ax.set_yticks(np.arange(map_shape[0]), minor=False)

    # Add a colorbar for neighborhood distances
    plt.colorbar(c, ax=ax, label="Neighborhood Distance")

    # Invert y-axis to match the matrix layout with top left corner as (0, 0)
    ax.invert_yaxis()

    # Add a title that includes the map shape
    plt.title(f"Neuron Map - Shape: {map_shape}", fontsize=16)

    plt.show()




def plot_map_with_test_data(train_neuron_map, test_neuron_map, dataset_id_to_name, markers, colors):
    """
    Plot the training neuron map using quadmesh with neighborhood distances, and overlay class markers for test data.
    Each neuron will show test data class distributions with corresponding markers, class names, and sample sizes.
    """
    # Get the neighborhood distances array from the training neurons
    neighborhood_distances = calculate_neighborhood_distances_for_map(train_neuron_map)
    
    # Get the map shape from the training neuron map
    map_shape = train_neuron_map.map_shape()

    # Create a meshgrid for plotting, shifted by 0.5 to center the quads
    x, y = np.meshgrid(np.arange(map_shape[1]+1), np.arange(map_shape[0]+1))

    # Plot the neuron grid using quadmesh (training data neighborhood distances)
    fig, ax = plt.subplots(figsize=(map_shape[1]*3, map_shape[0]*2.3))
    cmap = plt.cm.gist_gray  # Use grayscale color map
    c = ax.pcolormesh(x - 0.5, y - 0.5, neighborhood_distances, cmap=cmap, shading='auto', alpha=0.8, edgecolor='gray')

    # markers and colors can be re-defined here.
    # markers = { "Down Syndrome": "v", ... # triangle }
    # colors = { "Down Syndrome": "#e41a1c", ... # red }
    
    # Loop through the neurons and overlay test data class information
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            if (i, j) in train_neuron_map.neurons:
                # Overlay test data information (test class counts)
                if (i, j) in test_neuron_map.neurons:
                    test_neuron = test_neuron_map.neurons[(i, j)]
                    
                    # Ensure that test_neuron.input_dataset is not None before proceeding
                    if test_neuron.input_dataset is not None and len(test_neuron.input_dataset) > 0:
                        # Count the occurrences of each class label (test data)
                        test_class_counts = {}
                        for label in test_neuron.input_labels:
                            test_class_counts[label] = test_class_counts.get(label, 0) + 1

                        total_data = len(test_neuron.input_dataset)  # Total number of test data points in this neuron

                        # Display the markers, class names, and corresponding test sample sizes
                        y_offset = -0.20  # Start slightly above the center
                        for label, count in test_class_counts.items():
                            class_name = dataset_id_to_name.get(label, "Unknown")
                            marker = markers.get(class_name, "o")
                            color = colors.get(class_name, "black")

                            # Plot the marker at the neuron position with a small offset
                            ax.scatter(j - 0.3, i + y_offset, marker=marker, color=color, s=100, edgecolor='gray')

                            # Add class name and count next to the marker, closer to the center
                            ax.text(j - 0.18, i + y_offset, f"{class_name[:8]}: {count}", ha='left', va='center', fontsize=10, color='white').set_path_effects([
                                path_effects.Stroke(linewidth=2, foreground='red'),  # The shadow (gray outline)
                                path_effects.Normal()  # The actual text
                            ])

                            # Increase the y_offset for the next marker, ensuring it stays within the neuron quad
                            y_offset += 0.12

                        # Write the total number of test data points in the center of the quad
                        ax.text(j, i - 0.37, f"Total: {total_data}", ha='center', va='center', fontsize=10, color='magenta').set_path_effects([
                                path_effects.Stroke(linewidth=2, foreground='white'),  # The shadow (gray outline)
                                path_effects.Normal()  # The actual text
                            ])

    # Set axis tick locators to force integer labels
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust tick positions to align with quad centers
    ax.set_xticks(np.arange(map_shape[1]), minor=False)
    ax.set_yticks(np.arange(map_shape[0]), minor=False)

    # Add a colorbar for neighborhood distances
    plt.colorbar(c, ax=ax, label="Neighborhood Distance")

    # Invert y-axis to match the matrix layout with top left corner as (0, 0)
    ax.invert_yaxis()

    # Add a title that includes the map shape
    plt.title(f"Neuron Map with Test Data Overlay - Shape: {map_shape}", fontsize=16)

    # Add a legend for the test data markers
    handles = [plt.Line2D([0], [0], marker=markers[k], color='w', markerfacecolor=colors[k], label=k, markersize=10, linestyle='None') 
               for k in markers]
    #ax.legend(handles=handles, title="Test Data Categories")

    plt.show()



def plot_map_with_test_data_significant_bmus(train_neuron_map, test_neuron_map, dataset_id_to_name, percent_threshold=10, markers=None, colors=None):
    """
    Plot the training neuron map using quadmesh with neighborhood distances, and overlay class markers for test data.
    A neuron is considered a significant BMU if it contains at least percent_threshold (e.g., 10%) of the data
    from any individual category.
    """
    
    # Get the total number of test data points for each category
    total_category_counts = {}
    for neuron in test_neuron_map.neurons.values():
        if neuron.input_labels is not None:
            for label in neuron.input_labels:
                total_category_counts[label] = total_category_counts.get(label, 0) + 1

    # Get the neighborhood distances array from the training neurons
    neighborhood_distances = calculate_neighborhood_distances_for_map(train_neuron_map)
    
    # Get the map shape from the training neuron map
    map_shape = train_neuron_map.map_shape()

    # Create a meshgrid for plotting, shifted by 0.5 to center the quads
    x, y = np.meshgrid(np.arange(map_shape[1]+1), np.arange(map_shape[0]+1))

    # Plot the neuron grid using quadmesh (training data neighborhood distances)
    fig, ax = plt.subplots(figsize=(map_shape[1]*3, map_shape[0]*2.3))
    cmap = plt.cm.gist_gray  # Use grayscale color map
    c = ax.pcolormesh(x - 0.5, y - 0.5, neighborhood_distances, cmap=cmap, shading='auto', alpha=0.8, edgecolor='gray')

    # markers and colors can be re-defined here.
    # markers = { "Down Syndrome": "v", ... # triangle }
    # colors = { "Down Syndrome": "#e41a1c", ... # red }

    # Loop through the neurons and overlay test data class information
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            if (i, j) in train_neuron_map.neurons:
                # Overlay test data information only if the test neuron exists
                if (i, j) in test_neuron_map.neurons:
                    test_neuron = test_neuron_map.neurons[(i, j)]

                    # Ensure that test_neuron.input_dataset is not None and has data before proceeding
                    if test_neuron.input_dataset is not None and test_neuron.input_dataset.size > 0:
                        # Count the occurrences of each class label (test data)
                        test_class_counts = {}
                        for label in test_neuron.input_labels:
                            test_class_counts[label] = test_class_counts.get(label, 0) + 1

                        # Check if any category has more than percent_threshold of its total number of points
                        significant_bmu = False
                        for label, count in test_class_counts.items():
                            total_category_count = total_category_counts.get(label, 0)
                            if total_category_count > 0 and count >= (total_category_count * (percent_threshold / 100)):
                                significant_bmu = True
                                break

                        # If this is a significant BMU, overlay markers
                        if significant_bmu:
                            y_offset = -0.20  # Start slightly above the center
                            for label, count in test_class_counts.items():
                                class_name = dataset_id_to_name.get(label, "Unknown")
                                marker = markers.get(class_name, "o")
                                color = colors.get(class_name, "black")

                                # Plot the marker at the neuron position with a small offset
                                ax.scatter(j - 0.3, i + y_offset, marker=marker, color=color, s=100, edgecolor='gray')

                                # Add class name and count next to the marker, closer to the center
                                ax.text(j - 0.18, i + y_offset, f"{class_name[:8]}: {count}", ha='left', va='center', fontsize=10, color='white').set_path_effects([
                                    path_effects.Stroke(linewidth=2, foreground='red'),  # The shadow (gray outline)
                                    path_effects.Normal()  # The actual text
                                ])

                                # Increase the y_offset for the next marker, ensuring it stays within the neuron quad
                                y_offset += 0.12

                            # Write the total number of test data points in the center of the quad
                            ax.text(j, i - 0.37, f"Total: {len(test_neuron.input_dataset)}", ha='center', va='center', fontsize=10, color='magenta').set_path_effects([
                                    path_effects.Stroke(linewidth=2, foreground='white'),  # The shadow (gray outline)
                                    path_effects.Normal()  # The actual text
                                ])

    # Set axis tick locators to force integer labels
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust tick positions to align with quad centers
    ax.set_xticks(np.arange(map_shape[1]), minor=False)
    ax.set_yticks(np.arange(map_shape[0]), minor=False)

    # Add a colorbar for neighborhood distances
    plt.colorbar(c, ax=ax, label="Neighborhood Distance")

    # Invert y-axis to match the matrix layout with top left corner as (0, 0)
    ax.invert_yaxis()

    # Add a title that includes the map shape
    plt.title(f"Neuron Map with Significant BMUs - Shape: {map_shape}", fontsize=16)

    # Add a legend for the test data markers
    handles = [plt.Line2D([0], [0], marker=markers[k], color='w', markerfacecolor=colors[k], label=k, markersize=10, linestyle='None') 
               for k in markers]
    # ax.legend(handles=handles, title="Test Data Categories")

    plt.show()




def plot_map_with_test_data_markers(train_neuron_map, test_neuron_map, dataset_id_to_name, percent_threshold=10, markers=None, colors=None):
    """
    Plot the training neuron map using quadmesh with neighborhood distances, and overlay class markers for test data.
    Marker sizes will be proportional to the class sizes in each neuron.
    A neuron is considered significant if it contains at least percent_threshold (e.g., 10%) of the data from any category.
    """
    
    # Get the total number of test data points for each category and calculate the max points for normalization
    total_category_counts = {}
    max_category_points = {}  # Track the max number of points in any neuron for each category for normalization
    for neuron in test_neuron_map.neurons.values():
        if neuron.input_labels is not None:
            for label in neuron.input_labels:
                total_category_counts[label] = total_category_counts.get(label, 0) + 1
                max_category_points[label] = max(max_category_points.get(label, 0), np.count_nonzero(neuron.input_labels == label))

    # Get the neighborhood distances array from the training neurons
    neighborhood_distances = calculate_neighborhood_distances_for_map(train_neuron_map)
    
    # Get the map shape from the training neuron map
    map_shape = train_neuron_map.map_shape()

    # Create a meshgrid for plotting, shifted by 0.5 to center the quads
    x, y = np.meshgrid(np.arange(map_shape[1]+1), np.arange(map_shape[0]+1))

    # Plot the neuron grid using quadmesh (training data neighborhood distances)
    fig, ax = plt.subplots(figsize=(map_shape[1]*3, map_shape[0]*2.3))
    cmap = plt.cm.gist_gray  # Use grayscale color map
    c = ax.pcolormesh(x - 0.5, y - 0.5, neighborhood_distances, cmap=cmap, shading='auto', alpha=0.8, edgecolor='gray')

    # markers and colors can be re-defined here.
    # markers = { "Down Syndrome": "v", ... # triangle }
    # colors = { "Down Syndrome": "#e41a1c", ... # red }

    # Define a base marker size
    base_marker_size = 1500  # Adjusted marker size for better visibility

    # Loop through the neurons and overlay test data class information
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            if (i, j) in train_neuron_map.neurons:
                # Overlay test data information only if the test neuron exists
                if (i, j) in test_neuron_map.neurons:
                    test_neuron = test_neuron_map.neurons[(i, j)]

                    # Ensure that test_neuron.input_dataset is not None and has data before proceeding
                    if test_neuron.input_dataset is not None and test_neuron.input_dataset.size > 0:
                        # Count the occurrences of each class label (test data)
                        test_class_counts = {}
                        for label in test_neuron.input_labels:
                            test_class_counts[label] = test_class_counts.get(label, 0) + 1

                        # Check if any category has more than percent_threshold of its total number of points
                        significant_bmu = False
                        for label, count in test_class_counts.items():
                            total_category_count = total_category_counts.get(label, 0)
                            if total_category_count > 0 and count >= (total_category_count * (percent_threshold / 100)):
                                significant_bmu = True
                                break

                        # If this is a significant BMU, overlay markers
                        if significant_bmu:
                            x_offset = 0.04  # Start slightly to the right of the center
                            y_offset = -0.19  # Start slightly above the center
                            for label, count in test_class_counts.items():
                                class_name = dataset_id_to_name.get(label, "Unknown")
                                marker = markers.get(class_name, "o")
                                color = colors.get(class_name, "black")

                                # Normalize the marker size based on the max number of points for the specific category
                                max_points_for_category = max_category_points.get(label, 1)  # Avoid division by zero
                                marker_size = (count / max_points_for_category) * base_marker_size  # Scale by category

                                # Plot the marker at the neuron position with a small offset
                                ax.scatter(j - 0.3 + x_offset, i + y_offset, marker=marker, color=color, s=marker_size, edgecolor='gray')

                                # Increase the y_offset for the next marker, ensuring it stays within the neuron quad
                                y_offset += 0.11
                                x_offset += 0.11

                            # Write the total number of test data points in the center of the quad
                            ax.text(j, i - 0.37, f"Total: {len(test_neuron.input_dataset)}", ha='center', va='center', fontsize=12, color='magenta').set_path_effects([
                                    path_effects.Stroke(linewidth=2, foreground='white'),  # The shadow (gray outline)
                                    path_effects.Normal()  # The actual text
                                ])

    # Set axis tick locators to force integer labels
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust tick positions to align with quad centers
    ax.set_xticks(np.arange(map_shape[1]), minor=False)
    ax.set_yticks(np.arange(map_shape[0]), minor=False)

    # Add a colorbar for neighborhood distances
    plt.colorbar(c, ax=ax, label="Neighborhood Distance")

    # Invert y-axis to match the matrix layout with top left corner as (0, 0)
    ax.invert_yaxis()

    # Add a title that includes the map shape
    plt.title(f"Neuron Map with Significant BMUs - Shape: {map_shape}", fontsize=16)

    # Add a legend for the test data markers
    handles = [plt.Line2D([0], [0], marker=markers[k], color='w', markerfacecolor=colors[k], label=k, markersize=10, linestyle='None') 
            for k in markers]
    ax.legend(handles=handles, title="Test Data Categories", loc='center', bbox_to_anchor=(0.5, 1.15), ncol=len(markers), frameon=False)

    plt.show()



def plot_map_with_ibis_asd_samples(train_neuron_map, test_neuron_map, dataset_id_to_name, colormap_defined):
    """
    Plot the training neuron map using quadmesh with neighborhood distances, and overlay small markers for ASD samples.
    Markers will be scattered inside their respective nodes to avoid masking each other.
    """
    
    # Get the neighborhood distances array from the training neurons
    neighborhood_distances = calculate_neighborhood_distances_for_map(train_neuron_map)
    
    # Get the map shape from the training neuron map
    map_shape = train_neuron_map.map_shape()

    # Create a meshgrid for plotting, shifted by 0.5 to center the quads
    x, y = np.meshgrid(np.arange(map_shape[1]+1), np.arange(map_shape[0]+1))

    # Plot the neuron grid using quadmesh (training data neighborhood distances)
    fig, ax = plt.subplots(figsize=(map_shape[1]*1.5, map_shape[0]*1.15))
    cmap = colormap_defined  # Use truncated colormap focusing on lighter tones
    c = ax.pcolormesh(x - 0.5, y - 0.5, neighborhood_distances, cmap=cmap, shading='auto', alpha=0.8, edgecolor='gray', linewidth=0.4)

    # Set the outer linewidth of the plot
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_edgecolor('gray')

    # Define marker and color for ASD samples
    marker = "*"
    color = "#ffd700"  # yellow

    # Loop through the neurons and overlay ASD sample markers
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            if (i, j) in train_neuron_map.neurons:
                # Overlay ASD sample markers only if the test neuron exists
                if (i, j) in test_neuron_map.neurons:
                    test_neuron = test_neuron_map.neurons[(i, j)]

                    # Ensure that test_neuron.input_dataset is not None and has data before proceeding
                    if test_neuron.input_dataset is not None and test_neuron.input_dataset.size > 0:
                        # Filter ASD samples
                        asd_indices = [idx for idx, label in enumerate(test_neuron.input_labels) if dataset_id_to_name.get(label) == "IBIS-ASD"]
                        asd_samples = test_neuron.input_dataset[asd_indices]

                        # Scatter small markers for each ASD sample inside the node
                        for sample in asd_samples:
                            jitter_x = np.random.uniform(-0.4, 0.4)
                            jitter_y = np.random.uniform(-0.4, 0.4)
                            ax.scatter(j + jitter_x, i + jitter_y, marker=marker, color=color, s=100, edgecolor='black', linewidths=0.3)

    # Set axis tick locators to force integer labels
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust tick positions to align with quad centers
    ax.set_xticks(np.arange(map_shape[1]), minor=False)
    ax.set_yticks(np.arange(map_shape[0]), minor=False)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a colorbar for neighborhood distances
    plt.colorbar(c, ax=ax, label="Neighborhood Distance")

    # Invert y-axis to match the matrix layout with top left corner as (0, 0)
    ax.invert_yaxis()

    # Add a title that includes the map shape
    #plt.title(f"Neuron Map with ASD Samples - Shape: {map_shape}", fontsize=16)

    plt.show()




def plot_map_with_selected_asd_samples(train_neuron_map, test_neuron_map, dataset_id_to_name, key_list, marker_size, colormap_defined):
    """
    Plot the training neuron map using quadmesh with neighborhood distances, and overlay small markers for selected ASD samples.
    Markers will be scattered inside their respective nodes to avoid masking each other.
    """
    
    # Get the neighborhood distances array from the training neurons
    neighborhood_distances = calculate_neighborhood_distances_for_map(train_neuron_map)
    
    # Get the map shape from the training neuron map
    map_shape = train_neuron_map.map_shape()

    # Create a meshgrid for plotting, shifted by 0.5 to center the quads
    x, y = np.meshgrid(np.arange(map_shape[1]+1), np.arange(map_shape[0]+1))

    # Plot the neuron grid using quadmesh (training data neighborhood distances)
    fig, ax = plt.subplots(figsize=(map_shape[1]*1.5, map_shape[0]*1.15))
    cmap = colormap_defined  # Use truncated colormap focusing on lighter tones
    c = ax.pcolormesh(x - 0.5, y - 0.5, neighborhood_distances, cmap=cmap, shading='auto', alpha=0.8, edgecolor='gray', linewidth=0.4)

    # Set the outer linewidth of the plot
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_edgecolor('gray')

    # Define marker and color for ASD samples
    marker = "*"
    #color = "#ffd700"  # yellow
    color = "#eb4e00" # dark orange
    # Loop through the neurons and overlay selected ASD sample markers
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            if (i, j) in train_neuron_map.neurons:
                # Overlay selected ASD sample markers only if the test neuron exists
                if (i, j) in test_neuron_map.neurons:
                    test_neuron = test_neuron_map.neurons[(i, j)]

                    # Ensure that test_neuron.input_dataset is not None and has data before proceeding
                    if test_neuron.input_dataset is not None and test_neuron.input_dataset.size > 0:
                        # Filter selected ASD samples
                        selected_asd_indices = [idx for idx, identifier in enumerate(test_neuron.input_identifiers) if identifier in key_list and dataset_id_to_name.get(test_neuron.input_labels[idx]) == "IBIS-ASD"]
                        selected_asd_samples = test_neuron.input_dataset[selected_asd_indices]

                        # Scatter small markers for each selected ASD sample inside the node
                        for sample in selected_asd_samples:
                            jitter_x = np.random.uniform(-0.4, 0.4)
                            jitter_y = np.random.uniform(-0.4, 0.4)
                            ax.scatter(j + jitter_x, i + jitter_y, marker=marker, color=color, s=marker_size, edgecolor='black', linewidths=0.3)

    # Set axis tick locators to force integer labels
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust tick positions to align with quad centers
    ax.set_xticks(np.arange(map_shape[1]), minor=False)
    ax.set_yticks(np.arange(map_shape[0]), minor=False)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a colorbar for neighborhood distances
    plt.colorbar(c, ax=ax, label="Neighborhood Distance")

    # Invert y-axis to match the matrix layout with top left corner as (0, 0)
    ax.invert_yaxis()

    # Add a title that includes the map shape
    #plt.title(f"Neuron Map with Selected ASD Samples - Shape: {map_shape}", fontsize=16)

    plt.show()


def plot_map_with_abcd_asd_samples(train_neuron_map, test_neuron_map, dataset_id_to_name, key_list, marker_size, colormap_defined):
    """
    Plot the training neuron map using quadmesh with neighborhood distances, and overlay small markers for selected ASD samples.
    Markers will be scattered inside their respective nodes to avoid masking each other.
    """
    
    # Get the neighborhood distances array from the training neurons
    neighborhood_distances = calculate_neighborhood_distances_for_map(train_neuron_map)
    
    # Get the map shape from the training neuron map
    map_shape = train_neuron_map.map_shape()

    # Create a meshgrid for plotting, shifted by 0.5 to center the quads
    x, y = np.meshgrid(np.arange(map_shape[1]+1), np.arange(map_shape[0]+1))

    # Plot the neuron grid using quadmesh (training data neighborhood distances)
    fig, ax = plt.subplots(figsize=(map_shape[1]*1.5, map_shape[0]*1.15))
    cmap = colormap_defined  # Use truncated colormap focusing on lighter tones
    c = ax.pcolormesh(x - 0.5, y - 0.5, neighborhood_distances, cmap=cmap, shading='auto', alpha=0.8, edgecolor='gray', linewidth=0.4)

    # Set the outer linewidth of the plot
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_edgecolor('gray')

    # Define marker and color for ASD samples
    marker = "*"
    #color = "#ffd700"  # yellow
    color = "#fd76f4" # pink
    # Loop through the neurons and overlay selected ASD sample markers
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            if (i, j) in train_neuron_map.neurons:
                # Overlay selected ASD sample markers only if the test neuron exists
                if (i, j) in test_neuron_map.neurons:
                    test_neuron = test_neuron_map.neurons[(i, j)]

                    # Ensure that test_neuron.input_dataset is not None and has data before proceeding
                    if test_neuron.input_dataset is not None and test_neuron.input_dataset.size > 0:
                        # Filter selected ASD samples
                        # selected_asd_indices = [idx for idx, identifier in enumerate(test_neuron.input_identifiers) if identifier in key_list and dataset_id_to_name.get(test_neuron.input_labels[idx]) == "IBIS-ASD"]
                        # selected_asd_samples = test_neuron.input_dataset[selected_asd_indices]

                        # Filter ASD samples
                        selected_asd_indices = [idx for idx, label in enumerate(test_neuron.input_labels) if dataset_id_to_name.get(label) == "ABCD-ASD"]
                        selected_asd_samples = test_neuron.input_dataset[selected_asd_indices]

                        # Scatter small markers for each selected ASD sample inside the node
                        for sample in selected_asd_samples:
                            jitter_x = np.random.uniform(-0.4, 0.4)
                            jitter_y = np.random.uniform(-0.4, 0.4)
                            ax.scatter(j + jitter_x, i + jitter_y, marker=marker, color=color, s=marker_size, edgecolor='black', linewidths=0.3)

    # Set axis tick locators to force integer labels
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust tick positions to align with quad centers
    ax.set_xticks(np.arange(map_shape[1]), minor=False)
    ax.set_yticks(np.arange(map_shape[0]), minor=False)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a colorbar for neighborhood distances
    plt.colorbar(c, ax=ax, label="Neighborhood Distance")

    # Invert y-axis to match the matrix layout with top left corner as (0, 0)
    ax.invert_yaxis()

    # Add a title that includes the map shape
    #plt.title(f"Neuron Map with Selected ASD Samples - Shape: {map_shape}", fontsize=16)

    plt.show()



def plot_map_with_selected_asd_samples_batches(train_neuron_map, test_neuron_map, dataset_id_to_name, key_list, ibis_asd_site_dict, ibis_asd_sex_dict, marker_size, colormap_defined):
    """
    Plot the training neuron map with neighborhood distances using quadmesh,
    overlay selected ASD sample markers inside the nodes, where marker shape
    is determined by sex (using ibis_asd_sex_dict) and marker color by scanner
    (using ibis_asd_site_dict). Only "Male" and "Female" labels are used.
    
    After the main plot, a separate figure shows the marker legend for sex.
    """
    # Get the neighborhood distances array from the training neurons
    neighborhood_distances = calculate_neighborhood_distances_for_map(train_neuron_map)
    
    # Get the map shape from the training neuron map
    map_shape = train_neuron_map.map_shape()

    # Create a meshgrid for plotting (shifted by 0.5 to center the quads)
    x, y = np.meshgrid(np.arange(map_shape[1] + 1), np.arange(map_shape[0] + 1))

    # Plot the neuron grid using quadmesh (training data neighborhood distances)
    fig, ax = plt.subplots(figsize=(map_shape[1] * 1.5, map_shape[0] * 1.15))
    cmap = colormap_defined  # Using your truncated colormap focusing on lighter tones
    c = ax.pcolormesh(x - 0.5, y - 0.5, neighborhood_distances, cmap=cmap, shading='auto',
                      alpha=0.8, edgecolor='gray', linewidth=0.4)

    # Set the outer linewidth and color for the spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_edgecolor('gray')
    
    # --- Define mapping dictionaries for marker shapes and scanner colors ---
    # Marker shapes based only on existing sex labels ("Male" and "Female")
    # sex_marker_map = {"Male": "*", "Female": "o"}
    # Scanner color mapping as specified:
    site_color_map = {
            "UNC"  :  "blue",
            "STL"  :  "green",
            "SEA"  :  "red",
            "PHI"  :  "purple"
    }

    # Loop through neurons to overlay selected ASD sample markers
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            if (i, j) in train_neuron_map.neurons:
                # Overlay markers only if the test neuron exists in this node
                if (i, j) in test_neuron_map.neurons:
                    test_neuron = test_neuron_map.neurons[(i, j)]
                    # Check that the input_dataset is non-empty
                    if test_neuron.input_dataset is not None and test_neuron.input_dataset.size > 0:
                        # Select indices for ASD samples from this neuron
                        selected_asd_indices = [
                            idx for idx, identifier in enumerate(test_neuron.input_identifiers)
                            if identifier in key_list and dataset_id_to_name.get(test_neuron.input_labels[idx]) == "IBIS-ASD"
                        ]
                        
                        # Loop over selected indices to plot each sample marker
                        for idx in selected_asd_indices:
                            # Get identifier for the sample
                            identifier = test_neuron.input_identifiers[idx]
                            
                            # Look up the sex using ibis_asd_sex_dict; default to "Female" if not found
                            # sex_val = ibis_asd_sex_dict.get(identifier)
                            # marker = sex_marker_map.get(sex_val)
                            # Ignore sex entirely:
                            marker = "*"
                            # Look up scanner label and assign color (default to black if not found)
                            scanner_label = ibis_asd_site_dict.get(identifier)
                            color = site_color_map.get(scanner_label)
                            
                            # Plot the marker with some random jitter within the node
                            jitter_x = np.random.uniform(-0.4, 0.4)
                            jitter_y = np.random.uniform(-0.4, 0.4)
                            ax.scatter(j + jitter_x, i + jitter_y, marker=marker, color=color, s=marker_size,
                                       edgecolor='black', linewidths=0.3)

    # Set tick locators for integer labels
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Adjust tick positions and remove tick labels
    ax.set_xticks(np.arange(map_shape[1]), minor=False)
    ax.set_yticks(np.arange(map_shape[0]), minor=False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a colorbar for the neighborhood distances
    plt.colorbar(c, ax=ax, label="Neighborhood Distance")
    
    # Invert y-axis to match matrix layout (top-left corner as (0, 0))
    ax.invert_yaxis()

    plt.show()
    
    # # --- Create a separate figure for the marker legend (sex) ---
    # legend_elements = [
    #     mlines.Line2D([], [], marker=sex_marker_map["Male"], color='black', linestyle='None',
    #                   markersize=10, label='Male'),
    #     mlines.Line2D([], [], marker=sex_marker_map["Female"], color='black', linestyle='None',
    #                   markersize=10, label='Female')
    # ]
    # Create a compact legend plot
    fig_leg, ax_leg = plt.subplots(figsize=(4, 2))
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis('off')

    # ----- Plot the Sex legend in the left column ----- 
    # ax_leg.text(0.1, 0.85, "Sex", fontsize=12, fontweight='bold', ha='left')
    # for idx, (sex, marker) in enumerate(sex_marker_map.items()):
    #     y = 0.75 - idx * 0.15  # Adjust vertical spacing between entries
    #     ax_leg.scatter(0.1, y, marker=marker, color='black', s=100)
    #     ax_leg.text(0.2, y, sex, fontsize=10, va='center')

    # ----- Plot the Site legend in the right column -----
    ax_leg.text(0.55, 0.85, "Site", fontsize=12, fontweight='bold', ha='left')
    for idx, (site, color) in enumerate(site_color_map.items()):
        y = 0.75 - idx * 0.15
        ax_leg.scatter(0.55, y, marker="o", color=color, s=100)
        ax_leg.text(0.65, y, site, fontsize=10, va='center')

    plt.show()





def plot_map_with_selected_inlier_samples_batches(train_neuron_map, test_neuron_map, dataset_id_to_name, key_list, ibis_inlier_site_dict, ibis_inlier_sex_dict, marker_size, colormap_defined):
    """
    Plot the training neuron map with neighborhood distances using quadmesh,
    overlay selected ASD sample markers inside the nodes, where marker shape
    is determined by sex (using ibis_inlier_sex_dict) and marker color by scanner
    (using ibis_inlier_site_dict). Only "Male" and "Female" labels are used.
    
    After the main plot, a separate figure shows the marker legend for sex.
    """
    # Get the neighborhood distances array from the training neurons
    neighborhood_distances = calculate_neighborhood_distances_for_map(train_neuron_map)
    
    # Get the map shape from the training neuron map
    map_shape = train_neuron_map.map_shape()

    # Create a meshgrid for plotting (shifted by 0.5 to center the quads)
    x, y = np.meshgrid(np.arange(map_shape[1] + 1), np.arange(map_shape[0] + 1))

    # Plot the neuron grid using quadmesh (training data neighborhood distances)
    fig, ax = plt.subplots(figsize=(map_shape[1] * 1.5, map_shape[0] * 1.15))
    cmap = colormap_defined  # Using your truncated colormap focusing on lighter tones
    c = ax.pcolormesh(x - 0.5, y - 0.5, neighborhood_distances, cmap=cmap, shading='auto',
                      alpha=0.8, edgecolor='gray', linewidth=0.4)

    # Set the outer linewidth and color for the spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_edgecolor('gray')
    
    # --- Define mapping dictionaries for marker shapes and scanner colors ---
    # Marker shapes based only on existing sex labels ("Male" and "Female")
    # sex_marker_map = {"Male": "*", "Female": "o"}
    # Scanner color mapping as specified:
    site_color_map = {
            "UNC"  :  "blue",
            "STL"  :  "green",
            "SEA"  :  "red",
            "PHI"  :  "purple"
    }

    # Loop through neurons to overlay selected ASD sample markers
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            if (i, j) in train_neuron_map.neurons:
                # Overlay markers only if the test neuron exists in this node
                if (i, j) in test_neuron_map.neurons:
                    test_neuron = test_neuron_map.neurons[(i, j)]
                    # Check that the input_dataset is non-empty
                    if test_neuron.input_dataset is not None and test_neuron.input_dataset.size > 0:
                        # Select indices for ASD samples from this neuron
                        selected_inlier_indices = [
                            idx for idx, identifier in enumerate(test_neuron.input_identifiers)
                            if dataset_id_to_name.get(test_neuron.input_labels[idx]) == "LR-Typical"
                        ]
                        
                        # Loop over selected indices to plot each sample marker
                        for idx in selected_inlier_indices:
                            # Get identifier for the sample
                            identifier = test_neuron.input_identifiers[idx]
                            
                            # Look up the sex using ibis_inlier_sex_dict; default to "Female" if not found
                            # sex_val = ibis_inlier_sex_dict.get(identifier)
                            # marker = sex_marker_map.get(sex_val)
                            # Ignore sex entirely:
                            marker = "*"
                            # Look up scanner label and assign color (default to black if not found)
                            scanner_label = ibis_inlier_site_dict.get(identifier)
                            color = site_color_map.get(scanner_label)
                            
                            # Plot the marker with some random jitter within the node
                            jitter_x = np.random.uniform(-0.4, 0.4)
                            jitter_y = np.random.uniform(-0.4, 0.4)
                            ax.scatter(j + jitter_x, i + jitter_y, marker=marker, color=color, s=marker_size,
                                       edgecolor='black', linewidths=0.3)

    # Set tick locators for integer labels
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Adjust tick positions and remove tick labels
    ax.set_xticks(np.arange(map_shape[1]), minor=False)
    ax.set_yticks(np.arange(map_shape[0]), minor=False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a colorbar for the neighborhood distances
    plt.colorbar(c, ax=ax, label="Neighborhood Distance")
    
    # Invert y-axis to match matrix layout (top-left corner as (0, 0))
    ax.invert_yaxis()

    plt.show()
    
    # # --- Create a separate figure for the marker legend (sex) ---
    # legend_elements = [
    #     mlines.Line2D([], [], marker=sex_marker_map["Male"], color='black', linestyle='None',
    #                   markersize=10, label='Male'),
    #     mlines.Line2D([], [], marker=sex_marker_map["Female"], color='black', linestyle='None',
    #                   markersize=10, label='Female')
    # ]
    # Create a compact legend plot
    fig_leg, ax_leg = plt.subplots(figsize=(4, 2))
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis('off')

    # ----- Plot the Sex legend in the left column ----- 
    # ax_leg.text(0.1, 0.85, "Sex", fontsize=12, fontweight='bold', ha='left')
    # for idx, (sex, marker) in enumerate(sex_marker_map.items()):
    #     y = 0.75 - idx * 0.15  # Adjust vertical spacing between entries
    #     ax_leg.scatter(0.1, y, marker=marker, color='black', s=100)
    #     ax_leg.text(0.2, y, sex, fontsize=10, va='center')

    # ----- Plot the Site legend in the right column -----
    ax_leg.text(0.55, 0.85, "Site", fontsize=12, fontweight='bold', ha='left')
    for idx, (site, color) in enumerate(site_color_map.items()):
        y = 0.75 - idx * 0.15
        ax_leg.scatter(0.55, y, marker="o", color=color, s=100)
        ax_leg.text(0.65, y, site, fontsize=10, va='center')

    plt.show()



def plot_map_with_abcd_asd_samples_batches(
    train_neuron_map,
    test_neuron_map,
    dataset_id_to_name,
    key_list,
    abcd_asd_scanner_dict,
    marker_size=200,
    colormap_defined=plt.cm.Blues_r
):
    """
    Same as before, but:
      - All markers are '*'
      - 'SIEMENS | Prisma' and 'SIEMENS | Prisma_fit' are collapsed to
        'SIEMENS | Prisma and fit'
      - Missing scanner label -> 'Connectome'
    """
    # neighborhood distances & shape
    nd = calculate_neighborhood_distances_for_map(train_neuron_map)
    H, W = train_neuron_map.map_shape()
    x, y = np.meshgrid(np.arange(W+1), np.arange(H+1))

    # combined scanner-color map
    scanner_color_map = {
        'GE MEDICAL SYSTEMS | DISCOVERY MR750':    "#0a53c9",
        'Philips Medical Systems | Achieva dStream': "#cf3838",
        'Philips Medical Systems | Ingenia':       "#f75959",
        'SIEMENS | Prisma and fit':                "#117d02",  # choose a single green
        'Connectome':                              "#000000"
    }

    def clean_label(raw):
        # map both keys into our combined bucket
        if raw in ('SIEMENS | Prisma', 'SIEMENS | Prisma_fit'):
            return 'SIEMENS | Prisma and fit'
        # missing or unknown => Connectome
        return raw if raw in scanner_color_map else 'Connectome'

    # --- 1st plot: scatter raw samples ---
    fig, ax = plt.subplots(figsize=(W*1.5, H*1.15))
    c = ax.pcolormesh(x-0.5, y-0.5, nd, cmap=colormap_defined, shading='auto',
                      alpha=0.8, edgecolor='gray', linewidth=0.4)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4); sp.set_edgecolor('gray')

    for i in range(H):
        for j in range(W):
            if (i,j) in train_neuron_map.neurons and (i,j) in test_neuron_map.neurons:
                tn = test_neuron_map.neurons[(i,j)]
                if tn.input_dataset is None or tn.input_dataset.size==0:
                    continue

                for idx, identifier in enumerate(tn.input_identifiers):
                    if dataset_id_to_name.get(tn.input_labels[idx]) != "ABCD-ASD":
                        continue
                    raw_sc = abcd_asd_scanner_dict.get(identifier)
                    sc = clean_label(raw_sc)
                    color = scanner_color_map[sc]

                    # always star
                    jitter_x, jitter_y = np.random.uniform(-0.4,0.4,2)
                    ax.scatter(j + jitter_x, i + jitter_y,
                               marker="*", color=color,
                               s=marker_size,
                               edgecolor='black', linewidths=0.3)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(c, ax=ax, label="Neighborhood Distance")
    ax.invert_yaxis()
    plt.show()

    # --- 2nd plot: counts per cell, stacked ---
    fig2, ax2 = plt.subplots(figsize=(W*1.5, H*1.15))
    c2 = ax2.pcolormesh(x-0.5, y-0.5, nd, cmap=colormap_defined, shading='auto',
                        alpha=0.3, edgecolor='gray', linewidth=0.4)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.invert_yaxis()

    # ordering for consistent stacks
    # (we only have scanner now; sex is gone)
    scanner_order = {
        'GE MEDICAL SYSTEMS | DISCOVERY MR750': 0,
        'Philips Medical Systems | Achieva dStream': 1,
        'Philips Medical Systems | Ingenia': 2,
        'SIEMENS | Prisma and fit': 3,
        'Connectome': 4
    }

    for i in range(H):
        for j in range(W):
            if (i,j) not in test_neuron_map.neurons:
                continue
            tn = test_neuron_map.neurons[(i,j)]
            if tn.input_dataset is None or tn.input_dataset.size==0:
                continue

            # count per scanner
            cnt = Counter()
            for idx, identifier in enumerate(tn.input_identifiers):
                if dataset_id_to_name.get(tn.input_labels[idx]) == "ABCD-ASD":
                    raw_sc = abcd_asd_scanner_dict.get(identifier)
                    sc = clean_label(raw_sc)
                    cnt[sc] += 1

            # sort & stack
            combos = sorted(cnt.items(), key=lambda kv: scanner_order.get(kv[0], 99))
            start = -(len(combos)-1)/2
            for k, (sc, count) in enumerate(combos):
                y_off = start + k
                ax2.scatter(j, i + y_off*0.2,
                            marker="*", color=scanner_color_map[sc],
                            s=80, edgecolor='black', linewidths=0.3)
                ax2.text(j+0.15, i + y_off*0.2, str(count),
                         fontsize=8, va='center', color='black')

    plt.colorbar(c2, ax=ax2, label="Neighborhood Distance (underlay)")
    plt.show()

    # --- legend (just scanner) ---
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            linestyle='None',
            marker='*',
            markersize=10,
            color=color,           # this sets the star fill color
            label=label
        )
        for label, color in scanner_color_map.items()
    ]

    fig_leg, ax_leg = plt.subplots(figsize=(4, 2))
    ax_leg.legend(
        handles=legend_handles,
        loc='center',
        frameon=False,
        ncol=1,        # one column of entries
        handletextpad=1.0,
        labelspacing=1.0
    )
    ax_leg.axis('off')
    plt.tight_layout()
    plt.show()



def plot_map_with_abcd_inlier_samples_batches(
    train_neuron_map,
    test_neuron_map,
    dataset_id_to_name,
    key_list,
    abcd_inlier_scanner_dict,
    marker_size,
    colormap_defined
):
    """
    Same as before, but:
      - all markers are '*'
      - 'SIEMENS | Prisma' + 'SIEMENS | Prisma_fit' → 'SIEMENS | Prisma and fit'
      - missing/unknown scanner → 'Connectome'
    """
    # 1) prep mesh + distances
    nd = calculate_neighborhood_distances_for_map(train_neuron_map)
    H, W = train_neuron_map.map_shape()
    x, y = np.meshgrid(np.arange(W+1), np.arange(H+1))

    # 2) define your unified scanner→color map
    scanner_color_map = {
        'GE MEDICAL SYSTEMS | DISCOVERY MR750':    "#0a53c9",
        'Philips Medical Systems | Achieva dStream': "#cf3838",
        'Philips Medical Systems | Ingenia':       "#f75959",
        'SIEMENS | Prisma and fit':                "#117d02",
        'Connectome':                              "#000000",
    }

    def clean_scanner(raw):
        # merge the two Siemens labels
        if raw in ('SIEMENS | Prisma', 'SIEMENS | Prisma_fit'):
            return 'SIEMENS | Prisma and fit'
        # if it's already in our map, keep it; else default ⟶ Connectome
        return raw if raw in scanner_color_map else 'Connectome'

    # --- FIRST FIG: scatter the inlier samples ---
    fig, ax = plt.subplots(figsize=(W*1.5, H*1.15))
    c = ax.pcolormesh(x-0.5, y-0.5, nd, cmap=colormap_defined,
                      shading='auto', alpha=0.8,
                      edgecolor='gray', linewidth=0.4)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4); sp.set_edgecolor('gray')

    for i in range(H):
        for j in range(W):
            if (i,j) in train_neuron_map.neurons and (i,j) in test_neuron_map.neurons:
                tn = test_neuron_map.neurons[(i,j)]
                if tn.input_dataset is None or tn.input_dataset.size==0:
                    continue

                # pick only ABCD inliers
                idxs = [
                    idx for idx, uid in enumerate(tn.input_identifiers)
                    if dataset_id_to_name.get(tn.input_labels[idx]) == "ABCD"
                ]
                for idx in idxs:
                    uid = tn.input_identifiers[idx]
                    raw_sc = abcd_inlier_scanner_dict.get(uid)
                    sc = clean_scanner(raw_sc)
                    color = scanner_color_map[sc]

                    # always star
                    dx, dy = np.random.uniform(-0.4,0.4,2)
                    ax.scatter(j+dx, i+dy,
                               marker="*",
                               color=color,
                               s=marker_size,
                               edgecolor='black',
                               linewidths=0.3)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(c, ax=ax, label="Neighborhood Distance")
    ax.invert_yaxis()
    plt.show()

    # --- SECOND FIG: count+stack per scanner only ---
    fig2, ax2 = plt.subplots(figsize=(W*1.5, H*1.15))
    c2 = ax2.pcolormesh(x-0.5, y-0.5, nd, cmap=colormap_defined,
                        shading='auto', alpha=0.3,
                        edgecolor='gray', linewidth=0.4)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.invert_yaxis()

    # ordering so stacks always come out in the same sequence
    scanner_order = {
        'GE MEDICAL SYSTEMS | DISCOVERY MR750': 0,
        'Philips Medical Systems | Achieva dStream': 1,
        'Philips Medical Systems | Ingenia': 2,
        'SIEMENS | Prisma and fit': 3,
        'Connectome': 4
    }

    for i in range(H):
        for j in range(W):
            if (i,j) not in test_neuron_map.neurons:
                continue
            tn = test_neuron_map.neurons[(i,j)]
            if tn.input_dataset is None or tn.input_dataset.size==0:
                continue

            # count by cleaned scanner
            cnt = Counter()
            for idx, uid in enumerate(tn.input_identifiers):
                if dataset_id_to_name.get(tn.input_labels[idx]) == "ABCD":
                    sc = clean_scanner(abcd_inlier_scanner_dict.get(uid))
                    cnt[sc] += 1

            if not cnt:
                continue

            # sort & stack
            items = sorted(cnt.items(), key=lambda kv: scanner_order.get(kv[0], 99))
            start = -(len(items)-1)/2
            for k, (sc, n) in enumerate(items):
                y_off = start + k
                ax2.scatter(j, i + y_off*0.2,
                            marker="*",
                            color=scanner_color_map[sc],
                            s=80,
                            edgecolor='black',
                            linewidths=0.3)
                ax2.text(j+0.15, i+y_off*0.2, str(n),
                         fontsize=8, va='center', color='black')

    plt.colorbar(c2, ax=ax2, label="Neighborhood Distance (underlay)")
    plt.show()

    # --- LEGEND: just scanner categories ---
    handles = [
        mlines.Line2D([], [], linestyle='None',
                      marker="*", markersize=10,
                      color=color, label=label)
        for label, color in scanner_color_map.items()
    ]
    fig_leg, ax_leg = plt.subplots(figsize=(4, 2))
    ax_leg.axis('off')
    ax_leg.legend(handles=handles,
                  loc='center',
                  frameon=False,
                  ncol=1,
                  handletextpad=1.0,
                  labelspacing=1.0)
    plt.tight_layout()
    plt.show()




