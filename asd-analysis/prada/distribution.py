import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .metrics import calculate_leaf_neuron_metrics, count_total_neurons, get_data_mapped_counts_list

def calculate_entropy(data_mapped_counts):
    """
    Calculate the Shannon entropy of the data_mapped_counts.
    
    :param data_mapped_counts: List of data mapped counts to leaf neurons
    :return: Entropy value
    """
    total_data = sum(data_mapped_counts)
    if total_data == 0:
        return 0.0

    # Calculate probability distribution
    probabilities = np.array(data_mapped_counts) / total_data
    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_and_plot_entropy_and_neurons(mapped_units, log_scale_t1, log_scale_t2, floats=3):
    """
    Calculate the entropy, total neurons, and total leaf neurons for each t1-t2 combination and plot it as separate colormaps.
    
    :param mapped_units: Dictionary containing the mapped units (train and test) for each (t1_r, t2_r) pair
    :param log_scale_t1: List of t1 values used for the y-axis
    :param log_scale_t2: List of t2 values used for the x-axis
    """
    entropy_values = np.zeros((len(log_scale_t1), len(log_scale_t2)))  # Store entropy values
    neuron_counts = np.zeros((len(log_scale_t1), len(log_scale_t2)))   # Store neuron counts
    annotations = np.empty((len(log_scale_t1), len(log_scale_t2)), dtype=object)  # Store annotations for both plots
    
    # Iterate over t1 and t2 values
    for i, t1 in enumerate(log_scale_t1):
        for j, t2 in enumerate(log_scale_t2):
            t1_r = round(t1, floats)
            t2_r = round(t2, floats)

            # Access the train_mapped_zero_unit for the current (t1_r, t2_r) combination
            train_mapped_zero_unit = mapped_units.get((t1_r, t2_r), {}).get("train_mapped_zero_unit")

            if train_mapped_zero_unit:
                # Calculate metrics and data_mapped_counts for the train_mapped_zero_unit
                metrics_for_zero_unit = calculate_leaf_neuron_metrics(train_mapped_zero_unit)
                data_mapped_counts = get_data_mapped_counts_list(metrics_for_zero_unit)

                # Calculate entropy from data_mapped_counts
                entropy = calculate_entropy(data_mapped_counts)
                entropy_values[i, j] = entropy  # Store entropy for this t1-t2 pair

                # Get total neurons and leaf neurons
                total_neurons = count_total_neurons(train_mapped_zero_unit)
                total_leaf_neurons = len(data_mapped_counts)
                neuron_counts[i, j] = total_neurons  # Store the total neuron count for the colormap

                # Create annotation string (same for both plots)
                annotations[i, j] = f"ent: {entropy:.2f}\n\nneurons: {total_neurons}\nleafs: {total_leaf_neurons}"
            else:
                entropy_values[i, j] = np.nan  # Set NaN if no data for this (t1_r, t2_r)
                neuron_counts[i, j] = np.nan
                annotations[i, j] = "N/A"
     
    # Plot the entropy and neuron colormaps with the same annotations
    plot_entropy_colormap(entropy_values, log_scale_t1, log_scale_t2, annotations, floats=floats)
    plot_neuron_colormap(neuron_counts, log_scale_t1, log_scale_t2, annotations, floats=floats)

    # Calculate and print the min and max entropy values (ignoring NaN values)
    min_entropy = np.nanmin(entropy_values)
    max_entropy = np.nanmax(entropy_values)
    print(f"Minimum entropy: {min_entropy:.2f}")
    print(f"Maximum entropy: {max_entropy:.2f}")

def plot_entropy_colormap(entropy_values, log_scale_t1, log_scale_t2, annotations, floats=3):
    """
    Plot entropy values as a colormap on t1-t2 axes, with custom annotations showing entropy, total neurons, and leaf neurons.
    
    :param entropy_values: 2D array of entropy values
    :param log_scale_t1: List of t1 values used for the y-axis
    :param log_scale_t2: List of t2 values used for the x-axis
    :param annotations: 2D array of strings to display in each cell (entropy, total neurons, leaf neurons)
    """
    # Round log_scale values to 3 decimal places for display on the axes
    rounded_log_scale_t1 = [round(val, floats) for val in log_scale_t1]
    rounded_log_scale_t2 = [round(val, floats) for val in log_scale_t2]
    
    plt.figure(figsize=(10, 8))
    # Create heatmap with custom annotations and disable automatic colorbar
    heatmap = sns.heatmap(entropy_values, xticklabels=rounded_log_scale_t2, yticklabels=rounded_log_scale_t1, cmap="viridis",
                annot=annotations, fmt="", cbar=False, linewidths=0.5)
    
    plt.title("Entropy Colormap with Neuron Counts (t1-t2 Axes)")
    plt.xlabel("t2 values")
    plt.ylabel("t1 values")
    plt.colorbar(heatmap.collections[0], label="Entropy")  # Attach a single colorbar
    plt.show()

def plot_neuron_colormap(neuron_counts, log_scale_t1, log_scale_t2, annotations, floats=3):
    """
    Plot neuron count values as a colormap on t1-t2 axes, with custom annotations showing entropy, total neurons, and leaf neurons.
    
    :param neuron_counts: 2D array of neuron count values
    :param log_scale_t1: List of t1 values used for the y-axis
    :param log_scale_t2: List of t2 values used for the x-axis
    :param annotations: 2D array of strings to display in each cell (entropy, total neurons, leaf neurons)
    """
    # Round log_scale values to 3 decimal places for display on the axes
    rounded_log_scale_t1 = [round(val, floats) for val in log_scale_t1]
    rounded_log_scale_t2 = [round(val, floats) for val in log_scale_t2]
    
    plt.figure(figsize=(10, 8))
    # Create heatmap with custom annotations and disable automatic colorbar
    heatmap = sns.heatmap(neuron_counts, xticklabels=rounded_log_scale_t2, yticklabels=rounded_log_scale_t1, cmap="coolwarm",
                annot=annotations, fmt="", cbar=False, linewidths=0.5)
    
    plt.title("Neuron Count Colormap with Entropy Values (t1-t2 Axes)")
    plt.xlabel("t2 values")
    plt.ylabel("t1 values")
    plt.colorbar(heatmap.collections[0], label="Neuron Count")  # Attach a single colorbar
    plt.show()



def analyze_grouped_data_distribution_with_percentage(mapped_units, t1_r, t2_r):
    """
    Analyze and display the distribution of data points mapped to leaf nodes, grouped by the number of samples
    each neuron has. Includes both the percentage of data mapped to each neuron and the total percentage for each group.
    
    :param mapped_units: Dictionary containing the mapped units (train and test) for each (t1_r, t2_r) pair
    :param t1_r: The rounded t1 value for analysis
    :param t2_r: The rounded t2 value for analysis
    :return: DataFrame summarizing the grouped data point distribution with percentages
    """
    # Access the train_mapped_zero_unit for the given t1_r and t2_r
    train_mapped_zero_unit = mapped_units.get((t1_r, t2_r), {}).get("train_mapped_zero_unit")

    if not train_mapped_zero_unit:
        print(f"No data available for t1: {t1_r}, t2: {t2_r}")
        return None

    # Calculate data_mapped_counts for the leaf nodes
    metrics_for_zero_unit = calculate_leaf_neuron_metrics(train_mapped_zero_unit)
    data_mapped_counts = get_data_mapped_counts_list(metrics_for_zero_unit)
    total_data_points = sum(data_mapped_counts)
    total_neurons = len(data_mapped_counts)

    # Create a DataFrame to store the data points mapped to each neuron
    data_distribution = pd.DataFrame({'Data Points Mapped': data_mapped_counts})

    # Count the number of neurons with the same number of data points mapped
    grouped_distribution = data_distribution.groupby('Data Points Mapped').size().reset_index(name='Number of Neurons')

    # Calculate total data mapped for each group and percentages
    grouped_distribution['Total Data Mapped'] = grouped_distribution['Data Points Mapped'] * grouped_distribution['Number of Neurons']
    grouped_distribution['Percentage of Data Mapped'] = (grouped_distribution['Data Points Mapped'] / total_data_points) * 100
    grouped_distribution['Total Percentage of Data Mapped'] = (grouped_distribution['Total Data Mapped'] / total_data_points) * 100

    # Include neurons with 0 data points mapped
    if 0 not in grouped_distribution['Data Points Mapped'].values:
        empty_neurons_count = total_neurons - len(data_mapped_counts)
        if empty_neurons_count > 0:
            grouped_distribution = pd.concat([
                pd.DataFrame({
                    'Data Points Mapped': [0],
                    'Number of Neurons': [empty_neurons_count],
                    'Total Data Mapped': [0],
                    'Percentage of Data Mapped': [0],
                    'Total Percentage of Data Mapped': [0]
                }),
                grouped_distribution
            ], ignore_index=True)

    # Display the grouped distribution
    print(f"Grouped data distribution for t1: {t1_r}, t2: {t2_r}")
    return grouped_distribution


def summarize_neuron_data_distribution(mapped_units, t1_r, t2_r, sample_threshold=4):
    """
    Summarize the data distribution for a specific t1 and t2 value, displaying:
    - Number of neurons with more than the specified sample threshold
    - Total number of neurons
    - Total number of samples
    
    :param mapped_units: Dictionary containing the mapped units (train and test) for each (t1_r, t2_r) pair
    :param t1_r: The rounded t1 value for analysis
    :param t2_r: The rounded t2 value for analysis
    :param sample_threshold: The sample count threshold for filtering neurons (default is 4)
    """
    # Access the train_mapped_zero_unit for the given t1_r and t2_r
    train_mapped_zero_unit = mapped_units.get((t1_r, t2_r), {}).get("train_mapped_zero_unit")

    if not train_mapped_zero_unit:
        print(f"No data available for t1: {t1_r}, t2: {t2_r}")
        return

    # Calculate data_mapped_counts for the leaf nodes
    metrics_for_zero_unit = calculate_leaf_neuron_metrics(train_mapped_zero_unit)
    data_mapped_counts = get_data_mapped_counts_list(metrics_for_zero_unit)
    total_data_points = sum(data_mapped_counts)
    total_neurons = len(data_mapped_counts)

    # Count the number of neurons with more than the specified sample threshold
    neurons_above_threshold = sum(1 for count in data_mapped_counts if count >= sample_threshold)
    neurons_below_threshold = sum(1 for count in data_mapped_counts if count < sample_threshold)

    # Print the summary statistics
    print(f"Summary for t1: {t1_r}, t2: {t2_r}")
    print(f"Number of neurons with and more than {sample_threshold} samples: {neurons_above_threshold}")
    print(f"Number of neurons with and less than {sample_threshold} samples: {neurons_below_threshold}")
    print(f"Total number of leaf neurons: {total_neurons}")
    print(f"Total number of samples: {total_data_points}")