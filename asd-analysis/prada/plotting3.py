
from .metrics import calculate_neighborhood_distances_for_map

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as path_effects
from collections import Counter
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap


def plot_map_with_conte_samples(train_neuron_map, test_neuron_map, dataset_id_to_name, colormap_defined):
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
                        asd_indices = [idx for idx, label in enumerate(test_neuron.input_labels) if dataset_id_to_name.get(label) == "CONTE-Outlier"]
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





def plot_map_with_selected_conte_samples(train_neuron_map, test_neuron_map, dataset_id_to_name, key_list, marker_size, colormap_defined):
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
                        selected_asd_indices = [idx for idx, identifier in enumerate(test_neuron.input_identifiers) if identifier in key_list and dataset_id_to_name.get(test_neuron.input_labels[idx]) == "CONTE-Outlier"]
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



def plot_map_with_selected_conte_samples_inlier(train_neuron_map, test_neuron_map, dataset_id_to_name, key_list, marker_size, colormap_defined):
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
    color = "#004bc4" # dark blue
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
                        selected_asd_indices = [idx for idx, identifier in enumerate(test_neuron.input_identifiers) if identifier in key_list and dataset_id_to_name.get(test_neuron.input_labels[idx]) == "CONTE-Inlier"]
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





def plot_map_with_conte_outlier_samples_batches(
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
        'P': "#cf3838",
        'T': "#0a53c9",
        'T3': "#117d02",  # choose a single green
    }

    def clean_label(raw):
        return raw # do not change the label, just use it as is

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
                    if dataset_id_to_name.get(tn.input_labels[idx]) != "CONTE-Outlier":
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
        'P': 0,
        'T': 1,
        'T3': 2,
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
                if dataset_id_to_name.get(tn.input_labels[idx]) == "CONTE-Outlier":
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



def plot_map_with_conte_inlier_samples_batches(
    train_neuron_map,
    test_neuron_map,
    dataset_id_to_name,
    key_list,
    abcd_inlier_scanner_dict,
    marker_size,
    colormap_defined
):
    """
    """
    # 1) prep mesh + distances
    nd = calculate_neighborhood_distances_for_map(train_neuron_map)
    H, W = train_neuron_map.map_shape()
    x, y = np.meshgrid(np.arange(W+1), np.arange(H+1))

    # 2) define your unified scanner→color map
    scanner_color_map = {
        'P': "#cf3838",
        'T': "#0a53c9",
        'T3': "#117d02",  # choose a single green
    }

    def clean_scanner(raw):
        return raw # do not change the scanner, just use it as is

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
                    if dataset_id_to_name.get(tn.input_labels[idx]) == "CONTE-Inlier"
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
        'P': 0,
        'T': 1,
        'T3': 2,
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
                if dataset_id_to_name.get(tn.input_labels[idx]) == "CONTE-Inlier":
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




