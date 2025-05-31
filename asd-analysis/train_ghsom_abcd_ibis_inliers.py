import numpy as np
import pickle
from collections import OrderedDict
from prada.ghsom_model.GHSOM import GHSOM
from prada.ghsom_model.GSOM import GSOM
from sklearn.decomposition import PCA

scale = lambda x,a,b: (b-a)*(x-min(x))/(max(x)-min(x)) + a

with open('data_scorenorm/combined_data_score_norms_v2.pkl', 'rb') as file:
    combined_data = pickle.load(file)

score_norms_abcd_val = combined_data['score_norms']['abcd-val']
score_norms_abcd_test = combined_data['score_norms']['abcd-test']
score_norms_ibis_typical = combined_data['score_norms']['ibis-inlier']
score_norms_ibis_hr_typical = combined_data['score_norms']['ibis-hr-inliers']
score_norms_ibis_atypical = combined_data['score_norms']['ibis-atypical']
score_norms_ibis_asd = combined_data['score_norms']['ibis-asd']
score_norms_ibis_ds = combined_data['score_norms']['ibis-ds-sa']

score_norms_abcd = np.concatenate([score_norms_abcd_val, score_norms_abcd_test], axis=0)

identifiers_abcd_val = combined_data['identifier']['abcd-val']
identifiers_abcd_test = combined_data['identifier']['abcd-test']
identifiers_ibis_typical = combined_data['identifier']['ibis-inlier']
identifiers_ibis_hr_typical = combined_data['identifier']['ibis-hr-inliers']
identifiers_ibis_atypical = combined_data['identifier']['ibis-atypical']
identifiers_ibis_asd = combined_data['identifier']['ibis-asd']
identifiers_ibis_ds = combined_data['identifier']['ibis-ds-sa']

identifiers_abcd = np.concatenate([identifiers_abcd_val, identifiers_abcd_test], axis=0)

# Squeeze the score_norms to have the same shape as abcd_train
score_norms_abcd_val = np.squeeze(score_norms_abcd_val)
score_norms_abcd_test = np.squeeze(score_norms_abcd_test)
score_norms_abcd = np.squeeze(score_norms_abcd)
score_norms_ibis_typical = np.squeeze(score_norms_ibis_typical)
score_norms_ibis_hr_typical = np.squeeze(score_norms_ibis_hr_typical)
score_norms_ibis_atypical = np.squeeze(score_norms_ibis_atypical)
score_norms_ibis_asd = np.squeeze(score_norms_ibis_asd)
score_norms_ibis_ds = np.squeeze(score_norms_ibis_ds)


dataset_id_to_name = {-1: "ABCD", 0:"LR-Typical", 1:"HR-Typical", 2:"Atypical", 3:"Down Syndrome", 4:"ASD"}
dataset_name_to_id = {v:k for k,v in dataset_id_to_name.items()}
#dataset_list = [abcd_data, ibis_typical, ibis_hr_typical, ibis_atypical, ibis_ds, ibis_asd]
dataset_list = [score_norms_abcd, score_norms_ibis_typical, score_norms_ibis_hr_typical, score_norms_ibis_atypical, score_norms_ibis_ds, score_norms_ibis_asd]

X_data = np.concatenate(dataset_list, axis=0)
y_labels = np.concatenate([[i-1]*len(d) for i,d in enumerate(dataset_list)])

assert len(y_labels)==len(X_data)
inlier_data = X_data[y_labels <= 0]
test_data = X_data[y_labels > 0]
test_labels = y_labels[y_labels > 0]
train_labels = y_labels[y_labels <= 0]


mu, std = np.mean(inlier_data, axis=0), np.std(inlier_data, axis=0)
inlier_data = (inlier_data - mu) / std
test_data = (test_data  - mu) / std

dssamples = test_data[test_labels == dataset_name_to_id["ASD"]]
X_data.shape, inlier_data.shape, len(dssamples), test_data.shape
# This it for stacking identifiers. When you have all the identifiers, you can combine this with the previous code.
identifiers_list = [identifiers_abcd, identifiers_ibis_typical, identifiers_ibis_hr_typical, identifiers_ibis_atypical, identifiers_ibis_ds, identifiers_ibis_asd]
indentifiers_data = np.concatenate(identifiers_list, axis=0)
inlier_identifiers = indentifiers_data[y_labels <= 0] 
test_identifiers = indentifiers_data[y_labels > 0]
# Some people suggest to keep the ratio of the height and width
# to be the ratio of the first two principal components

pca = PCA(n_components=5).fit(inlier_data)
print(pca.explained_variance_ratio_, pca.explained_variance_ratio_[0]/pca.explained_variance_ratio_[1])
num_neurons = 5*np.sqrt(test_data.shape[0])
grid_size = int(np.ceil(np.sqrt(num_neurons)))

ratio = pca.explained_variance_ratio_[0]/pca.explained_variance_ratio_[1]
h = np.sqrt(num_neurons / ratio)
w = h*ratio
ratio, h,w, num_neurons

n_samples, n_features = inlier_data.shape

print("dataset length: {}".format(n_samples))
print("features per example: {}".format(n_features))
# dummy labels for interactive_plot_with_labels()
labels = np.zeros(n_samples)
def formatted_time(time):
    # Calculate hours, minutes, seconds, and subseconds
    hours = int(time // 3600)
    minutes = int((time % 3600) // 60)
    seconds = int(time % 60)
    subseconds = int((time % 1) * 100)  # Subseconds as centiseconds
    return f"hr:min:sec:msc: {hours:02} : {minutes:02} : {seconds:02} : {subseconds:03}"
import itertools
import time
import pickle
# Function that encapsulates the training process for a single pair of t1 and t2
def train_ghsom(t1, t2, inlier_data, counter, counters):
    # Check if t1 and t2 are arrays or scalars
    if isinstance(t1, np.ndarray):
        t1_r = np.round(t1, 3)
    else:
        t1_r = round(t1, 3)

    if isinstance(t2, np.ndarray):
        t2_r = np.round(t2, 3)
    else:
        t2_r = round(t2, 3)

    ghsom = GHSOM(input_dataset=inlier_data, t1=t1, t2=t2, learning_rate=0.15, decay=0.95, gaussian_sigma=1.5)

    start_time = time.time()
    print(f"{counter}/{counters}.............................")
    print(f"Training... \t t1={t1_r} \t t2={t2_r},")

    zero_unit = ghsom.train(epochs_number=1000, dataset_percentage=1.0, min_dataset_size=30, seed=0, grow_maxiter=10)

    end_time = time.time()
    print(f"Execution time: {(end_time - start_time)}")
    print(f"Execution time: {formatted_time(end_time - start_time)}")

    # Save the trained model
    zero_unit_path = f'ghsom_outputs/trained_maps/trained_z_unit_grid_v08/{t1_r}_{t2_r}_ep-2000_mx-it-10_v2_zu.pkl'
    with open(zero_unit_path, 'wb') as f:
        pickle.dump(zero_unit, f)
    print("zero_unit object saved!")
# Create a logarithmic scale between 0.01 and 1
log_scale_min_t1, log_scale_max_t1, number_of_points_t1 = 0.025, 0.045, 5
log_scale_min_t2, log_scale_max_t2, number_of_points_t2 = 0.008, 0.01, 5

log_scale_t1 = np.logspace(np.log10(log_scale_min_t1), np.log10(log_scale_max_t1), num=number_of_points_t1)  # num: defines the number of points
log_scale_t2 = np.logspace(np.log10(log_scale_min_t2), np.log10(log_scale_max_t2), num=number_of_points_t2)  # num: defines the number of points
print("log_scale_t1:",log_scale_t2)
print("log_scale_t2:",log_scale_t2)


from joblib import Parallel, delayed
# Generate all combinations of t1 and t2 using itertools.product
t1_t2_combinations = list(itertools.product(log_scale_t1, log_scale_t2))

# Total number of combinations
counters = len(t1_t2_combinations)

start_t = time.time()

# Parallelize the training process using joblib
Parallel(n_jobs=-1)(delayed(train_ghsom)(t1, t2, inlier_data, counter + 1, counters)
                    for counter, (t1, t2) in enumerate(t1_t2_combinations))

end_t = time.time()
print(f"Total Execution time: {formatted_time(end_t - start_t)}")


