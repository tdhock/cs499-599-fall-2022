import pandas as pd
import numpy as np
zip_df = pd.read_csv(
    "~/teaching/cs499-599-fall-2022/data/zip.test.gz",
    header=None,
    sep=" ")
out_col = 0
is_output = zip_df.columns == out_col
zip_features = zip_df.iloc[:,is_output==False].to_numpy()
zip_labels = zip_df.iloc[:,is_output]
label_counts = zip_labels.value_counts()
nrow, ncol = zip_features.shape
n_classes = len(label_counts.index)
weight_mat = np.random.randn(ncol, n_classes-1)
weight_mat.shape
pred_z_mat = np.matmul(zip_features, weight_mat)
pred_z_mat.shape
outside_log = np.column_stack([np.repeat(1, nrow), pred_z_mat])
log_term = np.log(1+np.exp(pred_z_mat).sum(axis=1))
log_prob_mat = outside_log - log_term.reshape(nrow, 1)
pred_prob_mat = np.exp(log_prob_mat)
(pred_prob_mat < 0).sum()
(pred_prob_mat > 1).sum()
pred_prob_sum_vec = softmax_mat.sum(axis=1)
pred_prob_sum_vec.min()
pred_prob_sum_vec.max()
zip_label_vec = zip_labels.to_numpy().reshape(nrow)
pred_prob_mat[np.arange(nrow), zip_label_vec]
