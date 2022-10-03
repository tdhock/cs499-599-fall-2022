import pandas as pd
import numpy as np

spam_df = pd.read_csv(
    "~/teaching/cs570-spring-2022/data/spam.data",
    header=None,
    sep=" ")

spam_features = spam_df.iloc[:,:-1].to_numpy()
spam_labels = spam_df.iloc[:,-1].to_numpy()
# 1. feature scaling.
spam_mean = spam_features.mean(axis=0)
spam_sd = np.sqrt(spam_features.var(axis=0))
scaled_features = (spam_features-spam_mean)/spam_sd
scaled_features.mean(axis=0)
scaled_features.var(axis=0)

# 2. subtrain/validation split.
np.random.seed(1)
n_folds = 5
fold_vec = np.random.randint(low=0, high=n_folds, size=spam_labels.size)
validation_fold = 0
is_set_dict = {
    "validation":fold_vec == validation_fold,
    "subtrain":fold_vec != validation_fold,
}

set_features = {}
set_labels = {}
for set_name, is_set in is_set_dict.items():
    set_features[set_name] = scaled_features[is_set,:]
    set_labels[set_name] = spam_labels[is_set]
{set_name:array.shape for set_name, array in set_features.items()}

nrow, ncol = set_features["subtrain"].shape
import torch
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.weight_vec = torch.nn.Linear(ncol, 1)
    def forward(self, feature_mat):
        return self.weight_vec(feature_mat)

model = LinearModel()
loss_fun = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

optimizer.zero_grad()
feature_tensor = torch.from_numpy(set_features["subtrain"]).float()
label_01 = set_labels["subtrain"]
label_tensor = torch.from_numpy(label_01).float()
pred_tensor = model(feature_tensor).reshape(nrow)
loss_tensor = loss_fun(pred_tensor, label_tensor)
loss_tensor.backward()
optimizer.step()

# loss_tensor is mean logistic loss.
loss_tensor
label_neg_pos = np.where(label_01 == 1, 1, -1)
np.mean(np.log(1+np.exp(-label_neg_pos * pred_vec.detach().numpy())))

