import torchvision
import torch
import pandas as pd
import numpy as np
zip_df = pd.read_csv(
    "~/teaching/cs499-599-fall-2022/data/zip.test.gz",
    header=None,
    sep=" ")
out_col = 0
is_output = zip_df.columns == out_col
zip_features = zip_df.iloc[:,is_output==False].to_numpy()
nrow, ncol = zip_features.shape
zip_labels = zip_df.iloc[:,is_output]
zip_label_vec = zip_labels.to_numpy().reshape(nrow)
label_counts = zip_labels.value_counts()
n_classes = len(label_counts.index)

class TorchModel(torch.nn.Module):
    def __init__(self, *units_per_layer):
        super(TorchModel, self).__init__()
        seq_args = []
        for layer_i in range(len(units_per_layer)-1):
            units_in = units_per_layer[layer_i]
            units_out = units_per_layer[layer_i+1]
            seq_args.append(
                torch.nn.Linear(units_in, units_out))
            if layer_i != len(units_per_layer)-2:
                seq_args.append(torch.nn.ReLU())
        self.stack = torch.nn.Sequential(*seq_args)
    def forward(self, feature_mat):
        return self.stack(feature_mat)

zip_feature_tensor = torch.from_numpy(zip_features).float()
zip_label_tensor = torch.from_numpy(zip_label_vec)
net = Net(ncol, 1000, 100, n_classes)
pred_y_hat_mat = net(zip_feature_tensor)
pred_y_hat_mat.shape
pred_y_hat_mat[0,]
loss_fun = torch.nn.CrossEntropyLoss()
loss_value = loss_fun(pred_y_hat_mat, zip_label_tensor)
loss_value.backward()

## TODO weds.
np.random.seed(1)
weight_sd = 100
weight_mat = np.random.randn(ncol, n_classes-1)*weight_sd
weight_mat.shape

batch_size = 3
batch_features = zip_features[:batch_size, :]
batch_features.shape
batch_labels = zip_label_vec[:batch_size]

pred_y_mat = np.matmul(batch_features, weight_mat)
pred_y_mat.shape
pred_z_mat = np.column_stack([np.repeat(0, batch_size), pred_y_mat])
pred_z_mat.shape
pred_z_mat

## naive method.
exp_mat = np.exp(pred_z_mat)
denominator_vec = exp_mat.sum(axis=1).reshape(batch_size, 1)
naive_prob_mat = exp_mat/denominator_vec
inside_log = naive_prob_mat[np.arange(batch_size), batch_labels]
naive_loss_vec = -np.log(inside_log).reshape(batch_size, 1)
zero_one_mat = np.zeros((batch_size, n_classes))
zero_one_mat[np.arange(batch_size), batch_labels] = 1
naive_grad_mat = naive_prob_mat-zero_one_mat

def analyze_prob_mat(pmat):
    sum_vec = pmat.sum(axis=1)
    sum_not_nan = sum_vec[~np.isnan(sum_vec)]
    print("%d values <0:%d, >1:%d, nan:%d, row sums min=%f, max=%f\n"%(
        pmat.size,
        (pmat < 0).sum(),
        (pmat > 1).sum(),
        np.isnan(pmat).sum(),
        sum_not_nan.min(),
        sum_not_nan.max()))

analyze_prob_mat(naive_prob_mat)

## numerically stable method.
M_vec = pred_z_mat.max(axis=1).reshape(batch_size, 1)
inside_exp_mat = pred_z_mat-M_vec
sum_exp_vec = np.exp(inside_exp_mat).sum(axis=1).reshape(batch_size, 1)
L_vec = M_vec + np.log(sum_exp_vec)
stable_loss_vec = L_vec - pred_z_mat[
    np.arange(batch_size), batch_labels
].reshape(batch_size, 1)
stable_loss_vec - naive_loss_vec

stable_prob_mat = np.exp(pred_z_mat-L_vec)
stable_prob_mat - naive_prob_mat

analyze_prob_mat(stable_prob_mat)

zero_one_mat = np.zeros((batch_size, n_classes))
zero_one_mat[np.arange(batch_size), batch_labels] = 1
stable_grad_mat = stable_prob_mat - zero_one_mat
stable_grad_mat - naive_grad_mat

ds = torchvision.datasets.MNIST(
    root="~/teaching/cs499-599-fall-2022/data", 
    download=True,
    transform=torchvision.transforms.ToTensor(),
    train=False)
dl = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
for mnist_features, mnist_labels in dl:
    pass
mnist_features.flatten(start_dim=1).numpy()
mnist_labels.numpy()
