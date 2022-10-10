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

import torch
class CSV(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __getitem__(self, item):
        return self.features[item,:], self.labels[item]
    def __len__(self):
        return len(self.labels)
ds = CSV(set_features["subtrain"], set_labels["subtrain"])
ds[0]
dl = torch.utils.data.DataLoader(
    ds, batch_size=2, shuffle=True)
for batch_features, batch_labels in dl:
    pass
batch_features
batch_features.shape
batch_labels

class LinearModel(torch.nn.Module):
    def __init__(self, num_inputs):
        super(LinearModel, self).__init__()
        self.weight_vec = torch.nn.Linear(num_inputs, 1)
    def getitem(self, item):
        weights, intercept = [p for p in self.weight_vec.parameters()]
        return weights.data[0][item]
    def forward(self, feature_mat):
        return self.weight_vec(feature_mat)
LM=LinearModel(10)
LM.getitem(5)


class Net(torch.nn.Module):
    def __init__(self, *units_per_layer):
        super(Net, self).__init__()
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

nrow, ncol = set_features["subtrain"].shape
net = Net(ncol, 100, 10, 1)
net(batch_features.float())

sizes_dict = {
    "linear":((ncol, 1), 0.1),
    "nnet":((ncol, 100, 10, 1), 0.01),
    }
model_dict = {}
for model_name, (units_per_layer, step_size) in sizes_dict.items():
    model_dict[model_name] = Net(*units_per_layer)

net = Net(ncol, 200, 100, 50, 10, 1)
feature_tensor = torch.from_numpy(set_features["subtrain"]).float()
net(feature_tensor).shape
model = LinearModel(ncol)
[p for p in model.weight_vec.parameters()]
[p for p in model.parameters()]

loss_fun = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

optimizer.zero_grad()
[p.grad for p in model.weight_vec.parameters()]
label_01 = set_labels["subtrain"]
label_tensor = torch.from_numpy(label_01).float()
pred_tensor = model(feature_tensor).reshape(nrow)
loss_tensor = loss_fun(pred_tensor, label_tensor)
[p.grad for p in model.weight_vec.parameters()]

loss_tensor.backward()
[p.grad for p in model.weight_vec.parameters()]

[p for p in model.weight_vec.parameters()]
optimizer.step()
[p for p in model.weight_vec.parameters()]

# loss_tensor is mean logistic loss.
loss_tensor
label_neg_pos = np.where(label_01 == 1, 1, -1)
np.mean(np.log(1+np.exp(-label_neg_pos * pred_vec.detach().numpy())))


model = Net(ncol, 100, 10, 1)
loss_fun = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_df_list = []
# gradient descent 
MAX_EPOCHS = 100
for epoch in range(MAX_EPOCHS):
    for batch_features, batch_labels in dl:
        optimizer.zero_grad()
        pred_tensor = model(batch_features.float()).reshape(len(batch_labels))
        loss_tensor = loss_fun(pred_tensor, batch_labels.float())
        loss_tensor.backward()
        optimizer.step()
    for set_name in set_features:
        set_X = set_features[set_name]
        set_y = set_labels[set_name]
        set_X_tensor = torch.from_numpy(set_X).float()
        set_y_tensor = torch.from_numpy(set_y)
        set_pred = model(set_X_tensor).reshape(len(set_y))
        set_loss = loss_fun(set_pred, set_y_tensor.float())
        loss_df_list.append(pd.DataFrame({
            "set_name":set_name,
            "loss":float(set_loss),
            "epoch":epoch
        }, index=[0]))
loss_df =pd.concat(loss_df_list)

import plotnine as p9
gg = p9.ggplot()+\
    p9.geom_line(
        p9.aes(
            x="epoch",
            y="loss",
            color="set_name"
        ),
        data=loss_df)
gg.save("06_torch_linear_model.png")
