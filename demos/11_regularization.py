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
    set_features[set_name] = torch.from_numpy(
        scaled_features[is_set,:]).float()
    spam_labels_set = spam_labels[is_set]
    set_labels[set_name] = torch.from_numpy(
        spam_labels_set).float().reshape(len(spam_labels_set),1)
{set_name:array.shape for set_name, array in set_features.items()}

import torch
class NumpyData(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __getitem__(self, item):
        return self.features[item,:], self.labels[item]
    def __len__(self):
        return len(self.labels)
ds = NumpyData(set_features["subtrain"], set_labels["subtrain"])
ds[0]
dl = torch.utils.data.DataLoader(
    ds, batch_size=200, shuffle=True)
for batch_features, batch_labels in dl:
    pass
batch_features
batch_features.shape
batch_labels

class Net(torch.nn.Module):
    def __init__(self, n_hidden_layers, units_in_each_hidden_layer=100):
        super(Net, self).__init__()
        units_per_layer = [ncol]
        for layer_i in range(n_hidden_layers):
            units_per_layer.append(units_in_each_hidden_layer)
        units_per_layer.append(1)
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
pred_tensor = model(batch_features)
loss_fun = torch.nn.BCEWithLogitsLoss()
loss_tensor = loss_fun(pred_tensor, batch_labels)

max_hidden_layers = 5
loss_df_list = []
for n_hidden_layers in range(max_hidden_layers):
    model = Net(n_hidden_layers)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    # gradient descent 
    MAX_EPOCHS = 100
    for epoch in range(MAX_EPOCHS):
        for batch_features, batch_labels in dl:
            optimizer.zero_grad()
            pred_tensor = model(batch_features)
            loss_tensor = loss_fun(pred_tensor, batch_labels)
            loss_tensor.backward()
            optimizer.step()
        for set_name in set_features:
            set_X = set_features[set_name]
            set_y = set_labels[set_name]
            set_pred = model(set_X)
            set_loss = loss_fun(set_pred, set_y)
            epoch_loss_df = pd.DataFrame({
                "n_hidden_layers":n_hidden_layers,
                "set_name":set_name,
                "loss":float(set_loss),
                "epoch":epoch
            }, index=[0])
            print(epoch_loss_df)
            loss_df_list.append(epoch_loss_df)
loss_df =pd.concat(loss_df_list)

import plotnine as p9
loss_df.index = range(len(loss_df))
def get_min(series_obj):
    return series_obj.index[series_obj.argmin()]
rows_to_plot = loss_df.groupby(
    ["n_hidden_layers", "set_name"]
)["loss"].apply(get_min)
layers_plot_data = loss_df.iloc[rows_to_plot, :]

set_colors = {"subtrain":"red","validation":"blue"}
gg = p9.ggplot()+\
    p9.scale_color_manual(values=set_colors)+\
    p9.geom_line(
        p9.aes(
            x="n_hidden_layers",
            y="loss",
            color="set_name"
        ),
        data=layers_plot_data)
gg.save("11_regularization_layers.png", width=10, height=5)

validation_df = loss_df.query("set_name=='validation'")
min_i = validation_df.loss.argmin()
min_row = pd.DataFrame(dict(validation_df.iloc[min_i,:]), index=[0])
gg = p9.ggplot()+\
    p9.facet_grid(". ~ n_hidden_layers")+\
    p9.scale_color_manual(values=set_colors)+\
    p9.scale_fill_manual(values=set_colors)+\
    p9.geom_line(
        p9.aes(
            x="epoch",
            y="loss",
            color="set_name"
        ),
        data=loss_df)+\
    p9.geom_point(
        p9.aes(
            x="epoch",
            y="loss",
            fill="set_name"
        ),
        color="black",
        data=min_row)
gg.save("11_regularization.png", width=10, height=5)
