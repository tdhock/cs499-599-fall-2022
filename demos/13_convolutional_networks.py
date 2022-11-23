import pandas as pd
import numpy as np
import os
# work-around for rendering plots under windows, which hangs within
# emacs python shell: instead write a PNG file and view in browser.
import webbrowser
on_windows = os.name == "nt"
in_render = r.in_render if 'r' in dir() else False
using_agg = on_windows and not in_render
if using_agg:
    import matplotlib
    matplotlib.use("agg")
def show(g):
    if not using_agg:
        return g
    g.save("tmp.png")
    webbrowser.open('tmp.png')


zip_df = pd.read_csv(
    "~/teaching/cs570-spring-2022/data/zip.train.gz",
    header=None,
    sep=" ")
zip_71 = zip_df[ zip_df.iloc[:,0].isin([7,1]) ]
zip_features = zip_71.iloc[:,1:257].to_numpy()
zip_labels_71 = zip_71.iloc[:,0].to_numpy()
zip_labels = np.where(zip_labels_71==1, 1, 0)
np.random.seed(1)
n_folds = 5
fold_vec = np.random.randint(low=0, high=n_folds, size=zip_labels.size)
validation_fold = 0
is_set_dict = {
    "validation":fold_vec == validation_fold,
    "subtrain":fold_vec != validation_fold,
}

import torch
set_features = {}
set_labels = {}
for set_name, is_set in is_set_dict.items():
    set_features[set_name] = torch.from_numpy(
        zip_features[is_set,:]).float()
    set_labels[set_name] = torch.from_numpy(
        zip_labels[is_set]).float()
{set_name:array.shape for set_name, array in set_features.items()}

nrow, ncol = set_features["subtrain"].shape
class FullyConnected(torch.nn.Module):
    def __init__(self, n_hidden_units):
        super(FullyConnected, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(ncol, n_hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_units, 1))
    def forward(self, feature_mat):
        return self.seq(feature_mat)

n_pixels = int(np.sqrt(ncol))
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, out_channels=20, kernel_size=4, stride=4),
            torch.nn.ReLU(),
            # maybe max pooling.
            # another convolutional layer.
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(320,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,1))
    def forward(self, feature_mat):
        return self.seq(feature_mat)

loss_fun = torch.nn.BCEWithLogitsLoss()
class CSVtoImage(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __getitem__(self, item):
        return (
            self.features[item,:].reshape(1,n_pixels,n_pixels),
            self.labels[item])
    def __len__(self):
        return len(self.labels)

subtrain_dataset = CSVtoImage(
    set_features["subtrain"], set_labels["subtrain"])
[item.shape for item in subtrain_dataset[0]]
batch_size = 20
# random path through batches of subtrain data when shuffle=True
subtrain_dataloader = torch.utils.data.DataLoader(
    subtrain_dataset, batch_size=batch_size, shuffle=True)
for batch_features, batch_labels in subtrain_dataloader:
    print(batch_features.shape)    
conv_layer = torch.nn.Conv2d(
    in_channels=1, out_channels=20, kernel_size=4, stride=4, padding=0)
batch_labels.shape
batch_features.shape
conv_output = conv_layer(batch_features)
conv_output.shape
activation = torch.nn.ReLU()
act_out = activation(conv_output)
act_out.shape
pooling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=1)
pooling_out = pooling_layer(act_out)
pooling_out.shape
flatten_instance = torch.nn.Flatten(start_dim=1)
flat_out = flatten_instance(pooling_out)
n_data, n_hidden = flat_out.shape
conv_fully_connected = torch.nn.Linear(n_hidden, 100)
fully_out = conv_fully_connected(flat_out)
fully_out.shape

[p.shape for p in conv_layer.parameters()]
import math
def get_n_params(module):
    return sum(
        [math.prod(list(p.shape)) for p in module.parameters()])
n_conv_params = get_n_params(conv_layer)+get_n_params(conv_fully_connected)

batch_features_flat = flatten_instance(batch_features)
n_data, n_input_features = batch_features_flat.shape
n_big_hidden = math.floor(n_conv_weights/(n_input_features+1))
big_linear = torch.nn.Linear(n_input_features, n_big_hidden)

conv_model = ConvNet()
conv_model(batch_features)

class CleverConvNet(torch.nn.Module):
    def __init__(self):
        super(CleverConvNet, self).__init__()
        self.conv_seq = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, out_channels=20, kernel_size=4, stride=4),
            torch.nn.ReLU(),
            torch.nn.Flatten(start_dim=1),
            )
        conv_seq_out = self.conv_seq(batch_features)
        n_data, conv_hidden = conv_seq_out.shape
        print(conv_hidden)
        linear_hidden = 10
        self.linear_seq = torch.nn.Sequential(
            torch.nn.Linear(conv_hidden,linear_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(linear_hidden,1))
        self.seq = torch.nn.Sequential(
            self.conv_seq,
            self.linear_seq)
    def forward(self, feature_mat):
        return self.seq(feature_mat)
clever = CleverConvNet()
get_n_params(clever)
get_n_params(clever.conv_seq)
get_n_params(clever.linear_seq)

class Net(torch.nn.Module):
    def __init__(self, *units_per_layer):
        super(Net, self).__init__()
        seq_args = [torch.nn.Flatten(start_dim=1)]
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
net_hidden = 13
net=Net(ncol, net_hidden, net_hidden, 1)
get_n_params(net)
net(batch_features)
clever(batch_features)

def compute_loss(features, labels):
    pred_vec = model(features)
    return loss_fun(
        pred_vec.reshape(len(pred_vec)),
        labels)

opt_lr = {
    "SGD":0.05,
    "Adagrad":0.005,
    "Adam":0.0005,
    "RMSprop":0.0005
    }

loss_df_dict = {}
max_epochs=100
weight_decay = 0
opt_name="SGD"
model_dict = {
    "convolutional":CleverConvNet(),
    "fully_connected":Net(ncol, net_hidden, net_hidden, 1),
}
for model_name, model in model_dict.items():   
    if (model_name, max_epochs, "validation") not in loss_df_dict:
        torch.manual_seed(1)
        optimizer_class = getattr(torch.optim, opt_name)
        optimizer = optimizer_class(model.parameters(), lr=opt_lr[opt_name])
        for epoch in range(max_epochs+1):
            for batch_features, batch_labels in subtrain_dataloader:
                optimizer.zero_grad()
                batch_loss = compute_loss(batch_features, batch_labels)
                batch_loss.backward()
                optimizer.step()
            # then compute subtrain/validation loss.
            for set_name in set_features:
                feature_mat = set_features[set_name]
                label_vec = set_labels[set_name]
                feature_tensor = feature_mat.reshape(
                    len(label_vec),1,n_pixels,n_pixels)
                set_loss = compute_loss(feature_tensor, label_vec)
                set_df = pd.DataFrame({
                    "model_name":model_name,
                    "set_name":set_name,
                    "loss":set_loss.detach().numpy(),
                    "epoch":epoch,
                    }, index=[0])
                print(set_df)
                loss_df_dict[
                    (model_name,epoch,set_name)
                ] = set_df
loss_df = pd.concat(loss_df_dict.values())
# DF.groupby("set")["correct"].mean()*100
validation_df = loss_df.query('set_name == "validation"')
validation_df.iloc[validation_df["loss"].argmin(), :]

import plotnine as p9
gg = p9.ggplot()+\
    p9.geom_line(
        p9.aes(
            x="epoch",
            y="loss",
            color="set_name"
        ),
        data=loss_df)+\
    p9.facet_grid(". ~ model_name", labeller="label_both", scales="free")
show(gg)
