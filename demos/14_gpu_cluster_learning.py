import pandas as pd
import numpy as np
import os
import torch
none_or_str = os.getenv("SLURM_JOB_CPUS_PER_NODE")
CPUS = int(1 if none_or_str is None else none_or_str)
torch.set_num_interop_threads(CPUS)
torch.set_num_threads(CPUS)
device = "cuda" if torch.cuda.is_available() else "cpu"
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
    "~/teaching/cs499-599-fall-2022/data/zip.train.gz",
    header=None,
    sep=" ")
zip_71 = zip_df[ zip_df.iloc[:,0].isin([7,1]) ]
zip_labels_71 = zip_71.iloc[:,0].to_numpy()
zip_numpy_dict = {
    "X" : zip_71.iloc[:,1:257].to_numpy(),
    "y" : np.where(zip_labels_71==1, 1, 0),
    }
zip_tensor_dict = {
    data_type:torch.from_numpy(data_array).float().to(device)
    for data_type, data_array in zip_numpy_dict.items()
    }
np.random.seed(1)
n_folds = 5
fold_vec = np.random.randint(low=0, high=n_folds, size=zip_labels_71.size)
validation_fold = 0
is_set_dict = {
    "validation":fold_vec == validation_fold,
    "subtrain":fold_vec != validation_fold,
}

set_data_dict = {}
for set_name, is_set in is_set_dict.items():
    set_data_dict[set_name] = {
        "X":zip_tensor_dict["X"][is_set,:],
        "y":zip_tensor_dict["y"][is_set]
    }        
{set_name:x_and_y["X"].shape for set_name, x_and_y in set_data_dict.items()}
##learner.fit(**set_data_dict["train"])

nrow, ncol = zip_tensor_dict["X"].shape
n_pixels = int(np.sqrt(ncol))
loss_fun = torch.nn.BCEWithLogitsLoss()
class CSVtoImage(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    def __getitem__(self, item):
        return (
            self.features[item,:].reshape(1,n_pixels,n_pixels),
            self.labels[item])
    def __len__(self):
        return len(self.labels)

subtrain_dataset = CSVtoImage(**set_data_dict["subtrain"])
[item.shape for item in subtrain_dataset[0]]
batch_size = 20
# random path through batches of subtrain data when shuffle=True
subtrain_dataloader = torch.utils.data.DataLoader(
    subtrain_dataset, batch_size=batch_size, shuffle=True)
for batch_features, batch_labels in subtrain_dataloader:
    print(batch_features.shape)    
class CleverConvNet(torch.nn.Module):
    def __init__(self):
        super(CleverConvNet, self).__init__()
        self.conv_seq = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, out_channels=200, kernel_size=4, stride=4),
            torch.nn.ReLU(),
            torch.nn.Flatten(start_dim=1),
            ).to(device)
        conv_seq_out = self.conv_seq(batch_features)
        n_data, conv_hidden = conv_seq_out.shape
        linear_hidden = 1000
        self.linear_seq = torch.nn.Sequential(
            torch.nn.Linear(conv_hidden,linear_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(linear_hidden,1))
        self.seq = torch.nn.Sequential(
            self.conv_seq,
            self.linear_seq)
    def forward(self, feature_mat):
        return self.seq(feature_mat)
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
net_hidden = 130
def compute_loss(features, labels):
    pred_vec = model(features)
    return loss_fun(
        pred_vec.reshape(len(pred_vec)),
        labels)

loss_df_dict = {}
max_epochs=400
weight_decay = 0
opt_name="SGD"
cpu_model_dict = {
    "convolutional":CleverConvNet(),
    "fully_connected":Net(ncol, net_hidden, net_hidden, 1),
}
dev_model_dict = {
    model_name:model.to(device)
    for model_name, model in cpu_model_dict.items()
}
from time import time
before_grad_desc = time()
for model_name, model in dev_model_dict.items():   
    if (model_name, max_epochs, "validation") not in loss_df_dict:
        torch.manual_seed(1)
        optimizer_class = getattr(torch.optim, opt_name)
        optimizer = optimizer_class(model.parameters(), lr=0.1)
        for epoch in range(max_epochs+1):
            for batch_features, batch_labels in subtrain_dataloader:
                optimizer.zero_grad()
                batch_loss = compute_loss(batch_features, batch_labels)
                batch_loss.backward()
                optimizer.step()
            # then compute subtrain/validation loss.
            for set_name, x_and_y in set_data_dict.items():
                feature_mat = x_and_y["X"]
                label_vec = x_and_y["y"]
                feature_tensor = feature_mat.reshape(
                    len(label_vec),1,n_pixels,n_pixels)
                set_loss = compute_loss(feature_tensor, label_vec)
                set_df = pd.DataFrame({
                    "model_name":model_name,
                    "set_name":set_name,
                    "loss":set_loss.detach().cpu().numpy(),
                    "epoch":epoch,
                    }, index=[0])
                print(set_df)
                loss_df_dict[
                    (model_name,epoch,set_name)
                ] = set_df
after_grad_desc = time()
elapsed_seconds = after_grad_desc - before_grad_desc
loss_df = pd.concat(loss_df_dict.values())
# DF.groupby("set")["correct"].mean()*100
validation_df = loss_df.query('set_name == "validation"')
validation_df.iloc[validation_df["loss"].argmin(), :]
print("%s seconds of gradient descent loop"%elapsed_seconds)

loss_df.to_csv(device+".csv")

loss_df = pd.read_csv("~/teaching/cs499-599-fall-2022/demos/cuda.csv")

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
