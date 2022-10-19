import pandas as pd
import numpy as np

airfoil_df = pd.read_csv(
    "~/teaching/cs499-599-fall-2022/data/airfoil_self_noise.tsv",
    sep="\t")
out_name = 'Scaled_sound_pressure_level.decibels'
is_output = airfoil_df.columns == out_name
airfoil_features = airfoil_df.iloc[:,is_output==False].to_numpy()
airfoil_labels = airfoil_df.iloc[:,is_output].to_numpy()
# 1. feature scaling.
airfoil_mean = airfoil_features.mean(axis=0)
airfoil_sd = np.sqrt(airfoil_features.var(axis=0))
scaled_features = (airfoil_features-airfoil_mean)/airfoil_sd
scaled_features.mean(axis=0)
scaled_features.var(axis=0)
label_mean = airfoil_labels.mean()
label_sd = np.sqrt(airfoil_labels.var())
scaled_labels = (airfoil_labels-label_mean)/label_sd
scaled_labels.mean()
scaled_labels.var()

# 2. subtrain/validation split.
np.random.seed(1)
n_folds = 5
fold_vec = np.random.randint(low=0, high=n_folds, size=airfoil_labels.size)
validation_fold = 1
is_set_dict = {
    "validation":fold_vec == validation_fold,
    "subtrain":fold_vec != validation_fold,
}

set_features = {}
set_labels = {}
for set_name, is_set in is_set_dict.items():
    set_features[set_name] = scaled_features[is_set,:]
    set_labels[set_name] = scaled_labels[is_set]
{set_name:array.shape for set_name, array in set_features.items()}

nrow, ncol = set_features["subtrain"].shape
feature_mat = set_features["subtrain"]
label_vec = set_labels["subtrain"].reshape(nrow, 1)
weight_vec = np.random.randn(ncol).reshape(ncol, 1)

class InitialNode:
    def __init__(self, value, name):
        self.value = value
        self.name = name
    def backward(self):
        print("backward from "+self.name)

weight_node = InitialNode(weight_vec, "weight")#w
feature_node =InitialNode(feature_mat, "feature")#h
label_node = InitialNode(label_vec, "label") #y

class Operation:
    def __init__(self, *node_list):
        for index in range(len(node_list)):
            setattr(self, self.input_names[index], node_list[index])
        self.value = self.get_value()
    def backward(self):
        grad_tuple = self.gradient()
        # store each gradient in the corresponding parent_node.grad
        for index in range(len(grad_tuple)):
            parent_node = getattr(self, self.input_names[index])
            parent_node.grad = grad_tuple[index]
            parent_node.backward()
        
class mean_squared_error(Operation):
    input_names = ["pred_vec", "label_vec"]
    def get_value(self):
        return np.mean(
            (self.label_vec.value - self.pred_vec.value)**2)
    def gradient(self):
        return [
            (self.pred_vec.value-self.label_vec.value)*(
                2/len(self.label_vec.value)),
            "gradient with respect to labels"]

class mm(Operation):
    input_names = ["features", "weights"]
    def get_value(self):
        return np.matmul(self.features.value, self.weights.value)
    def gradient(self):
        return [
            np.matmul(self.grad, self.weights.value.T),
            np.matmul(self.features.value.T, self.grad)]

class relu(Operation):
    input_names = ["features_before_activation"]
    def get_value(self):
        return np.where(
            features_before_activation.value > 0, 
            features_before_activation.value, 0)
    def gradient(self):
        return [
            np.where(self.features_before_activation < 0, 0, self.grad)]

pred_node = mm(feature_node, weight_node)
loss_node = mean_squared_error(pred_node, label_node)
loss_node.backward()
pred_node.grad.shape
weight_node.value.shape
weight_node.grad
# gradient descent (one step)
step_size=0.1
weight_node.value -= step_size * weight_node.grad

class AutoReg:
    def __init__(self, step_size=0.1, max_epochs=100):
        self.step_size = step_size
        self.max_epochs = max_epochs
        self.weight_vec = np.random.randn(ncol).reshape(ncol, 1)/1000
        self.weight_node = InitialNode(self.weight_vec, "weight")#w
    def get_loss_node(self, feature_mat, label_vec):
        feature_node =InitialNode(feature_mat, "feature")#h
        label_node = InitialNode(label_vec, "label") #y
        pred_node = mm(feature_node, self.weight_node)
        return mean_squared_error(pred_node, label_node)
    def take_step(self, feature_mat, label_vec):
        "Updates weights with weight grad"
        self.loss_node = self.get_loss_node(feature_mat, label_vec)
        self.loss_node.backward()
        self.weight_node.value -= self.step_size * self.weight_node.grad
    def fit(self, train_features, train_labels):
        loss_df_list = []
        for epoch in range(self.max_epochs):
            self.take_step(train_features, train_labels)
            for set_name in set_features:
                set_X = set_features[set_name]
                set_y = set_labels[set_name]
                loss_node = self.get_loss_node(set_X, set_y)
                loss_df_list.append(pd.DataFrame({
                    "epoch":epoch,
                    "loss":loss_node.value,
                    "set_name":set_name
                }, index=[0]))
        self.loss_df = pd.concat(loss_df_list)
learner_obj = AutoReg()
learner_obj.weight_vec
learner_obj.take_step(set_features["subtrain"], set_labels["subtrain"])
learner_obj.loss_node.value
learner_obj.fit(set_features["subtrain"], set_labels["subtrain"])
learner_obj.loss_df

import plotnine as p9
gg = p9.ggplot()+\
    p9.geom_line(
        p9.aes(
            x="epoch",
            y="loss",
            color="set_name",
        ),
        data=learner_obj.loss_df)
gg.save("08_regression.png")
