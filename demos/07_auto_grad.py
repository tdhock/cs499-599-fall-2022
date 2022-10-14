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
feature_mat = set_features["subtrain"]
label_vec = set_labels["subtrain"]
weight_vec = np.random.randn(ncol)

class InitialNode:
    def __init__(self, value):
        self.value = value

weight_node = InitialNode(weight_vec)
feature_node =InitialNode(feature_mat)
label_node = InitialNode(label_vec)

class Operation:
    def __init__(self, *node_list):
        for index in range(len(node_list)):
            setattr(self, self.input_names[index], node_list[index])
        self.value = self.get_value()
    def backward(self):
        grad_tuple = self.gradient()
        # store each gradient in the corresponding parent_node.grad
        # call parent_node.backward()
        
class mean_logistic_loss(Operation):
    input_names = ["pred_vec", "subtrain_labels"]
    def get_value(self):
        return np.mean(np.log(
            1+np.exp(-self.subtrain_labels.value * self.pred_vec.value)))
    def gradient(self):
        return [
            -self.subtrain_labels.value/(
                1+np.exp(self.subtrain_labels.value * self.pred_vec.value)),
            "gradient with respect to labels"]

class mm(Operation):
    input_names = ["features", "weights"]
    def get_value(self):
        return np.matmul(self.features.value, self.weights.value)

class relu(Operation):
    input_names = ["features_before_activation"]
    def get_value(self):
        return np.where(
            features_before_activation.value > 0, 
            features_before_activation.value, 0)

pred_node = mm(feature_node, weight_node)
loss_node = mean_logistic_loss(pred_node, label_node)
loss_node.backward()
weight_node.grad
act_node = relu(pred_node)
