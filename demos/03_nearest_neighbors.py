import pandas as pd
import pdb
import numpy as np

zip_df = pd.read_csv(
    "~/teaching/cs570-spring-2022/data/zip.test.gz",
    header=None,
    sep=" ")

is01 = zip_df[0].isin([0,1])
zip01_df = zip_df.loc[is01,:]

zip_features = zip01_df.loc[:,1:].to_numpy()
zip_labels = zip01_df[0].to_numpy()

dir(np.random)
help(np.random.randint)
help(np.random.set_state)
np.random.seed(1)
n_folds = 5
fold_vec = np.random.randint(low=0, high=n_folds, size=zip_labels.size)

# just the first fold/split.
test_fold = 0
is_set_dict = {
    "test":fold_vec == test_fold,
    "train":fold_vec != test_fold,
}

# list comprehension.
[logical_array.sum() for logical_array in is_set_dict.values()]
[(set_name, logical_array.sum()) for set_name, logical_array in is_set_dict.items()]

result_list = []
for set_name, logical_array in is_set_dict.items():
    result_list.append( (set_name, logical_array.sum()) )
result_list

# dict comprehension
set_features = {
    set_name:zip_features[is_set,:]
    for set_name, is_set in is_set_dict.items()
}
{set_name:array.shape for set_name, array in set_features.items()}
set_labels = {
    set_name:zip_labels[is_set]
    for set_name, is_set in is_set_dict.items()
}

set_features = {}
set_labels = {}
for set_name, is_set in is_set_dict.items():
    set_labels[set_name] = zip_labels[is_set]
    set_features[set_name] = zip_features[is_set,:]

set_data = {}
for set_name, is_set in is_set_dict.items():
    set_data[set_name] = {
        "X":zip_features[is_set,:],
        "y":zip_labels[is_set]
        }

set_labels.keys()
train_label_counts = pd.Series(set_labels["train"]).value_counts()
featureless_pred_label = train_label_counts.idxmax()

class Featureless:
    def fit(self, X, y):
        train_features = X
        train_labels = y
        pdb.set_trace()
        train_label_counts = pd.Series(train_labels).value_counts()
        self.featureless_pred_label = train_label_counts.idxmax()
    def predict(self, X):
        test_features = X
        test_nrow, test_ncol = test_features.shape
        return np.repeat(self.featureless_pred_label, test_nrow)

## instantiation (creating an instance of a class)
featureless_instance = Featureless()
dir(featureless_instance)
featureless_instance.__class__        
Featureless
dir(featureless_instance.__class__)
featureless_instance.__class__
featureless_instance.__class__.__name__
featureless_instance.fit(set_features["train"], set_labels["train"])
dir(featureless_instance)
featureless_instance.featureless_pred_label
featureless_instance.predict(set_features["test"])
featureless_instance.predict(set_features["train"])

class MyKNN:
    pass

some_fun(**some_dict)

class MyCV:
    def __init__(self, estimator, param_grid, cv=3):
        pass
    def __init__(self, **kwargs):
        kwargs.setdefault("cv", 3)
        for key, value in kwargs.items():
            setattr(self, key, value)
            # self.something = value
    def fit(self, X, y):
        train_labels = y
        train_features = X
        fold_vec = np.random.randint(
            low=0, high=self.cv, size=zip_labels.size)
        for validation_fold in range(self.cv):
            is_set_dict = {
                "validation":fold_vec == test_fold,
                "subtrain":fold_vec != test_fold,
            }
            # ...
            for param_dict in self.param_grid:
                for param_name, param_value in param_dict.items():
                    setattr(self.estimator, param_name, param_value)
                    self.estimator.fit(**set_data["subtrain"])
                    self.estimator.predict(set_data["validation"]["X"])
                    # compute accuracy

mycv_instance = MyCV(
    estimator=MyKNN(), # instance of learner.
    param_grid=[ # grid of hyper-parameter values to search over.
        {'n_neighbors':n_neighbors} 
        for n_neighbors in
        range(20)
    ])
dir(mycv_instance)
mycv_instance.param_grid
mycv_instance.cv

list_of_learning_instances = [
    Featureless(),
    MyCV(...),
    GridSearchCV(...),
    LogisticRegressionCV()]
pred_dict = {
    "linear_model":something,
    "nearest_neighbors":some_other_preditions
}
pred_dict {}
for instance in list_of_learning_instances:
    pred_dict[instance.__class__.__name__] = instance.predict(test_features)

# nearest neighbors.
test_i = 0 # index of test observation.
test_i_features = set_features["test"][test_i,:]
diff_mat = set_features["train"] - test_i_features
squared_diff_mat = diff_mat ** 2
dir(diff_mat)
help(diff_mat.sum)
squared_diff_mat.sum(axis=0) # sum over columns, for each row.
distance_vec = squared_diff_mat.sum(axis=1) # sum over rows
sorted_indices = distance_vec.argsort()
n_neighbors = 3
nearest_indices = sorted_indices[:n_neighbors]
nearest_labels = set_labels["train"][nearest_indices]
