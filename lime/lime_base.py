"""
Contains abstract functionality for learning locally linear sparse model.
"""
import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path,LinearRegression,SGDRegressor
from sklearn.utils import check_random_state
from Losses.listNet import listNet 

import torch
import torch.nn as nn
import torch.optim as optim

class WeightedRegression(nn.Module):
    def __init__(self, input_dim, ndocs):
        super(WeightedRegression, self).__init__()
        self.linear = nn.Linear(input_dim, ndocs)  

    def forward(self, x):
        return self.linear(x)

def custom_loss_function(predicted_ranking, actual_ranking, weight):
    loss = listNet(predicted_ranking, actual_ranking, weight)
    return loss

class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            
            nvocab = data.shape[1]
            ndocs = labels.shape[1]
            fs_model = WeightedRegression(nvocab,ndocs)
            optimizer = optim.SGD(fs_model.parameters(), lr=0.01)

            X_train = torch.from_numpy(data).float()
            y_train = torch.from_numpy(labels)
            print('X_train: ',X_train.shape)
            print('y_train: ',y_train.shape)

            for epoch in range(1000):
                fs_model.train()
                optimizer.zero_grad()

                predictions = fs_model(X_train)

                loss = custom_loss_function(predictions, y_train, weights)

                loss.backward()
                optimizer.step()

                if epoch % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {loss.item()}')


            coef =  fs_model.linear.weight.data.mean(dim=0)
            print('coef: ',coef.shape)

            
            weighted_data = coef * data[0]
            feature_weights = sorted(
                zip(range(data.shape[1]), weighted_data),
                key=lambda x: np.abs(x[1]),
                reverse=True)
            return np.array([x[0] for x in feature_weights[:num_features]])
        
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """


        nvocab = neighborhood_data.shape[1]
        ndocs = neighborhood_labels.shape[1]
        weights = self.kernel_fn(distances) 


        used_features = self.feature_selection(neighborhood_data,
                                               neighborhood_labels,
                                               weights,
                                               num_features,
                                               feature_selection) 
        
        n_features = len(used_features)
        
        model = WeightedRegression(n_features,ndocs)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    
        X_train = torch.from_numpy(neighborhood_data[:, used_features]).float()
        y_train = torch.from_numpy(neighborhood_labels)
       
      
        for epoch in range(1000):
            model.train()
            optimizer.zero_grad()

            predictions = model(X_train)

            loss = custom_loss_function(predictions, y_train, weights)

            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        
        model_output = model(X_train)
        prediction_score = custom_loss_function(model_output, y_train, weights).detach().numpy()
       
        local_pred = model(X_train[0]).detach().numpy()

       
        coefs = model.linear.weight.data.mean(dim=0).numpy()
        intercept = model.linear.bias.detach().numpy()

        if self.verbose:
            print('Intercept', intercept)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0])


        return (intercept,
                sorted(zip(used_features, coefs),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)
