
import numpy as np
from UTILS import config, performance_metrics, utils


def cross_validation_score(method, dataset, num_folds=10):
    """ Cross-validation """
    scores = np.zeros((num_folds, ))

    for i_fold in range(num_folds):

        # shuffle and re-split the data between training and test
        dataset.shuffle_and_split()

        # check model type: feature_based or non_feature_based
        if method.type == 'feature_based':
            if method.paradigm == 'stl':
                perf = list()
                for k in dataset.datasets:
                    method.fit(dataset.data['train']['x'][k],
                               y=dataset.data['train']['y'][k],
                               cat_point=dataset.cat_point)
                    y_pred = method.predict(dataset.data['test']['x'][k])
                    y_true = dataset.testRatings[k]
                    if method.output_shape == 'array':
                        y_pred = utils.predMatrix(y_true, y_pred)  # move back to rating matrix
                    # dict to save performance metrics for the t-th task
                    perf.append(performance_metrics.rmse(y_pred, y_true))
            elif method.paradigm == 'mtl':
                method.fit(dataset.data['train']['x'],
                           dataset.data['train']['y'],
                           cat_point=dataset.cat_point)
                y_pred = method.predict(dataset.data['train']['x'])
                perf = list()
                for k in dataset.datasets:
                    y_true_k = dataset.testRatings[k]
                    y_pred_k = utils.predMatrix(y_true_k, y_pred[k])  # move back to ratings matrix
                    perf.append(performance_metrics.rmse(y_pred_k, y_true_k))
            else:
                raise ValueError('Unknown paradigm: {}'.format(method.paradigm))
        elif method.type == 'non_feature_based':
            if method.paradigm == 'stl':
                perf = list()
                for k in dataset.datasets:
                    method.fit(dataset.trainRatings[k])
                    y_pred_k = method.predict(dataset.testRatings[k])
                    y_true_k = dataset.testRatings[k]
                    if method.output_shape == 'array':
                        y_pred_k = utils.predMatrix(y_true_k, y_pred_k)  # move back to rating matrix
                    # store results to dict of all performances
                    perf.append(performance_metrics.rmse(y_pred_k, y_true_k))
            elif method.paradigm == 'mtl':
                method.fit(dataset.trainRatings)
                y_pred = method.predict(dataset.testRatings)
                perf = list()
                for k in dataset.datasets:
                    y_true_k = dataset.testRatings[k]
                    if method.output_shape == 'array':
                        y_pred_k = utils.predMatrix(y_true_k, y_pred[k])  # move back to rating matrix
                    perf.append(performance_metrics.rmse(y_pred_k, y_true_k))
            else:
                raise ValueError('Unknown paradigm: {}'.format(method.paradigm))
        else:
            raise ValueError('Unknown type %s' % (method.type))

        perf = np.array(perf)
        scores[i_fold] = np.mean(perf)

    return scores.mean()
