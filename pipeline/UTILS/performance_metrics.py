import numpy as np
import sklearn.metrics 
from sklearn.preprocessing import StandardScaler


def score_to_exact_rank(s):
    return (-1 * s).argsort().argsort()


def ncdg(pred, R):
    all_ndcg = []
    if len(R.shape) == 1:
        test_drug_bool = ~np.isnan(R)
        if test_drug_bool.sum() != 0:
            s_u = R[test_drug_bool]
            r_u = score_to_exact_rank(s_u)
            s_u_pred = pred[test_drug_bool]
            r_u_pred = score_to_exact_rank(s_u_pred)
            G_u_max = np.sum((np.power(2, s_u)) / np.log(r_u + 2))
            G_u = np.sum((np.power(2, s_u)) / np.log(r_u_pred + 2))
            # print G_u, G_u_max, G_u / G_u_max
            all_ndcg = [G_u / G_u_max]
    else:
        for u in range(R.shape[0]):
            test_drug_bool = ~np.isnan(R[u, :])
            if test_drug_bool.sum() != 0:
                s_u = R[u, :][test_drug_bool]
                r_u = score_to_exact_rank(s_u)
                s_u_pred = pred[u, :][test_drug_bool]
                r_u_pred = score_to_exact_rank(s_u_pred)
                G_u_max = np.sum((np.power(2, s_u)) / np.log(r_u + 2))
                G_u = np.sum((np.power(2, s_u)) / np.log(r_u_pred + 2))
                # print G_u, G_u_max, G_u / G_u_max
                all_ndcg += [G_u / G_u_max]
    return np.mean(all_ndcg)

# def ndcg2(pred, R):
#     pred, R = pred.flatten(), R.flatten()
#     pred, R =  pred[~np.isnan(pred)], R[~np.isnan(R)]
#     print(pred.shape, R.shape)
#     return ndcg_score(pred,R)


def rmse(pred, R):
    inds = np.where(~np.isnan(R))
    err = 0
    if len(inds[0]) > 0:
        rvals = R[inds]
        predvals = pred[inds]
        err = np.sqrt((1 / len(rvals)) * np.sum((predvals - rvals) ** 2))
    return err

def r2(pred, R):
    inds = np.where(~np.isnan(R))
    err = 0
    if len(inds[0]) > 0:
        rvals = R[inds]
        predvals = pred[inds]
        err = sklearn.metrics.r2_score(rvals, predvals)
    return err

def explained_variance_score(pred, R):
    inds = np.where(~np.isnan(R))
    err = 0
   
    if len(inds[0]) > 0:
        rvals = R[inds]
        predvals = pred[inds]
        err = sklearn.metrics.explained_variance_score(rvals.reshape(-1,1), predvals.reshape(-1,1))
    return err

def mae(pred, R):
    inds = np.where(~np.isnan(R))
    err = 0
    if len(inds[0]) > 0:
        rvals = R[inds]
        predvals = pred[inds]
        err = sklearn.metrics.mean_absolute_error(rvals, predvals)
    return err

def rmse_by_line(pred, R):
    """RMSE of predictions by cell line"""
    err = np.array([])
    # print(pred.shape, "PRED SHAPE")
    R = np.nan_to_num(R)  # default arg that converts nans to 0's so I can preserve array shape
    pred = np.nan_to_num(pred)
    resids = R - pred
    num_lines = resids.shape[0]     # cell lines resiudals
    for i in range(num_lines):
        err = np.append(err, np.sqrt((1 / num_lines) * np.sum(resids[i]) ** 2))
    # print(err.shape,"Err shape")
    return err.tolist()


def rmse_by_drug(pred, R):
    """RMSE of predictions by drug"""
    err = np.array([])
    R = np.nan_to_num(R)  # default arg that converts nans to 0's so I can preserve array shape
    pred = np.nan_to_num(pred)
    resids = R - pred
    num_lines = resids.shape[1]  # cell lines resiudals
    for i in range(num_lines):
        err = np.append(err, np.sqrt((1 / num_lines) * np.sum(resids[:,i]) ** 2))
    return err.tolist()

def rmse_for_NCF(pred, R):
    """custom RMSE for NCF function which assigns 1 to a positive drug/cell line
    and 0 for either no interaction or negative
    inputs
    -------------------------------------------------------------------------------
    pred: list of predicted values between 0 and 1
    R: list of actual IC50 thresholds (lower IC50 is better)

    output
    --------------------------------------------------------------------------------
    rmse score after embedding R
    TODO: get keras to produce outputs on correct magnitude so I don't have to have a magic number
    or go back to how the paper is using the model
    """
    inds = np.where(~np.isnan(R))
    err = 0
    if len(inds[0]) > 0:
        rvals = R[inds]
        predvals = pred[inds] 
        err = np.sqrt((1 / len(rvals)) * np.sum((predvals - rvals) ** 2))
    return err
