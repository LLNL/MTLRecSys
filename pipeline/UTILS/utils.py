"""Generic object pickler and compressor

This module saves and reloads compressed representations of generic Python
objects to and from the disk.
"""

import pickle
import gzip
import numpy as np


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def save(object, filename, bin=1):
    """Saves a compressed object to disk
    """
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object, bin))
    file.close()


def load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    buffer = ""
    while 1:
        data = file.read()
        if data == "":
            break
        buffer += data
    object = pickle.loads(buffer)
    file.close()
    return object

def predMatrix(testData, pred):
    r, c = np.where(~np.isnan(testData))
    out = np.zeros(testData.shape)
    for idx, i, j in zip(range(len(r)), r, c):
        out[i, j] = pred[idx]
    return out

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def iterdict(d,s):
    for k,v in d.items():
        if isinstance(v, dict):
            ss = k.replace(' ','_')
            s = s + '_' + ss
            s = iterdict(v,s)
        else:
            if type(v) != type(np.array(0)):
                ss = k.replace(' ','_')
                s = s + '_' + ss + '_' + str(v)
    return s

def cleanupDatasetString(s):
    idx0 = s.find('[')
    idx1 = s.find(']')
    a = s[idx0+1:idx1]
    b = a.replace("'",'').replace(", ",'_')
    s = b + '_' + s[idx1+1:]
    s = s.replace('__drug_cell_ratings_data_train_x_y_test_x_y_trainCell_testCell_trainRatings_testRatings_trainDrug','')
    return s

def datasetParams2str(d):
    s = ''
    s = iterdict(d,s)
    return cleanupDatasetString(s)   
