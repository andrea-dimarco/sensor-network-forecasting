import warnings
warnings.filterwarnings("ignore")


import torch
import random
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from hyperparameters import Config
from data_generation.wiener_process import multi_dim_wiener_process
from sklearn.ensemble import GradientBoostingClassifier


def create_dataset(dataset:np.ndarray,
                   lookback:int,
                   discretization:int=Config().discretization
                   ):
    """
    Transform a time series into a prediction dataset
    
    Args:
        `dataset`: A numpy array of time series, first dimension is the time steps
        `lookback`: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback].reshape(-1) # ( lookback*data_dim )
        # TODO: is this index correct
        step = dataset[i+lookback] # ( data_dim )
        # TODO: compute std and mean on the whole dataset
        std = feature.std()
        mu = feature.mean()
        target = [] # ( data_dim, discretization )
        for sensor in step:
            classification = [0 for i in range(discretization*2+1)] # ( discretization )
            c = int((sensor-mu)/std)
            # lower bound
            c = max(c, -discretization)
            # upper bound
            c = min(c, discretization)
            classification[c+discretization] = 1
            # discretization = 2
            # [1, 0, 0, 0, 0] (-inf)*std < x < (-2)*std
            # [0, 1, 0, 0, 0]   (-2)*std < x < (-1)*std
            # [0, 0, 1, 0, 0]   (-1)*std < x < (+0)*std
            # [0, 0, 1, 0, 0]   (+0)*std < x < (+1)*std
            # [0, 0, 0, 1, 0]   (+1)*std < x < (+2)*std
            # [0, 0, 0, 0, 1]   (+2)*std < x < (+inf)*std
            target.append(classification)
        X.append(feature) 
        y.append(target)
    X = torch.tensor(X).type(torch.float32) # ( n_seq, lookback*data_dim )
    y = torch.tensor(y).type(torch.float32) # ( n_seq, data_dim, discretization )
    return X, y


def get_data(verbose=True):
    '''
    Gets and returns the datasets as torch.Tensors
    '''
    hparams = Config()
    lookback = hparams.lookback
    dataset_name = hparams.dataset_name

    # LOAD DATASET
    if dataset_name == 'wien':
        dataset = multi_dim_wiener_process(p=hparams.data_dim, N=hparams.num_samples)

    elif dataset_name == 'real':
        dataset_path = "./datasets/sensor_data_multi.csv"
        dataset = np.loadtxt(dataset_path, delimiter=",", dtype=np.float32)
        n_samples = dataset.shape[0]

    else:
        raise ValueError

    n_samples = dataset.shape[0]
    try:
        data_dim = dataset.shape[1]
    except:
        data_dim = 1
        dataset = dataset.reshape(n_samples, data_dim)

    train_size = int(n_samples*hparams.train_test_split)
    train = dataset[:train_size]
    test = dataset[train_size:]


    # SPLIT DATASET
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)


    # CONVERT TO TORCH
    dataset = torch.from_numpy(dataset).type(torch.float32)
    train = torch.from_numpy(train).type(torch.float32)
    test = torch.from_numpy(test).type(torch.float32)


    # RETURN RESULTS
    if verbose:
        print(f"Training Features: {X_train.size()}, Training Targets: {y_train.size()}")
        print(f"Testing Features: {X_test.size()}, Testing Targets: {y_test.size()}")
        print(f"Shape Train: ( num_sequences, lookback*data_dim )\nShape Test: ( num_sequences, data_dim, discretization )")

    return X_train, y_train, X_test, y_test


def set_seed(seed=0):
    '''
    Sets the global seed
    
    Arguments:
        - `seed`: the seed to be set
    '''
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # Can have performance impact
    torch.backends.cudnn.benchmark = False

    _ = pl.seed_everything(seed)


def show_confusion_matrix(actual:torch.Tensor, 
                          predicted:torch.Tensor,
                          model_name:str='model'
                          ) -> np.ndarray:
    '''
    Computes and displays confusion matrix
    '''
    cm = confusion_matrix(actual,predicted,normalize='pred')
    discretization = Config.discretization
    labels = [i for i in range(-discretization,discretization+1)]
    sns.heatmap(cm * 100, 
                annot=True,
                fmt='g', 
                xticklabels=labels,
                yticklabels=labels)
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)

    plt.savefig(f"img/{model_name}_confusion_matrix.png")
    plt.show()

    f1_val = f1_score(actual,predicted,average=None,labels=labels)
    precision_val = precision_score(actual, predicted, average=None, labels=labels)
    recall = recall_score(actual, predicted, average=None, labels=labels)
    print("Precision: ", precision_val)
    print("Recall:    ", recall)
    print("F1 score:  ", f1_val)
    return cm


if __name__ == '__main__':
    # setup
    hparams = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(hparams.seed)

    X_train, y_train, X_test, y_test = get_data()

    if (y_train.size()[1] > 1):
        print("Can only handle one sensor at a time!!")
        import os
        os._exit(0)

    _, y_train = torch.max(y_train, 2) # ( n_seq, data_dim, discretizaton )
    y_train -= hparams.discretization # ( n_seq, data_dim )
    _, y_test = torch.max(y_test, 2)
    y_test -= hparams.discretization
    
    # Fit Tree
    print("Fitting Gradient Boost Adversary.")
    clf = GradientBoostingClassifier(n_estimators=hparams.n_estimators,
                                     learning_rate=hparams.boost_lr,
                                     max_depth=hparams.max_depth,
                                     random_state=hparams.seed
                                     ).fit(X_train, y_train)
    
    print("Gradient Boost score: ", clf.score(X_test, y_test))
        
    # Validation  
    predicted = clf.predict(X_test)
    show_confusion_matrix(actual=y_test,
                          predicted=predicted,
                          model_name='ADV'
                          )