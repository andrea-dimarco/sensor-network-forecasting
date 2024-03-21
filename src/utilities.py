import torch
import random
import numpy as np
import numpy as np
import pandas as pd
from typing import List
import matplotlib as mpl
from torch.nn import Module
from torch import Tensor, cat
import dataset_handling as dh
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from random import uniform, randint
from sklearn.decomposition import PCA

    

def plot_process(samples, labels:List[str]|None=None,
                 save_picture=False, show_plot=True,
                 img_idx=0, img_name:str="plot",
                 folder_path:str="./") -> None:
    '''
    Plots all the dimensions of the generated dataset.
    '''
    if save_picture or show_plot:
        for i in range(samples.shape[1]):
            if labels is not None:
                plt.plot(samples[:,i], label=labels[i])
            else:
                plt.plot(samples[:,i])

        # giving a title to my graph 
        if labels is not None:
            plt.legend()
        
        # function to show the plot 
        if save_picture:
            plt.savefig(f"{folder_path}{img_name}-{img_idx}.png")
        if show_plot:
            plt.show()
        plt.clf()


def compare_sequences(real: Tensor, fake: Tensor,
                      real_label:str="Real sequence", fake_label:str="Fake Sequence",
                      show_graph:bool=False, save_img:bool=False,
                      img_idx:int=0, img_name:str="plot", folder_path:str="./"):
    '''
    Plots two graphs with the two sequences.

    Arguments:
        - `real`: the first sequence with dimension [seq_len, data_dim]
        - `fake`: the second sequence with dimension [seq_len, data_dim]
        - `show_graph`: whether to display the graph or not
        - `save_img`: whether to save the image of the graph or not
        - `img_idx`: the id of the graph that will be used to name the file
        - `img_name`: the file name of the graph that will be used to name the file
        - `folder_path`: path to the folder where to save the image

    Returns:
        - numpy matrix with the pixel values for the image
    '''
    mpl.use('Agg')
    fig, (ax0, ax1) = plt.subplots(2, 1, layout='constrained')
    ax0.set_xlabel('Time-Steps')

    for i in range(real.shape[1]):
        ax0.plot(real.cpu()[:,i])
    ax0.set_ylabel(real_label)

    for i in range(fake.shape[1]):
        ax1.plot(fake.cpu()[:,i])
    ax1.set_ylabel(fake_label)

    if show_graph:
        plt.show()
    if save_img:
        plt.savefig(f"{folder_path}{img_name}-{img_idx}.png")


    # return picture as array
    canvas = fig.canvas
    canvas.draw()  # Draw the canvas, cache the renderer
    plt.clf()

    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    return image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)
    

def PCA_visualization(ori_data:torch.Tensor, generated_data:torch.Tensor, 
                               show_plot:bool=False, save_plot:bool=True,
                               folder_path:str="./", img_name:str="pca-visual"
                               ) -> None:
    """
    Using PCA for generated and original data visualization
     on both the original and synthetic datasets (flattening the temporal dimension).
     This visualizes how closely the distribution of generated samples
     resembles that of the original in 2-dimensional space

    Args:
    - `ori_data`: original data (num_sequences, seq_len, data_dim)
    - `generated_data`: generated synthetic data (num_sequences, seq_len, data_dim)
    - `show_plot`: display the plot
    - `save_plot`: save the .png of the plot
    - `folder_path`: where to save the file
    """  
    if show_plot or save_plot:
        # Data preprocessing
        N, data_dim = ori_data.size()  
        p = data_dim

        prep_data = ori_data.reshape(N,p).numpy()
        prep_data_hat = generated_data.reshape(N,p).numpy()
        
        # Visualization parameter        
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        red = ["red" for i in range(N)]
        blue = ["blue" for i in range(N)]
        f, ax = plt.subplots(1)    
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c=red, alpha = 0.2, label = "Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                    c=blue, alpha = 0.2, label = "Synthetic")

        ax.legend()  
        plt.title('Distribution comparison')
        if save_plot:
            plt.savefig(f"{folder_path}{img_name}.png")
        if show_plot:
            plt.show()
        plt.clf()


def set_seed(seed=0) -> None:
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    _ = pl.seed_everything(seed)


def save_timeseries(samples, folder_path:str="./", file_name="timeseries.csv") -> None:
    '''
    Save the samples as a csv file.
    '''
    # Save it
    df = pd.DataFrame(samples)
    df.to_csv(f"{folder_path}{file_name}", index=False, header=False)


def generate_data(datasets_folder="./datasets/"):
    '''
    Generate the required datasets for training and testing.
    '''
    from hyperparameters import Config
    from data_generation import sine_process, iid_sequence_generator, wiener_process
    import dataset_handling as dh
    from numpy import loadtxt, float32

    hparams = Config()
    print("Generating datasets.")
    if hparams.dataset_name in ['sine', 'wien', 'iid', 'cov']:
        # Generate and store the dataset as requested
        dataset_path = f"{datasets_folder}{hparams.dataset_name}_generated_stream.csv"
        if hparams.dataset_name == 'sine':
            sine_process.save_sine_process(p=hparams.data_dim,
                                           N=hparams.num_samples,
                                           file_path=dataset_path)
        elif hparams.dataset_name == 'wien':
            wiener_process.save_wiener_process(p=hparams.data_dim,
                                               N=hparams.num_samples,
                                               file_path=dataset_path)
        elif hparams.dataset_name == 'iid':
            iid_sequence_generator.save_iid_sequence(p=hparams.data_dim,
                                                     N=hparams.num_samples,
                                                     file_path=dataset_path)
        elif hparams.dataset_name == 'cov':
            iid_sequence_generator.save_cov_sequence(p=hparams.data_dim,
                                                     N=hparams.num_samples,
                                                     file_path=dataset_path)
        else:
            raise ValueError
        print(f"The {hparams.dataset_name} dataset has been succesfully created and stored into:\n\t- {dataset_path}")
    elif hparams.dataset_name == 'real':
        pass
    else:
        raise ValueError("Dataset not supported.")
    

    if hparams.dataset_name in ['sine', 'wien', 'iid', 'cov']:
        train_dataset_path = f"{datasets_folder}{hparams.dataset_name}_training.csv"
        test_dataset_path  = f"{datasets_folder}{hparams.dataset_name}_testing.csv"
        val_dataset_path   = f"{datasets_folder}{hparams.dataset_name}_validating.csv"

        # Train & Test
        dh.train_test_split(X=loadtxt(dataset_path, delimiter=",", dtype=float32),
                        split=hparams.train_test_split,
                        train_file_name=train_dataset_path,
                        test_file_name=test_dataset_path    
                        )
        
        # Train & Validation
        dh.train_test_split(X=np.loadtxt(train_dataset_path, delimiter=",", dtype=np.float32),
                        split=hparams.train_val_split,
                        train_file_name=train_dataset_path,
                        test_file_name=val_dataset_path    
                        )
        print(f"The {hparams.dataset_name} dataset has been split successfully into:\n\t- {train_dataset_path}\n\t- {val_dataset_path}")
    
    elif hparams.dataset_name == 'real':
        train_dataset_path = datasets_folder + hparams.train_file_name
        val_dataset_path   = datasets_folder + hparams.val_file_name
        test_dataset_path  = datasets_folder + hparams.test_file_name
        
    else:
        raise ValueError("Dataset not supported.")
    
    return train_dataset_path, val_dataset_path, test_dataset_path


def validate_model(model,
                   train_dataset_path:str,
                   test_dataset_path:str,
                   hparams:Config=Config(),
                   lookback:int=Config().lookback,
                   model_type:str='SSF'
                   ) -> None:
    '''
    Plots a graph with the predictions on the training set and on the test set.
    '''
    with torch.no_grad():
        model.eval()
        model.cpu()

        # On the TRAINING set
        if model_type == 'FFSF':
            dataset_train = dh.FeedDataset(file_path=train_dataset_path,
                                           lookback=hparams.lookback,
                                           privileged_lookback=hparams.privileged_lookback
                                           )
        elif model_type == 'PSF':
            dataset_train = dh.PrivilegedDataset(file_path=train_dataset_path,
                                                 lookback=hparams.lookback,
                                                 privileged_lookback=hparams.privileged_lookback
                                                 )
        elif model_type in ['SSF', 'GSF']:
            dataset_train = dh.RealDataset(file_path=train_dataset_path,
                                           lookback=hparams.lookback
                                           )
        else:
            raise ValueError
        print("Loaded training dataset.")
        
        horizon_train = min(int(hparams.plot_horizon/2), dataset_train.n_samples)
        synth_plot_train = np.ones((dataset_train.n_samples, dataset_train.data_dim)) * np.nan

        if model_type == 'PSF':
            y_pred = model(dataset_train.get_all_sequences(), dataset_train.get_all_pi()
                        ).reshape(-1,dataset_train.data_dim)
        elif model_type in ['SSF', 'GSF']:
            y_pred = model(dataset_train.get_all_sequences()
                        ).reshape(-1,dataset_train.data_dim)
        elif model_type == "FFSF":
            y_pred = model(dataset_train.get_all_sequences(), dataset_train.get_all_pi()
                        ).reshape(-1, dataset_train.data_dim)

        if model_type in ['SSF', 'PSF', 'GSF']:
            synth_plot_train[lookback:] = y_pred[+1:-1]
        elif model_type == 'FFSF':
            print("y_pred:", y_pred.shape)
            synth_plot_train = y_pred

        print("Predictions on training set done.")


        # On the TEST set
        if model_type == 'FFSF':
            dataset_test = dh.FeedDataset(file_path=test_dataset_path,
                                        lookback=hparams.lookback,
                                        privileged_lookback=hparams.privileged_lookback
                                        )
        elif model_type == 'PSF':
            dataset_test = dh.PrivilegedDataset(file_path=test_dataset_path,
                                        lookback=hparams.lookback,
                                        privileged_lookback=hparams.privileged_lookback
                                        )
        elif model_type in ['SSF', 'GSF']:
            dataset_test = dh.RealDataset(file_path=test_dataset_path,
                                    lookback=hparams.lookback
                                    )
        else:
            raise ValueError
        print("Loaded testing dataset.")

        horizon_test = min(int(hparams.plot_horizon/2), dataset_test.n_samples)
        synth_plot_test = np.ones((dataset_train.n_samples+dataset_test.n_samples, dataset_test.data_dim)) * np.nan
        plot_test = np.ones((dataset_train.n_samples+dataset_test.n_samples, dataset_test.data_dim)) * np.nan

        if model_type == 'FFSF':
            plot_test = plot_test.reshape(-1, dataset_test.data_dim)
            plot_test[dataset_train.n_samples:] = dataset_test.get_all_targets()
            synth_plot_test = synth_plot_test.reshape(-1, dataset_test.data_dim)
        else:
            plot_test[dataset_train.n_samples:] = dataset_test.get_whole_stream()

        if model_type == 'PSF':
            y_pred = model(dataset_test.get_all_sequences(), dataset_test.get_all_pi()
                        ).reshape(-1,dataset_train.data_dim)
            
        elif model_type in ['SSF', 'GSF']:
            y_pred = model(dataset_test.get_all_sequences()
                        ).reshape(-1,dataset_train.data_dim)

        elif model_type == "FFSF":
            y_pred = model(dataset_test.get_all_sequences(), dataset_test.get_all_pi()
                        ).reshape(-1, dataset_test.data_dim)

        if model_type in ['SSF', 'PSF', 'GSF']:
            synth_plot_test[dataset_train.n_samples+lookback:] = y_pred[lookback+1:-1]
        elif model_type == 'FFSF':
            synth_plot_test[dataset_train.n_samples:] = y_pred

        print("Predictions on testing set done.")

        #plt.figure(figsize=(50,20),dpi=300)
        plt.grid(True)
        fig,ax = plt.subplots()
        ax.grid(which = "major", linewidth = 1)
        ax.grid(which = "minor", linewidth = 0.2)
        #plt.minorticks_on()
        fig.set_size_inches(18.5, 10.5)
        ax.minorticks_on()
        # Only plot the first dimension
        plt.plot(dataset_train.get_whole_stream()[:horizon_train,0], c='b')
        plt.plot(synth_plot_train[:horizon_train,0], c='r')

        plt.plot(plot_test[dataset_train.n_samples-horizon_test : dataset_train.n_samples+horizon_test,0], c='b')
        plt.plot(synth_plot_test[dataset_train.n_samples-horizon_test : dataset_train.n_samples+horizon_test,0], c='g')

        print("Plot done.")
        plt.savefig(f"img/{model_type}-{hparams.n_epochs}-e-{hparams.hidden_dim}-hs-{hparams.seed}-seed.png",dpi=300)
        plt.show()