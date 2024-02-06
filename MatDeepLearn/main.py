import os
import argparse
import time
import csv
import sys
import json
import random
import numpy as np
import pprint
import yaml as pyyaml
import torch
import torch.multiprocessing as mp
import ray
from ray import tune
from matdeeplearn import models, process, training
from sklearn.metrics import r2_score
import pandas as pd
from matplotlib import pyplot as plt
import scipy as sp
import math
from math import inf
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import warnings
import os
import shutil
import yaml

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
   
   
################################################################################
#  MatDeepLearn code
################################################################################

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="MatDeepLearn inputs")
    
    ###Job arguments (Config File Path, Run Mode, Job Name, Model, Seed, Model Path, Save/Load Model, Write Oupout, Parallel, Reprocess) 
    parser.add_argument("--config_path", default="config.yml", type=str, help="Location of config file (default: config.yml)",)
    parser.add_argument("--run_mode", default=None,type=str, help="run modes: Training, Predict, Repeat, CV, Hyperparameter, Ensemble, Analysis",)
    parser.add_argument("--job_name", default=None, type=str, help="name of your job and output files/folders",)
    parser.add_argument("--model", default=None, type=str, help="CGCNN_demo, MPNN_demo, SchNet_demo, MEGNet_demo, GCN_demo",)
    parser.add_argument("--seed", default=None, type=int, help="seed for data split, 0=random",)
    parser.add_argument("--model_path", default=None, type=str, help="path of the model .pth file",)
    parser.add_argument("--save_model", default=None, type=str, help="Save model",)
    parser.add_argument("--load_model", default=None, type=str, help="Load model",)
    parser.add_argument("--write_output", default=None, type=str, help="Write outputs to csv",)
    parser.add_argument("--parallel", default=None, type=str, help="Use parallel mode (ddp) if available",)
    parser.add_argument("--reprocess", default=None, type=str, help="Reprocess data since last run",)
    
    ###Processing arguments (Data Path, Format (Default JSON), Feature Vector)
    parser.add_argument("--data_path", default=None, type=str, help="Location of data containing structures (json or any other valid format) and accompanying files",)
    parser.add_argument("--format", default=None, type=str, help="format of input data")
    parser.add_argument("--features", default="False", type=bool, help="format of input data")
    
    ###Training arguments (Train/Valid/Test Split)
    parser.add_argument("--train_ratio", default=None, type=float, help="train ratio")
    parser.add_argument("--val_ratio", default=None, type=float, help="validation ratio")
    parser.add_argument("--test_ratio", default=None, type=float, help="test ratio")
    parser.add_argument("--verbosity", default=None, type=int, help="prints errors every x epochs")
    parser.add_argument("--target_index", default=None, type=int, help="which column to use as target property in the target file",)
    
    ###Model arguments (Epochs, Btach Size, Learning Rate, Regularization Rate/Function, Loss Function)
    parser.add_argument("--epochs", default=None, type=int, help="number of total epochs to run",)
    parser.add_argument("--batch_size", default=None, type=int, help="batch size")
    parser.add_argument("--lr", default=None, type=float, help="learning rate")
    parser.add_argument("--reg", default=None, type=float, help="regularization rate")
    parser.add_argument("--reg_funct", default="L2", type=str, help="regularization function")
    parser.add_argument("--loss", default="l1_loss", type=str, help="regularization function")

    ##Get arguments from command line
    args = parser.parse_args(sys.argv[1:])

    ##Open provided config file
    assert os.path.exists(args.config_path), ("Config file not found in " + args.config_path)
    with open(args.config_path, "r") as ymlfile: config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    ##Update config values from command line
    if args.run_mode != None:
        config["Job"]["run_mode"] = args.run_mode
    run_mode = config["Job"].get("run_mode")
    config["Job"] = config["Job"].get(run_mode) 
    if config["Job"] == None:
        print("Invalid run mode")
        sys.exit()
    if args.job_name != None:
        config["Job"]["job_name"] = args.job_name
    if args.model != None:
        config["Job"]["model"] = args.model
    if args.seed != None:
        config["Job"]["seed"] = args.seed
    if args.model_path != None:
        config["Job"]["model_path"] = args.model_path
    if args.load_model != None:
        config["Job"]["load_model"] = args.load_model
    if args.save_model != None:
        config["Job"]["save_model"] = args.save_model
    if args.write_output != None:
        config["Job"]["write_output"] = args.write_output
    if args.parallel != None:
        config["Job"]["parallel"] = args.parallel
    if args.reprocess != None:
        config["Job"]["reprocess"] = args.reprocess
    if args.data_path != None:
        config["Processing"]["data_path"] = args.data_path
    if args.features != None:
        config["Processing"]["features"] = args.features
    if args.format != None:
        config["Processing"]["data_format"] = args.format
    if args.train_ratio != None:
        config["Training"]["train_ratio"] = args.train_ratio
    if args.val_ratio != None:
        config["Training"]["val_ratio"] = args.val_ratio
    if args.test_ratio != None:
        config["Training"]["test_ratio"] = args.test_ratio
    if args.verbosity != None:
        config["Training"]["verbosity"] = args.verbosity
    if args.target_index != None:
        config["Training"]["loss"] = args.loss
    if args.target_index != None:
        config["Training"]["target_index"] = args.target_index
    for key in config["Models"]:
        if args.epochs != None:
            config["Models"][key]["epochs"] = args.epochs
        if args.batch_size != None:
            config["Models"][key]["batch_size"] = args.batch_size
        if args.lr != None:
            config["Models"][key]["lr"] = args.lr
        if args.reg != None:
            config["Models"][key]["reg"] = args.reg
        if args.reg != None:
            config["Models"][key]["reg_funct"] = args.reg_funct
    if run_mode == "Predict":
        config["Models"] = {}
    elif run_mode == "Ensemble":
        config["Job"]["ensemble_list"] = config["Job"]["ensemble_list"].split(",")
        models_temp = config["Models"]
        config["Models"] = {}
        for i in range(0, len(config["Job"]["ensemble_list"])):
            config["Models"][config["Job"]["ensemble_list"][i]] = models_temp.get(config["Job"]["ensemble_list"][i])
    else:
        config["Models"] = config["Models"].get(config["Job"]["model"])
    if config["Job"]["seed"] == 0:
        config["Job"]["seed"] = np.random.randint(1, 1e6)

    ################################################################################
    #  Begin processing
    ################################################################################

    if run_mode != "Hyperparameter":
        process_start_time = time.time()
        dataset = process.get_dataset(config["Processing"]["data_path"], config["Training"]["target_index"], config["Processing"]["features"], config["Job"]["reprocess"], config["Processing"],)

    ################################################################################
    #  Training begins
    ################################################################################

    ##Regular training
    if run_mode == "Training":
        world_size = torch.cuda.device_count()
        if world_size == 0:
            training.train_regular("cpu", world_size, config["Processing"]["data_path"], config["Processing"]["features"], config["Job"], config["Training"], config["Models"], )
        elif world_size > 0:
            if config["Job"]["parallel"] == "True":
                print("Running on", world_size, "GPUs")
                mp.spawn(training.train_regular,args=(world_size, config["Processing"]["data_path"], config["Processing"]["features"], config["Job"], config["Training"], config["Models"],), nprocs=world_size, join=True,)
            if config["Job"]["parallel"] == "False": training.train_regular("cuda", world_size, config["Processing"]["data_path"], config["Processing"]["features"], config["Job"], config["Training"], config["Models"],)

    ##Predicting from a trained model
    elif run_mode == "Predict":
        #print("Starting prediction from trained model")
        train_error = training.predict(dataset, config["Training"]["loss"], config["Job"])
        #print("Test Error: {:.5f}".format(train_error))

    ##Running n fold cross validation
    elif run_mode == "CV":
        world_size = torch.cuda.device_count()
        if world_size == 0:
            #print("Running on CPU - this will be slow")
            training.train_CV("cpu", world_size, config["Processing"]["data_path"], config["Job"], config["Training"], config["Models"],)
        elif world_size > 0:
            if config["Job"]["parallel"] == "True":
                #print("Running on", world_size, "GPUs")
                mp.spawn(training.train_CV, args=(world_size, config["Processing"]["data_path"], config["Job"], config["Training"], config["Models"], ), nprocs=world_size, join=True,)
            if config["Job"]["parallel"] == "False":
                #print("Running on one GPU")
                training.train_CV( "cuda", world_size, config["Processing"]["data_path"], config["Job"], config["Training"], config["Models"],)

    ##Hyperparameter optimization
    elif run_mode == "Hyperparameter":

        print("Starting hyperparameter optimization")
        print("running for " + str(config["Models"]["epochs"])+ " epochs"+ " on "+ str(config["Job"]["model"])+ " model")

        ##Reprocess here if not reprocessing between trials
        if config["Job"]["reprocess"] == "False":
            process_start_time = time.time()

            dataset = process.get_dataset(config["Processing"]["data_path"], config["Training"]["target_index"], config["Processing"]["features"], config["Job"]["reprocess"], config["Processing"],)

            #print("Dataset used:", dataset)
            #print(dataset[0])

            if config["Training"]["target_index"] == -1: config["Models"]["output_dim"] = len(dataset[0].y[0])
            # print(len(dataset[0].y))

        ##Set up search space for each model; these can subject to change
        hyper_args = {}
        dim1 = [x * 10 for x in range(5, 20)]
        dim2 = [x * 10 for x in range(5, 20)]
        dim3 = [x * 10 for x in range(5, 20)]
        hyper_args["SchNet"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "dim3": tune.choice(dim3),
            "gc_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]),  
            "cutoff": config["Processing"]["graph_max_radius"],
        }
            
        hyper_args["CGCNN"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "gc_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]),
        }
        hyper_args["MPNN"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "dim3": tune.choice(dim3),
            "gc_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]),
        }
        hyper_args["MEGNet"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "dim3": tune.choice(dim3),
            "gc_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]),
        }
        hyper_args["GCN"] = {
            "dim1": tune.choice(dim1),
            "dim2": tune.choice(dim2),
            "gc_count": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "post_fc_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "pool": tune.choice(["global_mean_pool", "global_add_pool", "global_max_pool", "set2set"]),
        }
        ##Run tune setup and trials
        best_trial = training.tune_setup(hyper_args[config["Job"]["model"]], config["Job"], config["Processing"], config["Training"],  config["Models"],)
        
        ##Write hyperparameters to file
        hyperparameters = best_trial.config["hyper_args"]
        hyperparameters = {k: round(v, 6) if isinstance(v, float) else v
            for k, v in hyperparameters.items()
        }
        with open(
            config["Job"]["job_name"] + "_optimized_hyperparameters.json", "w", encoding="utf-8",) as f: json.dump(hyperparameters, f, ensure_ascii=False, indent=4)

        ##Print best hyperparameters
        print("Best trial hyper_args: {}".format(hyperparameters))
        print("Best trial final validation error: {:.5f}".format(best_trial.last_result["loss"]))

    ##Ensemble mode using simple averages
    elif run_mode == "Ensemble":

       # print("Starting simple (average) ensemble training")
        #print("Ensemble list: ", config["Job"]["ensemble_list"])
        training.train_ensemble(config["Processing"]["data_path"], config["Job"], config["Training"], config["Models"],)

    else:
        print("No valid mode selected, try again")

run_mode = 1 #0: Training, #1: CV
mode = 0
#0: Train for n conditions
#1: Test

split = [0, 0, 1]

r2_test, r2_val, r2_train = 0, 999, 999
MAE_test, MAE_val, MAE_train = 0, 999, 999
RMSE_test, RMSE_val, RMSE_train = 0, 999, 999

if mode == 0:
    n = 25
    for i in range(n):
        #print("Iteration: ", i+1)
        s_time = time.time()
        if __name__ == "__main__":
            main()
            if run_mode == 0:
                df, df_train, df_valid  = pd.read_csv('Test_test_outputs.csv'), pd.read_csv('Test_train_outputs.csv'), pd.read_csv('Test_val_outputs.csv')
                df, df_train, df_valid = df.drop(['ids'], axis = 1), df_train.drop(['ids'], axis = 1), df_valid.drop(['ids'], axis = 1)
                df, df_train, df_valid = df.dropna(), df_train.dropna(), df_valid.dropna()
                y_test, y_train, y_valid  = df['target'], df_train['target'], df_valid['target']
                y_pred, y_pred_train, y_pred_valid = df['prediction'], df_train['prediction'], df_valid['prediction']

                if (i == 0 or (split[2]*mean_absolute_error(y_test, y_pred) + split[1]*mean_absolute_error(y_valid, y_pred_valid) + split[0]*mean_absolute_error(y_train, y_pred_train)) < (split[2]*MAE_test + split[1]*MAE_val + split[0]*MAE_train)):
                    r2_test, r2_val, r2_train = r2_score(y_test, y_pred), r2_score(y_valid, y_pred_valid), r2_score(y_train, y_pred_train)
                    MAE_test, MAE_val, MAE_train = mean_absolute_error(y_test, y_pred), mean_absolute_error(y_valid, y_pred_valid), mean_absolute_error(y_train, y_pred_train)
                    RMSE_test, RMSE_val, RMSE_train = math.sqrt(mean_squared_error(y_test, y_pred)), math.sqrt(mean_squared_error(y_valid, y_pred_valid)), math.sqrt(mean_squared_error(y_train, y_pred_train))
                    os.replace("my_model.pth", f"ModelSavePath_{test_mae}/my_model.pth")
                    os.replace("Test_train_outputs.csv", f"ModelSavePath/ModelSavePath_{train_mse}/Test_train_outputs.csv")
                    os.replace("Test_val_outputs.csv", f"ModelSavePath/ModelSavePath_{val_mse}/Test_val_outputs.csv")
                    os.replace("Test_test_outputs.csv", f"ModelSavePath/ModelSavePath_{test_mae}/Test_test_outputs.csv")

                    df = pd.read_csv('plot.csv')
                    plot = df.to_numpy()
                    epochs = range(1, len(plot[:,0]) + 1)
                    plt.clf()
                    plt.plot(epochs, plot[:, 0], 'r', label='Training Error')
                    plt.plot(epochs, plot[:, 1], 'b', label='Validation Error')
                    plt.title('Training and Validation Error', fontsize=18)
                    plt.xlabel('Epochs', fontsize=18)
                    plt.ylabel('Error', fontsize=18)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.legend(fontsize=18)
                    plt.savefig(f"ModelSavePath_{test_mae}/error.png")
                    os.replace("plot.csv", f"ModelSavePath/ModelSavePath_{test_mae}/plot.csv")
                    
            if run_mode == 1:
                df = pd.read_csv('Test_CV_outputs.csv')
                df = df.drop(['ids'], axis = 1)
                df = df.dropna()
                y_test = df['target']
                y_pred = df['prediction']

                if (i == 0 or split[2]*mean_absolute_error(y_test, y_pred) < split[2]*MAE_test):
                    r2_test = r2_score(y_test, y_pred)
                    MAE_test = mean_absolute_error(y_test, y_pred)
                    RMSE_test = math.sqrt(mean_squared_error(y_test, y_pred))
                    new_folder_path = f"ModelSavePath/ModelSavePath_{MAE_test}"
                    os.makedirs(new_folder_path, exist_ok=True)
                    shutil.move("my_model.pth", f"{new_folder_path}/my_model.pth")
                    shutil.move("Test_CV_outputs.csv", f"{new_folder_path}/Test_CV_outputs.csv")

                    df = pd.read_csv('plot.csv')
                    plot = df.to_numpy()
                    epochs = range(1, len(plot[:,0]) + 1)
                    plt.clf()
                    plt.plot(epochs, plot[:, 0], 'r', label='Training Error')
                    plt.plot(epochs, plot[:, 1], 'b', label='Validation Error')
                    plt.title('Training and Validation Error', fontsize=18)
                    plt.xlabel('Epochs', fontsize=18)
                    plt.ylabel('Error', fontsize=18)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.legend(fontsize=18)
                    plt.savefig("error.png")
                    shutil.move("plot.csv", f"{new_folder_path}/plot.csv")
                    shutil.move("error.png", f"{new_folder_path}/error.png")
                        
            if run_mode == 0:        
                print("+-------------------------------------------------+")
                print("|---------------------- R² ----- MAE ---- RMSE ---|")
                print("|-------------------------------------------------|")
                print('| Training Set :  ', ' | ', '{:.3f}'.format(round(r2_train, 3)), ' | ', '{:.3f}'.format(round(MAE_train, 3)), ' | ', '{:.3f}'.format(round(RMSE_train, 3)), ' |')
                print("|-------------------------------------------------|")
                print('| Validation Set :', ' | ', '{:.3f}'.format(round(r2_val, 3)), ' | ', '{:.3f}'.format(round(MAE_val, 3)), ' | ', '{:.3f}'.format(round(RMSE_val, 3)), ' |')
                print("|-------------------------------------------------|")
                print('| Test Set :      ', ' | ', '{:.3f}'.format(round(r2_test, 3)), ' | ', '{:.3f}'.format(round(MAE_test, 3)), ' | ', '{:.3f}'.format(round(RMSE_test, 3)), ' | ')
                print("+-------------------------------------------------+")
                print("--- %s total seconds elapsed ---" % (time.time() - s_time))
            if run_mode == 1:
                print("+-------------------------------------------------+")
                print("|---------------------- R² ----- MAE ---- RMSE ---|")
                print("|-------------------------------------------------|")
                print('| Test Set :      ', ' | ', '{:.3f}'.format(round(r2_test, 3)), ' | ', '{:.3f}'.format(round(MAE_test, 3)), ' | ', '{:.3f}'.format(round(RMSE_test, 3)), ' | ')
                print("+-------------------------------------------------+")
                print("--- %s total seconds elapsed ---" % (time.time() - s_time))
else:
    if __name__ == "__main__":
        s_time = time.time()
        main()
        print("--- %s total seconds elapsed ---" % (time.time() - s_time))
        df = pd.read_csv('Test_predicted_outputs.csv')
        df = df.drop(['ids'], axis = 1)
        df = df.dropna()
        df = df.dropna()
        y_test = df['target']
        y_pred = df['prediction']
    
        r2 = r2_score(y_test, y_pred)
        MAE = mean_absolute_error(y_test, y_pred)
        RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
        
        print("+-------------------------------------------------+")
        print("|---------------------- R² ----- MAE ---- RMSE ---|")
        print("|-------------------------------------------------|")
        print("|-------------------------------------------------|")
        print('| Test Set :      ', ' | ', '{:.3f}'.format(round(r2_test, 3)), ' | ', '{:.3f}'.format(round(MAE_test, 3)), ' | ', '{:.3f}'.format(round(RMSE_test, 3)), ' | ')
        print("+-------------------------------------------------+")