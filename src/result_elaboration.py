import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hyperparameters import Config

#extract confusion matrix 
def getAverageConfusionMatrix(modelType):
    paths = getExperimentPaths(modelType)
    cms = [[0 for _ in range(5)] for _ in range(5)]
    cm_list = []
    cms = np.matrix(cms)
    for p in paths:
        log = getExperimentLog(p)
        cm = getConfusionMatrix(log)
        cm = np.matrix(cm)
        cm_list.append(cm)
        cms = cms + cm
        #print(cm)
    cms = cms / len(paths)
    cms = np.round(cms,2)
    cm_std = np.round(np.std(cm_list,axis=0),2)
    print(cm_std)
    plt.clf()
    plt.close()
    labels = [i for i in range(-Config.discretization,Config.discretization+1)]
    norm = plt.Normalize(0,100)

    sns.heatmap(cms, 
                annot=True,
                fmt='g', 
                xticklabels=labels,
                yticklabels=labels, 
                norm=norm)
    for i in range(5):
        for j in range(5):                             
            plt.text(j+0.5,i+0.8,f"± {cm_std[i][j]}", ha="center",va="center", color="white", fontsize=10)
            if i==2 and j ==2:
                plt.text(j+0.5,i+0.8,f"± {cm_std[i][j]}", ha="center",va="center", fontsize=10)

    plt.savefig(f"img/{modelType}_average_confusion_matrix.png",dpi=300)

    plt.show()
    plt.clf()
    plt.close()


def getConfusionMatrix(file):
    log = open(file,"r")
    string_matrix = ""
    start_copying = False
    for line in log.readlines():
        if line.startswith("Precision"): start_copying = False  
        elif start_copying:
            string_matrix += line
        elif line.startswith("Confusion matrix"): 
            start_copying = True
    cm = []
    string_matrix = string_matrix.replace("\n","").strip("[]").split("] [")
    for el in string_matrix:
        new_el = el.split()
        cm.append([float(x) * 100 for x in new_el])
    return cm
def getStatistic(file,statistic):
    if statistic not in ["Precision", "F1", "Recall"]: 
        print("Nope")
        return
    log = open(file,"r")
    result = []
    for line in log.readlines():
        if line.startswith(statistic):
            statistic_array =  list(map(float,line
                                        .split(": ")[1]
                                        .replace("[","")
                                        .replace("]","")
                                        .split()))
            result = np.array(statistic_array)
    log.close()
    return result 

def getAverageStatistics(modelType):
    precision_result= np.zeros(5)
    recall_result= np.zeros(5)
    f1_result= np.zeros(5)
    
    counter = 0
    for experimentPath in getExperimentPaths(modelType):
        # get log
        logPath = getExperimentLog(experimentPath)
        counter +=1
        precision_result += getPrecision(logPath)
        recall_result += getRecall(logPath)
        #print("F1 Result start", f1_result)
        f1_result += getF1(logPath)
        #print("f1",getF1(logPath))
        #print("sum",f1_result)

    avg_precision =  precision_result / counter
    avg_recall = recall_result / counter
    avg_f1  = f1_result / counter
    #print("avg",avg_f1)
    return (avg_precision,avg_recall,avg_f1)

def getPrecision(file):
    return getStatistic(file,"Precision")

def getF1(file):
    return getStatistic(file,"F1")

def getRecall(file):
    return getStatistic(file,"Recall")

#extract time
def getTrainingTime(file):
    log = open(file,"r")
    result = []
    for line in log.readlines():
        if line.startswith("Training"):
            result = float(line.split()[2])
    log.close()
    return result 

def getExperimentPaths(modelType=""):
    data_dir = "img/"
    experiments = os.listdir(data_dir)
    result = []
    for experiment in experiments:
        full_exp_path = os.path.join(data_dir,experiment)
        if os.path.isdir(full_exp_path) and experiment.startswith("exp_" + modelType) :
            result.append(full_exp_path)
    return result

def getExperimentLog(path):
    return os.path.join(path,"log.txt")

def getAverageTime(isADV:bool=False):
    modelType = "ADV" if isADV else "SSD"
    paths = getExperimentPaths(modelType)
    #print(paths)
    result = 0
    counter = 0
    for i in range(len(paths)): 
        fp = getExperimentLog(paths[i])
        result += getTrainingTime(fp)
        counter +=1
    return result / counter

if __name__ == '__main__':
    #getAverageConfusionMatrix("ADV")
    # print("Average time elapsed: ",getAverageTime(isADV=True))
    paths = getExperimentPaths("ADV")
    best = 0
    best_sensor = ""
    for p in paths:
        log = getExperimentLog(p)
        cm = getConfusionMatrix(log)
        cm = np.matrix(cm)
        trace = np.trace(cm)
        if best < trace: 
            best = trace
            best_sensor = p

    print(best_sensor,best)

    worst = 1000000
    worst_sensor = ""
    for p in paths:
        log = getExperimentLog(p)
        cm = getConfusionMatrix(log)
        cm = np.matrix(cm)
        trace = np.trace(cm)
        if worst > trace: 
            worst = trace
            worst_sensor = p

    print(worst_sensor,worst)

    getAverageStatistics("SSD")
    # print("\nModel stats:")    
    # model_precision,model_recall,model_f1 = getAverageStatistics("SSD")
    # print("\tPrecision:\n\t", model_precision)
    # print("\tRecall:\n\t", model_recall)
    # print("\tF1:\n\t", model_f1)

    # print("\nAdversary stats:")
    # adv_precision,adv_recall,adv_f1= getAverageStatistics("ADV")
    # print("\tPrecision:\n\t", adv_precision)
    # print("\tRecall:\n\t", adv_recall)
    # print("\tF1:\n\t", adv_f1)

    #print(experiments)