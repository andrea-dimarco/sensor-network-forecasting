import os 
import numpy as np

#extract confusion matrix 

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

def getAverageStatistics(model_type):
    data_dir = "img/"
    experiments = os.listdir(data_dir)
    precision_result= np.empty(5)
    recall_result= np.empty(5)
    f1_result= np.empty(5)
    
    counter = 0
    for experiment in experiments:
        full_exp_path = os.path.join(data_dir,experiment)
        if os.path.isdir(full_exp_path) and experiment.startswith("exp_" + model_type) :
            counter +=1
            # get log
            log_path = os.path.join(full_exp_path,"log.txt")
            precision_result += getPrecision(log_path)
            recall_result += getRecall(log_path)
            f1_result += getF1(log_path)

            # open file 
            # do stuff
            # ???
            # profit
    print("Average Precision: ", precision_result / counter)
    print("Average Recall: ", recall_result / counter)
    print("Average F1: ", f1_result / counter)


def getPrecision(file):
    return getStatistic(file,"Precision")

def getF1(file):
    return getStatistic(file,"F1")

def getRecall(file):
    return getStatistic(file,"Recall")

#extract time
def getTrainingTime(file):
    return 

if __name__ == '__main__':
    print("Model stats:")
    getAverageStatistics("SSD")
    print("\nAdversary stats:")

    #print(experiments)