import pandas as pd 
import dataset_handling as dh
from hyperparameters import Config
import adversary as adv
import ssd_training as ssd
import utilities as ut
if __name__ == '__main__':

    ut.set_seed(Config.seed)

    # read sensor list from dataset 
    df = pd.read_csv("./datasets/sensor_data_cleaned.csv")
    sensors = df.columns
    half_index= len(df.columns) // 2
    sensor_list = sensors[47:half_index]
    batch_counter = 1
    for sensor in sensor_list:
        
        print(f"Executing sensor {sensor} [{batch_counter}/{len(sensor_list)}]")

        # create dataset 
        print("\tBuilding dataset")
        dh.select_sensors_diff_3([sensor])
        
        # execute adversary training and validation
        print("\tExecuting adversary training")
        adv.execute_training(sensor)

        # execute model training and validation
        print("\tExecuting model training")
        ssd.execute_training(sensor)

        batch_counter+=1