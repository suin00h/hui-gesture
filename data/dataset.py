import os
import glob
import json
import numpy as np
import pickle as pkl

from tqdm.auto import tqdm
from torch.utils.data import Dataset

DATA_ROOT_DIR = os.path.dirname(__file__)
DATA_RAW_DIR = os.path.join(DATA_ROOT_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_ROOT_DIR, "processed")

SIGN_LABELS = [chr(i + 65) for i in range(26)]

def is_raw__sign_downloaded():
    return os.path.isdir(
        os.path.join(DATA_RAW_DIR, "sign-lang")
    )

def get_raw_sign_data():
    if is_raw__sign_downloaded():
        print("Raw sign language dataset already exists!")
        return
    
    print("Fetching Italian sign language dataset ...")
    
    os.system("git clone https://github.com/airtlab/An-EMG-and-IMU"
        "-Dataset-for-the-Italian-Sign-Language-Alphabet.git " + 
        os.path.join(DATA_RAW_DIR, "sign-lang"))

def is_processed(filename):
    return os.path.isfile(filename)

def process_sign_data(filename):
    get_raw_sign_data()
    
    sign_emg_list = []
    sign_imu_acc_list = []
    sign_imu_gyro_list = []
    sign_imu_ori_list = []
    sign_label_list = []
    
    tqdm.write("Processing sign language dataset ...")
    for idx, label in tqdm(enumerate(SIGN_LABELS)):
        label_dir = os.path.join(
            DATA_RAW_DIR, 
            "sign-lang/Dataset", 
            label
        )
        for json_file in glob.glob(label_dir + "/*.json"):
            with open(json_file) as json_data:
                sign_data = json.load(json_data)
            sign_emg_list.append(sign_data["emg"]["data"])
            sign_imu_acc_list.append([imu["acceleration"] for imu in sign_data["imu"]["data"]])
            sign_imu_gyro_list.append([imu["gyroscope"] for imu in sign_data["imu"]["data"]])
            sign_imu_ori_list.append([imu["orientation"] for imu in sign_data["imu"]["data"]])
            sign_label_list.append(idx)
    
    sign_data_dict = {
        "emg": np.array(sign_emg_list),
        "imu_acc": np.array(sign_imu_acc_list),
        "imu_gyro": np.array(sign_imu_gyro_list),
        "imu_ori": np.array(sign_imu_ori_list),
        "label": np.array(sign_label_list)
    }
    
    with open(filename, "wb") as data_pkl:
        pkl.dump(sign_data_dict, data_pkl)
        print("Successfully processed sign data.")

def get_sign_data():
    processed_file = os.path.join(
        DATA_PROCESSED_DIR, 
        "sign-data-dict.pkl"
    )
    if not is_processed(processed_file):
        process_sign_data(processed_file)
    
    with open(processed_file, "rb") as data_pkl:
        sign_data_dict = pkl.load(data_pkl)
    return sign_data_dict

class Sign_Language_Dataset(Dataset):
    def __init__(self):
        sign_dict = get_sign_data() # Download and process the dataset
    
    def __len__(self):
        ...
    
    def __getitem__(self, idx):
        ...

if __name__ == "__main__":
    import os
    
    sdict = get_sign_data()
    print([value.shape for value in sdict.values()])
