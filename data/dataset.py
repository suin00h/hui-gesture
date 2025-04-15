import os
import glob
import json
import itertools
import numpy as np
import pandas as pd
import pickle as pkl

from scipy.io import loadmat
from tqdm.auto import tqdm
from torch.utils.data import Dataset

DATA_ROOT_DIR = os.path.dirname(__file__)
DATA_RAW_DIR = os.path.join(DATA_ROOT_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_ROOT_DIR, "processed")

SIGN_LABELS = [chr(i + 65) for i in range(26)]

class MyoGym_Dataset(Dataset):
    def __init__(self):
        self.data_git = "https://github.com/lok63/HAR-for-fitness-data.git "
        self.data_name = "myo-gym"
        self.pkl_name = "myo-gym-data-dict.pkl"
        self.data_dict = self.get_processed_data()
    
    def __len__(self):
        return len(self.data_dict["label"])
    
    def __getitem__(self, idx):
        return dict(
            emg=self.data_dict["emg"][idx],
            imu_acc=self.data_dict["imu_acc"][idx],
            imu_gyro=self.data_dict["imu_gyro"][idx],
            label=self.data_dict["label"][idx]
        )
    
    def get_processed_data(self):
        if not self.is_processed():
            self.process_data()
        
        return self.load_processed_data()
    
    def is_processed(self):
        return os.path.isfile(self.processed_data_dir())
    
    def process_data(self):
        data = self.get_raw_data()
        
        raw_data = pd.DataFrame(data["raw_data"])
        raw_labels = pd.DataFrame(data["raw_data_labels"])
        raw_labels.rename(columns={0:"exercise", 1:"person"}, inplace=True)
        
        raw_ = pd.concat([raw_data, raw_labels], axis=1)
        exercises = raw_["exercise"].unique()
        persons = raw_["person"].unique()
        
        batch_list = []
        label_list = []
        WINDOW_SIZE = 200 
        STEP_SIZE = 50 # sampling_freq = 50 Hz
        
        for exercise, person in itertools.product(exercises, persons):
            query_df = (raw_["exercise"] == exercise) & (raw_["person"] == person)
            raw_target = raw_.loc[query_df].to_numpy()
            
            num_windows = (len(raw_target) - WINDOW_SIZE) // STEP_SIZE + 1
            remainder = (len(raw_target) - WINDOW_SIZE) % STEP_SIZE
            for i in range(num_windows):
                start_idx = i * STEP_SIZE
                batch_list.append(raw_target[start_idx:start_idx + WINDOW_SIZE])
                label_list.append(exercise)
            if remainder:
                batch_list.append(raw_target[-WINDOW_SIZE:])
                label_list.append(exercise)
        # got (T, F) * 310
        # each (T, F) has same label
        # sliding window will make them into (Num_windows, Window_size, Feature_num)
        
        batched_data = np.stack(batch_list, axis=0)
        batched_label = np.array(label_list)
        
        data_dict = dict(
            emg=batched_data[:, :, 1:9],
            imu_acc=batched_data[:, :, 10:13],
            imu_gyro=batched_data[:, :, 14:],
            label=batched_label
        )
        
        self.save_processed_data(data_dict)
    
    def get_raw_data(self):
        if not self.is_raw_downloaded():
            self.download_raw_data()
        
        data_mat_dir = os.path.join(
            self.raw_data_dir(),
            "data/MyoGym/MyoGym.mat"
        )
        
        return loadmat(data_mat_dir)

    def is_raw_downloaded(self):
        return os.path.isdir(self.raw_data_dir())
    
    def download_raw_data(self):
        os.system(
            "git clone "
            + self.data_git
            + self.raw_data_dir()
        )
    
    def raw_data_dir(self):
        return os.path.join(DATA_RAW_DIR, self.data_name)
    
    def processed_data_dir(self):
        return os.path.join(DATA_PROCESSED_DIR, self.pkl_name)
    
    def save_processed_data(self, data_dict):
        with open(self.processed_data_dir(), "wb") as data_pkl:
            pkl.dump(data_dict, data_pkl)
    
    def load_processed_data(self):
        with open(self.processed_data_dir(), "rb") as data_pkl:
            data_dict =  pkl.load(data_pkl)
        return data_dict

class Sign_Language_Dataset(Dataset):
    def __init__(self):
        self.sign_dict = self.get_sign_data()
    
    def __len__(self):
        return self.sign_dict["label"].shape[0]
    
    def __getitem__(self, idx):
        return dict(
            emg = self.sign_dict["emg"][idx],
            imu_acc = self.sign_dict["imu_acc"][idx],
            imu_gyro = self.sign_dict["imu_gyro"][idx],
            imu_ori = self.sign_dict["imu_ori"][idx],
            label = self.sign_dict["label"][idx]
        )
    
    def get_sign_data(self):
        processed_file = os.path.join(
            DATA_PROCESSED_DIR,
            "sign-data-dict.pkl"
        )
        if not self.is_processed(processed_file):
            self.process_sign_data(processed_file)
        
        with open(processed_file, "rb") as data_pkl:
            sign_data_dict = pkl.load(data_pkl)
        return sign_data_dict
    
    def get_raw_sign_data(self):
        if self.is_raw_sign_downloaded():
            print("Raw sign language dataset already exists!")
            return
        
        print("Fetching Italian sign language dataset ...")
        
        os.system("git clone https://github.com/airtlab/An-EMG-and-IMU"
            "-Dataset-for-the-Italian-Sign-Language-Alphabet.git " + 
            os.path.join(DATA_RAW_DIR, "sign-lang"))
    
    @staticmethod
    def is_raw_sign_downloaded():
        return os.path.isdir(
            os.path.join(DATA_RAW_DIR, "sign-lang")
        )
    
    @staticmethod
    def is_processed(filename):
        return os.path.isfile(filename)

    def process_sign_data(self, filename):
        self.get_raw_sign_data()
        
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

if __name__ == "__main__":
    dataset = MyoGym_Dataset()
    # dataset = Sign_Language_Dataset()
    
    sample = dataset[100:102]
    print(sample["emg"].shape)
    print(sample["imu_acc"].shape)
    print(sample["imu_gyro"].shape)
    print(sample["label"].shape)
    print(sample["label"])
    print(len(dataset))
