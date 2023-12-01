import os
# import urllib.request as request
# import zipfile
# import shutil
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
import opendatasets as od
from cnnClassifier.entity.config_entity import DataIngestionConfig



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename = self.config.local_data_file
            # opendataset requires kaggle.json file which has username and key to download the file
            # it can be obtained through kaggle api and then put the downloaded file in root directory of project 
            od.download_kaggle_dataset(dataset_url=self.config.source_URL, data_dir=self.config.local_data_file)
            # od.download(dataset_id_or_url=self.config.source_URL, data_dir=self.config.local_data_file)
            logger.info(f"{filename} downloaded!")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  


    
    # def extract_zip_file(self):
    #     """
    #     zip_file_path: str
    #     Extracts the zip file into the data directory
    #     Function returns None
    #     """
    #     unzip_path = self.config.unzip_dir
    #     os.makedirs(unzip_path, exist_ok=True)
    #     shutil.unpack_archive(self.config.local_data_file, unzip_path)