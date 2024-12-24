import torch
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from random import randint
from torch.utils.data import Dataset
from utils import get_label_mapper, get_one_hot_label, resize_image
from flags import FLAGS

tags = FLAGS.tags.split(",")


class TextImageDataset(Dataset):
    def __init__(self, dtype="pt", task="behaviour", file_name_csv="/", shuffle=False, proportion=-1, seed=1):
        super().__init__()
        self.proportion = proportion
        self.shuffle = shuffle
        self.dtype = dtype
        self.task = task
        self.device = torch.device("cuda")
        self.image_dim = (500, 500)
        df = pd.read_csv(file_name_csv)
        # if self.proportion > 0:
        #     df = df.sample(frac=self.proportion/100,
        #                    random_state=seed).reindex()
        # df = df.sample(10, replace=False, random_state=seed).reindex()
        self.labels = [get_label_mapper(task)[i]
                       for i in df.category.values]
        if "seq" in tags:
            self.image_files = df.path_seq_saved.values
        else:
            self.image_files = df.path.values

    def __len__(self):
        return len(self.image_files)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        img = resize_image(self.image_files[ind], self.image_dim)
        return np.array(img), self.labels[ind]


class QuestionAnsweringDataset(TextImageDataset):
    def __init__(self, dtype="pt", task="behaviour", file_name_csv="/", shuffle=False,  proportion=-1):
        super().__init__(dtype=dtype, task=task, file_name_csv=file_name_csv,
                         shuffle=shuffle,  proportion=proportion)


class SimilarityDataset(TextImageDataset):
    def __init__(self, dtype="pt", task="behaviour", file_name_csv="/", shuffle=False,  proportion=-1):
        super().__init__(dtype=dtype, task=task, file_name_csv=file_name_csv,
                         shuffle=shuffle,  proportion=proportion)

    def __getitem__(self, ind):
        label = get_one_hot_label(self.labels[ind], self.task)
        label = np.array([label])
        img = resize_image(self.image_files[ind])
        return np.array(img), label
