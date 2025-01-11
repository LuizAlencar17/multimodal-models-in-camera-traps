from random import randint

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from flags import FLAGS
from PIL import Image
from torchvision import transforms
from utils import get_label_mapper, get_one_hot_label, resize_image

tags = FLAGS.tags.split(",")


class TextImageDataset:
    def __init__(self, dtype="pt", task="behaviour", file_name_csv="/", shuffle=False, proportion=-1, batch_size=10, seed=1):
        self.proportion = proportion
        self.shuffle = shuffle
        self.dtype = dtype
        self.task = task
        self.batch_size = batch_size
        self.device = torch.device("cuda")
        self.transform = transforms.ToTensor()
        self.image_dim = (500, 500)
        df = pd.read_csv(file_name_csv)
        self.labels = [get_label_mapper(task)[i] for i in df.category.values]
        self.labels_in_tensor = [torch.tensor(i) for i in self.labels]
        self.image_files = df.path_seq_saved.values if "seq" in tags else df.path.values

    def __len__(self):
        return len(self.image_files)

    def random_sample(self):
        return self.__getitem__(randint(0, len(self) - 1))

    def sequential_sample(self, ind):
        return self.__getitem__((ind + 1) % len(self))

    def skip_sample(self, ind):
        return self.random_sample() if self.shuffle else self.sequential_sample(ind)

    def __getitem__(self, ind):
        img = resize_image(self.image_files[ind], self.image_dim)
        return img, self.labels_in_tensor[ind]

    def get_total_batches(self):
        return len(self.labels) // self.batch_size + (1 if len(self.labels) % self.batch_size > 0 else 0)

    def get_batches(self):
        indices = list(range(len(self)))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_images, batch_labels = zip(
                *[self.__getitem__(idx) for idx in batch_indices])
            yield batch_images, batch_labels


class QuestionAnsweringDataset(TextImageDataset):
    pass


class SimilarityDataset(TextImageDataset):
    def __getitem__(self, ind):
        label = get_one_hot_label(self.labels[ind], self.task)
        img = resize_image(self.image_files[ind])
        return img, label
