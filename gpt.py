import json
import sys

import pandas as pd
import requests
import torch
from flags import FLAGS
from preprocess import QuestionAnsweringDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import (get_label_mapper, get_prompt, get_results_path,
                   read_img_in_base64, remove_non_numeric)

OPENAI_API_KEY = json.load(open("./secrets.json"))["gpt"][0]

OPENAI_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}


tags = FLAGS.tags.split(",")
task = FLAGS.task
model_name = FLAGS.model_name
mode = FLAGS.mode
dataset_name = FLAGS.dataset_name
batch_size = FLAGS.batch_size
test_filename = FLAGS.test_filename
checkpoint_path = FLAGS.checkpoint_path
results_path = FLAGS.results_path

label_mapper = get_label_mapper(task)
prompt = get_prompt(model_name, task)

device = torch.device("cuda")


def gpt4_generate(base64_image, prompt):
    """Generate content using GPT-4 API."""
    raw_response = requests.post("https://api.openai.com/v1/chat/completions", headers=OPENAI_HEADERS, json={
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an assistant that can identify animal behavior in images without learning from past data."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2
    })
    response = raw_response.json()
    pred = response['choices'][0]['message']['content'].lower()
    pred = remove_non_numeric(pred)
    return label_mapper.get(pred)


def evaluation(file_name_csv, proportion=-1):
    """Evaluate the dataset without using DataLoader."""
    print(f"Evaluation using: {file_name_csv}")
    pred, real, path = [], [], []
    dataset = QuestionAnsweringDataset(
        file_name_csv=file_name_csv, task=task, proportion=proportion, batch_size=batch_size)

    counter = 0
    for batch_images, batch_labels in tqdm(dataset.get_batches(),
                                           total=dataset.get_total_batches()):
        for image, label in zip(batch_images, batch_labels):
            try:
                image_path = dataset.image_files[counter]
                label_prompt_response = gpt4_generate(
                    read_img_in_base64(image_path), prompt)
                pred.append(label_prompt_response)
                real.append(int(label))
                path.append(image_path)
                counter += 1
            except Exception as e:
                print(f"[ERROR] {e}")

    print(f"Accuracy Score: {accuracy_score(real, pred)}")
    return pred, real, path


def test():
    """Run evaluation on the test dataset."""
    pred, real, path = evaluation(test_filename)
    df = pd.DataFrame({"pred": pred, "real": real, "path": path})
    path_to_save = get_results_path(
        results_path, dataset_name, model_name, tags)
    df.to_csv(path_to_save, index=False)


print(f"Using device: {device} {sys.argv}")
{"test": test}.get(mode)()
