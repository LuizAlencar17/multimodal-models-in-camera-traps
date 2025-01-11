import json
import sys
import time

import google.generativeai as genai
import pandas as pd
import torch
from flags import FLAGS
from google.api_core.exceptions import ResourceExhausted
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from preprocess import QuestionAnsweringDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import (get_label_mapper, get_prompt, get_results_path,
                   remove_non_numeric, resize_image)

GOOGLE_API_KEYS = {
    "idx": 0,
    "keys": json.load(open("./secrets.json"))["gemini"]
}

safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
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


def gemini_generate(path, prompt):
    """Generate content using Gemini API."""
    if GOOGLE_API_KEYS["idx"] >= len(GOOGLE_API_KEYS["keys"]):
        GOOGLE_API_KEYS["idx"] = 0
    try:
        genai.configure(
            api_key=GOOGLE_API_KEYS["keys"][GOOGLE_API_KEYS["idx"]])
        model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
        response = model.generate_content(
            [prompt, resize_image(path)],
            stream=True,
            safety_settings=safety_settings)
        response.resolve()
        pred = ''
        for part in response.parts:
            pred += part.text
        pred = remove_non_numeric(pred)
        return label_mapper.get(pred)
    except ResourceExhausted:
        seconds = 10
        GOOGLE_API_KEYS['idx'] += 1
        print(
            f"[ResourceExhausted] updating key index: {GOOGLE_API_KEYS['idx']} waiting: {seconds} secounds")
        time.sleep(seconds)
        return gemini_generate(path, prompt)
    except Exception as e:
        GOOGLE_API_KEYS['idx'] += 1
        print(f"[ERROR]: {e}")
        return -1


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
                label_prompt_response = gemini_generate(image_path, prompt)

                pred.append(label_prompt_response)
                real.append(int(label))
                path.append(image_path)

                # tmp: start
                path_to_save = get_results_path(
                    results_path, dataset_name, model_name, tags)
                df = pd.DataFrame({"pred": pred, "real": real, "path": path})
                df.to_csv(path_to_save, index=False)
                # end: start
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
