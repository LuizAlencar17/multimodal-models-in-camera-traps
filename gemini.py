import sys
import torch
import pandas as pd
import google.generativeai as genai

from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from preprocess import QuestionAnsweringDataset
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from flags import FLAGS
from utils import get_results_path, get_prompt, get_label_mapper, get_label_mapper_reverse, resize_image

GOOGLE_API_KEY = "AIzaSyBsXzWgDSE-naipvx7I79AeAnsGQlHMO2w"

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
label_mapper_reverse = get_label_mapper_reverse(task)
prompt = get_prompt(model_name, task)

device = torch.device("cuda")


def gemini_generate(path, prompt):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
    response = model.generate_content(
        [prompt, resize_image(path)],
        stream=True,
        safety_settings=safety_settings)
    response.resolve()
    pred = ''
    for part in response.parts:
        pred += part.text
    return pred


def evaluation(file_name_csv, proportion=-1):
    print(f"evaluation using: {file_name_csv}")
    pred, real, path = [], [], []
    dataset = QuestionAnsweringDataset(file_name_csv=file_name_csv, task=task,
                                       proportion=proportion)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size)
    pbar = tqdm(dataloader)
    counter = 0
    for batch in pbar:
        batch_images, batch_labels = batch
        for image, label in zip(batch_images, batch_labels):
            try:
                image_path = dataset.image_files[counter]
                label_prompt_response = gemini_generate(image_path, prompt)

                pred.append(label_prompt_response)
                real.append(str(int(label)))
                path.append(image_path)
                counter += 1
            except Exception as e:
                print(f"[ERROR] {e}")
    print(f"accuracy_score: {accuracy_score(real, pred)}")
    return pred, real, path


def test():
    pred, real, path = evaluation(test_filename)
    df = pd.DataFrame({"pred": pred, "real": real, "path": path})
    path_to_save = get_results_path(results_path, dataset_name,
                                    model_name, tags)
    df.to_csv(path_to_save, index=False)


print(f"using device: {device} {sys.argv}")
{"test": test}.get(mode)()
