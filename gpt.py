import sys
import torch
import requests
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from utils import read_img_in_base64, get_results_path, get_prompt, get_label_mapper, get_label_mapper_reverse
from preprocess import QuestionAnsweringDataset
from flags import FLAGS

OPENAI_API_KEY = '<KEY>'

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
label_mapper_reverse = get_label_mapper_reverse(task)
prompt = get_prompt(model_name, task)

device = torch.device("cuda")


def gpt4_generate(base64_image, prompt):
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
        "max_tokens": 1
    })
    response = raw_response.json()
    action = response['choices'][0]['message']['content'].lower()
    return action


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
                label_prompt_response = gpt4_generate(
                    read_img_in_base64(image_path), prompt)
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
