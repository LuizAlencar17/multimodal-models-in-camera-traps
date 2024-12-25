import torch
import numpy as np
from tensorflow.keras.utils import to_categorical
from transformers import AutoProcessor, BlipForQuestionAnswering, CLIPProcessor, CLIPModel


import base64
import numpy as np
from PIL import Image
from io import BytesIO


def resize_image(input_path, image_dim=(500, 500)):
    return Image.open(input_path)
    with Image.open(input_path) as img:
        return img.resize(image_dim, Image.Resampling.LANCZOS)


def read_img_in_base64(image_path):
    img = resize_image(image_path)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    buffered.seek(0)
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_string


def get_model_and_processor(model_name: str):
    if model_name == "blip":
        return BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base"), AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    if model_name == "clip":
        return CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_parsed_model_name(model_name):
    return model_name.replace('/', '-')


def get_model_path(checkpoint_path, dataset_name, model_name):
    return f"{checkpoint_path}{get_parsed_model_name(model_name)}_{dataset_name}.pth"


def get_label_by_logits(logits):
    # probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    probs = logits.softmax(dim=1).cpu().detach().numpy()[0]
    return np.argmax(probs)


def get_real_by_label(label):
    return int(torch.argmax(label, dim=0))


def get_results_path(results_path, dataset_name, model_name, tags):
    return f"{results_path}{get_parsed_model_name(model_name)}_{dataset_name}_{'-'.join(tags)}.csv"


def get_prompt(model_name: str, task: str):
    mapper = {
        "behaviour": {
            "gemini": "identify the behavior the animals are performing. Respond only with one of the following options: 0) moving, 1) eating and 2) resting",
            "gpt": "identify the behavior the animals are performing. Respond only with one of the following options: 0) moving, 1) eating and 2) resting",
            "blip": "identify the behavior the animals are performing. Respond only with one of the following options: 0) moving, 1) eating and 2) resting",
            "clip": ["a photo of an animal moving", "a photo of an animal eating", "a photo of an animal resting"]
        },
        "animal": {
            "gemini": "a photo of an animal: 0) no and 1) yes",
            "gpt": "a photo of an animal: 0) no and 1) yes",
            "blip": "a photo of an animal: 0) no and 1) yes",
            "clip": ["a photo of a background", "a photo of an animal"]
        }
    }
    return mapper[task][model_name]


def get_label_mapper(task: str):
    mapper = {
        "behaviour": {"moving": 0, "eating": 1, "resting": 2},
        "animal": {"no": 0, "yes": 1}
    }
    return mapper[task]


def get_label_mapper_reverse(task: str):
    label_mapper = get_label_mapper(task)
    return {value: key for key, value in label_mapper.items()}


def get_one_hot_label(category, task):
    mapper = {
        "behaviour": 3,
        "animal": 2
    }
    return to_categorical(category, mapper[task])
