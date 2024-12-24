import sys
import torch
import pandas as pd

from absl import flags
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from utils import get_model_and_processor, get_model_path, get_results_path, get_prompt, get_label_mapper, get_label_mapper_reverse
from preprocess import QuestionAnsweringDataset
from flags import FLAGS

tags = FLAGS.tags.split(",")
patience = FLAGS.patience
task = FLAGS.task
model_name = FLAGS.model_name
mode = FLAGS.mode
dataset_name = FLAGS.dataset_name
seed = FLAGS.seed
num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
learning_rate = FLAGS.learning_rate
train_filename = FLAGS.train_filename
val_filename = FLAGS.val_filename
test_filename = FLAGS.test_filename
checkpoint_path = FLAGS.checkpoint_path
results_path = FLAGS.results_path

label_mapper = get_label_mapper(task)
label_mapper_reverse = get_label_mapper_reverse(task)
prompt = get_prompt(model_name, task)
model, processor = get_model_and_processor(model_name)

device = torch.device("cuda")
model.to(device)


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
            inputs = processor(images=image, text=prompt,
                               return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=1)
            label_prompt_response = processor.decode(
                outputs[0], skip_special_tokens=True)
            pred.append(label_prompt_response)
            real.append(str(int(label)))
            path.append(dataset.image_files[counter])
            counter += 1
    print(f"accuracy_score: {accuracy_score(real, pred)}")
    return pred, real, path


def train():
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    dataset = QuestionAnsweringDataset(
        file_name_csv=train_filename, task=task, shuffle=True, proportion=-1)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    last_acc, current_patience = -1, 0
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(dataloader)

        # running batchs of the data
        for batch in pbar:
            optimizer.zero_grad()
            batch_images, batch_labels = batch
            loss = 0
            for image, label in zip(batch_images, batch_labels):
                inputs = processor(text=prompt, images=image,
                                   return_tensors="pt")
                labels = processor(
                    text=str(int(label)), return_tensors="pt").input_ids
                inputs["labels"] = labels
                outputs = model(**(inputs).to(device))
                loss += outputs.loss
            loss.backward()
            optimizer.step()
            pbar.set_description(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

        # running validation subset
        pred, real, path = evaluation(val_filename, 10)
        acc = accuracy_score(real, pred)
        if acc > last_acc:
            current_patience = 0
            print(
                f"last accuracy: {last_acc} current accuracy: {acc}. saving model..")
            last_acc = acc
            torch.save(model.state_dict(), get_model_path(
                checkpoint_path, dataset_name, model_name))
        else:
            current_patience += 1
            if current_patience > patience:
                print("no patience, quitting train")
                break


def test():
    if "pretrained" in tags:
        model.load_state_dict(torch.load(get_model_path(
            checkpoint_path, dataset_name, model_name)))
    pred, real, path = evaluation(test_filename)
    df = pd.DataFrame({"pred": pred, "real": real, "path": path})
    path_to_save = get_results_path(results_path, dataset_name,
                                    model_name, tags)
    df.to_csv(path_to_save, index=False)


print(f"using device: {device} {sys.argv}")
{"train": train, "test": test}.get(mode)()
