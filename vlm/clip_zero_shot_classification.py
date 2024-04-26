# Classification of MedMNIST data using pretrained CLIP image encoder and a projection head
# Zero-shot classification

import os
import random
import argparse
import numpy as np
from tqdm import tqdm

from medmnist import INFO
from utils import readable_timestamp, load_data_and_data_loaders

import torch
from transformers import CLIPProcessor, CLIPModel

parser = argparse.ArgumentParser()

## configuration/hyperparameters
timestamp = readable_timestamp()

parser.add_argument("--dataset", type=str, default="BLOCK")
parser.add_argument("--data_flag", type=str, default="octmnist")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--save", type=bool, help="save predictions", default=True)
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()

## Check if CUDA is available, set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current PyTorch device is set to", device)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

## load dataset
training_data, validation_data, test_data, training_loader, validation_loader, test_loader, _ = load_data_and_data_loaders(args.dataset,
                                                                                                   args.data_flag,
                                                                                                   args.batch_size)
label_choices = INFO[args.data_flag]["label"]
if args.data_flag == "pathmnist":
    label_choices = [f"a {label} tissue sample" for label in label_choices.values()]
elif args.data_flag == "chestmnist":
    label_choices = [f"an X-Ray image with {label} disease" for label in label_choices.values()]
elif args.data_flag == "octmnist":
    label_choices = [f"a Retinal OCT image with {label} disease" for label in label_choices.values()]
else:
    label_choices = list(label_choices.values())

## load CLIP model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

## zero shot classification
def zero_shot_classify(split, data_loader, save_dir, save: bool=args.save):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    predictions = []
    for batch in tqdm(data_loader):
        (x, _) = batch
        x = x.to(device)
        # print("Labels ", label_choices)
        try:
            inputs = processor(text=label_choices,
                               images=x,
                               return_tensors="pt",
                               do_rescale=False,
                               do_center_crop=False,
                               padding=True)

            for k, v in inputs.items():
                inputs[k] = v.to(device) # set each processed data to device

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
                predicted_label = probs.argmax(-1) # predicted label
                predictions.append(probs) # store the predicted labels
        except Exception as e:
            print("Unable to classify due to: ", e)

    # Concatenate predicted labels from all batches into a single tensor
    predicted_labels = torch.cat(predictions).cpu().numpy()
    if save:
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/{split}_{args.data_flag}.npy", predicted_labels)
    return predicted_labels

if __name__ == "__main__":
    data_list = [training_data, validation_data, test_data]
    loader_list = [training_loader, validation_loader, test_loader]
    for split, data, loader in zip(["train", "val", "test"], data_list, loader_list):
        save_dir = f"./vlm/results/zero-shot"
        preds = zero_shot_classify(split, loader, save_dir)