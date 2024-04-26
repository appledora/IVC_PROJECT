import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from medmnist import INFO

import torch
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_auroc
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPProcessor, CLIPModel
from utils import load_data_and_data_loaders, readable_timestamp

parser = argparse.ArgumentParser()

## configuration/hyperparameters
timestamp = readable_timestamp()

parser.add_argument("--dataset", type=str, default="BLOCK")
parser.add_argument("--data_flag", type=str, default="pathmnist")
parser.add_argument("--save_interval", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()

## Check if CUDA is available, set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("Current PyTorch device is set to", device)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

## load dataset
training_data, validation_data, test_data, training_loader, validation_loader, test_loader, _ = load_data_and_data_loaders(args.dataset,
                                                                                                   args.data_flag,
                                                                                                   args.batch_size)
label_dict = INFO[args.data_flag]["label"]
if args.data_flag == "pathmnist":
    label_choices = [f"a {label} tissue sample" for label in label_dict.values()]
elif args.data_flag == "chestmnist":
    label_choices = [f"an X-Ray image with {label} disease" for label in label_dict.values()]
elif args.data_flag == "octmnist":
    label_choices = [f"a Retinal OCT image with {label} disease" for label in label_dict.values()]
else:
    label_choices = list(label_dict.values())


## load CLIP model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

## Fine-tune model
def finetune(save_dir: str):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    writer = SummaryWriter()
    weights_path = Path(f"./vlm/{args.filename}_model_checkpoints")
    weights_path.mkdir(exist_ok=True)
    
    ## Loss function
    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()

    ## Fine-tuning layers
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-7, weight_decay=0.0001)

    num_batches_train = len(training_loader.dataset)/args.batch_size
    num_batches_val = len(validation_loader.dataset)/args.batch_size

    ## Training
    for epoch in range(args.num_epochs):
        print(f"Epoch: {epoch}")
        epoch_train_loss = 0
        model.train()
        for batch in tqdm(training_loader, total=num_batches_train):
            optimizer.zero_grad()
            
            ## Format input data
            (x, labels) = batch
            x = x.to(device)
            text_labels = [f"{label_dict[str(label.cpu().numpy()[0])]}" for label in labels]
            try:
                inputs = processor(text=text_labels, images=x, return_tensors="pt",
                                do_rescale=False,
                                do_center_crop=False,
                                padding=True)
                for k, v in inputs.items():
                    inputs[k] = v.to(device) # set each processed data to device

                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image # this is the image-text similarity score
                logits_per_text = outputs.logits_per_text # this is the image-text similarity score

                ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=device)

                total_train_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                total_train_loss.backward()
                epoch_train_loss += total_train_loss

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                if device == "cpu":
                    optimizer.step()
                else:
                    optimizer.step()
            except Exception as e:
                print("Unable to train due to: ", e)

        epoch_train_loss /= num_batches_train
        writer.add_scalar("Loss/train", epoch_train_loss, epoch)

        if epoch % args.save_interval == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_train_loss,
                }, weights_path / f"model_{epoch}.pt")
            print(f"Saved weights under model_checkpoint/model_{epoch}.pt")

        ## Compute val accuracy
        model.eval()
        acc_top1_list = []
        acc_top3_list = []
        auc_list=[]
        similarities = []

        epoch_val_loss = 0
        for _, batch in enumerate(tqdm(validation_loader, total=num_batches_val)):
            (val_x, val_labels) = batch
            val_x = val_x.to(device)
            val_labels = val_labels.to(device)
            text_labels = label_choices

            inputs = processor(text=text_labels, images=val_x, return_tensors="pt",
                               do_rescale=False,
                               do_center_crop=False,
                               padding=True)
                               
            for k, v in inputs.items():
                inputs[k] = v.to(device) # set each processed data to device

            with torch.no_grad():
                outputs = model(**inputs)

                total_loss = (loss_img(outputs.logits_per_image, val_labels.squeeze()))
                epoch_val_loss += total_loss

            image_features, text_features = outputs.image_embeds, outputs.text_embeds
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            acc_top1 = multiclass_accuracy(similarity, val_labels.squeeze(), num_classes=len(label_choices))
            acc_top3 = multiclass_accuracy(similarity, val_labels.squeeze(), top_k=3, num_classes=len(label_choices))
            auc = multiclass_auroc(similarity, val_labels.squeeze(), num_classes=len(label_choices))
            acc_top1_list.append(acc_top1)
            acc_top3_list.append(acc_top3)
            auc_list.append(auc)
            similarities.append(similarity)

        writer.add_scalar("Loss/val", epoch_val_loss / num_batches_val, epoch)
        print(f"Epoch {epoch} train loss: {epoch_train_loss / num_batches_train}")
        print(f"Epoch {epoch} val loss: {epoch_val_loss / num_batches_val}")

        # compute mean top3 accuracy and top1 accuracy
        mean_top3_accuracy = torch.stack(acc_top3_list).mean().cpu().numpy()
        print(f"Mean Top 3 Accuracy: {mean_top3_accuracy*100}%.")
        writer.add_scalar("Val Accuracy/top3", mean_top3_accuracy , epoch)
        mean_top1_accuracy = torch.stack(acc_top1_list).mean().cpu().numpy()
        print(f"Mean Top 1 Accuracy: {mean_top1_accuracy*100}%.")
        writer.add_scalar("Val Accuracy/Top1", mean_top1_accuracy, epoch)
        mean_auc = torch.stack(auc_list).mean().cpu().numpy()
        print(f"Mean Current AUC: {mean_auc*100}%")
        
        predicted_similarities = torch.cat(similarities).cpu().numpy()
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/val_preds.npy", predicted_similarities)

    writer.flush()
    writer.close()

if __name__ == '__main__':
    finetune(save_dir="./vlm/results/finetuning")