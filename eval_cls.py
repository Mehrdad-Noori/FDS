import argparse
import os
import csv
import tqdm
import torch
import pickle
import random
import numpy as np

from domainbed.datasets import transforms as DBT
from domainbed.datasets.datasets import GeneralDGEvalDataset

from domainbed import algorithms
from domainbed import hparams_registry

from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import entropy




### Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="DomainBed Evaluation Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory for the dataset")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--algorithm", type=str, default='ERM', choices=['ERM', 'OtherAlgorithm'], help="Algorithm to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--ckpt_dir", type=str, default=64, help="Path to a trained model")

    return parser.parse_args()


args = parse_args()

# Hyperparameters
hparams = {"data_augmentation": False,
           "resnet18": False, 
           "resnet_dropout": 0.0, 
           }



print("===="*10)
print(f"data dir: {args.data_dir}")
print(f"checkpoint_dir: {args.ckpt_dir}")

save_dir = args.save_dir

### Create save root
os.makedirs(save_dir, exist_ok=True)

### Data
dataset = GeneralDGEvalDataset(args.data_dir, -1 , hparams)

print(f"Num datasets: {len(dataset)}")
# dataset = iter(dataset)

combined_dataset = ConcatDataset(dataset)
class_to_idx = combined_dataset.datasets[0].class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}


### loading the model
default_hparams = hparams_registry.default_hparams(args.algorithm, dataset)
default_hparams.update(hparams)
hparams = default_hparams


algorithm_class = algorithms.get_algorithm_class(args.algorithm)
model = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset), 
                        hparams)
checkpoint = torch.load(args.ckpt_dir, map_location='cpu')
state_dict = checkpoint['model_dict']
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
msg = model.load_state_dict(new_state_dict, strict=False)
print("+++ msg", msg)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()


# Create a DataLoader
data_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Initialize accuracy tracking variables
total = 0
correct = 0

# Iterate over batches
with torch.no_grad():
    for batch in tqdm.tqdm(data_loader):
        x, y, paths = batch
        x = x.to(device)
        y = y.to(device)

        # Batch prediction
        outputs, embeddings = model.predict_emb(x)
        _, predicted = torch.max(outputs.data, 1)

        

        # Update correct and total counts for accuracy
        total += y.size(0)
        correct += (predicted == y).sum().item()



        # Process each item in the batch
        for idx, (path, pred, logit, label, emb) in enumerate(zip(paths, predicted, outputs.data.cpu(), y, embeddings)):
            correct_prediction = "Yes" if pred == label else "No"
            
            # Calculate entropy
            prob = torch.nn.functional.softmax(logit, dim=0).numpy()
            prediction_entropy = entropy(prob)

            # Save image information to CSV
            with open(os.path.join(save_dir, "image_predictions.csv"), 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([path, label.item(), pred.item(), correct_prediction, prediction_entropy, logit.tolist()])

            

# Calculate and print final accuracy
accuracy = 100 * correct / total
print(f'Final Accuracy: {accuracy}%')


# Save the summary information to a TXT file
summary_file = os.path.join(save_dir, "summary_info.txt")
with open(summary_file, 'w') as file:
    file.write(f'checkpoint_dir: {args.ckpt_dir}\n')
    file.write(f'Total images: {total}\n')
    file.write(f'Accuracy: {accuracy}%\n')

print(f"Saved summary information to {summary_file}")

