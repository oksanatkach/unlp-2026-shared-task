import os, json
import torch


def clear_dir(dir_path):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)


def read_dir_logits(dir_path, top_k):
    all_logits = []

    for seq_idx in range(top_k):
        file_path = os.path.join(dir_path, f"{seq_idx}.json")
        captured = json.load(open(file_path))
        all_logits.append(captured)

    return torch.tensor(all_logits)
