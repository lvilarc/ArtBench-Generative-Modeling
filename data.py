from __future__ import annotations

import sys
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# Relative paths (run this notebook from student_start_pack/)
PROJECT_ROOT = Path('')
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
KAGGLE_ROOT = PROJECT_ROOT / 'ArtBench-10'

if not KAGGLE_ROOT.exists() or not (SCRIPTS_DIR / 'artbench_local_dataset.py').exists():
    raise FileNotFoundError(
        'Could not resolve project folders from relative paths. '
        'Run this notebook from student_start_pack/ or adjust PROJECT_ROOT.'
    )

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# print('PROJECT_ROOT =', PROJECT_ROOT)
# print('KAGGLE_ROOT  =', KAGGLE_ROOT)

# Uses your existing project helper to load ArtBench-10 from local Kaggle-style files
from scripts.artbench_local_dataset import load_kaggle_artbench10_splits

# hf_ds = load_kaggle_artbench10_splits(KAGGLE_ROOT)
# train_hf = hf_ds["train"]

# print("Train size:", len(train_hf))
# print("Columns   :", train_hf.column_names)

# label_feature = train_hf.features["label"]
# class_names = list(label_feature.names)
# num_classes = len(class_names)
# print("Num classes:", num_classes)
# print("Class names:", class_names)


def load_train_split():
    hf_ds = load_kaggle_artbench10_splits(KAGGLE_ROOT)
    return hf_ds["train"]

def load_test_split():
    hf_ds = load_kaggle_artbench10_splits(KAGGLE_ROOT)
    return hf_ds["test"]

def get_class_names():
    train = load_train_split()
    return list(train.features["label"].names)

def create_fixed_test_csv(
    output_path: Path,
    num_samples: int = 5000,
    seed: int = 42
):
    test = load_test_split()

    n_total = len(test)

    rng = np.random.RandomState(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    selected = indices[:num_samples]

    rows = []

    for i, idx in enumerate(selected):
        ex = test[idx]

        rows.append({
            "selected_order": i,
            "test_id_original": int(idx),
            "label_id": int(ex["label"]),
            "label_name": test.features["label"].names[int(ex["label"])]
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["selected_order", "test_id_original", "label_id", "label_name"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved test CSV → {output_path}")

IMAGE_SIZE = 32
BATCH_SIZE = 64
NUM_WORKERS = 2

def safe_num_workers(requested: int) -> int:
    # Avoid notebook multiprocessing pickling issues on macOS/ipykernel.
    if "ipykernel" in sys.modules and int(requested) > 0:
        print("Notebook kernel detected: forcing num_workers=0 for DataLoader stability.")
        return 0
    return int(requested)

EFFECTIVE_NUM_WORKERS = safe_num_workers(NUM_WORKERS)
# TRAIN_FRACTION = 1.0  # Example: 0.5 means half of train split

def get_full_train_loader(batch_size=BATCH_SIZE):
    train = load_train_split()

    dataset = HFDatasetTorch(train, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=EFFECTIVE_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

# Class distribution summary
# train_counts = Counter(train_hf["label"])

# print("\nTrain class distribution:")
# for cid, name in enumerate(class_names):
#     print(f"  {cid:2d} | {name:>18s} | {train_counts.get(cid, 0):6d}")




transform = T.Compose([
    T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),  # outputs [0,1]
])


class HFDatasetTorch(Dataset):
    def __init__(self, hf_split, transform=None, indices=None):
        self.ds = hf_split
        self.transform = transform
        self.indices = list(range(len(hf_split))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        ex = self.ds[real_idx]
        img = ex["image"]
        y = int(ex["label"])
        x = self.transform(img) if self.transform else img
        return x, y, real_idx



# def make_subset_indices(n_total: int, fraction: float, seed: int = 42):
#     n_keep = max(1, int(round(n_total * fraction)))
#     g = np.random.RandomState(seed)
#     idx = np.arange(n_total)
#     g.shuffle(idx)
#     return idx[:n_keep].tolist()


# train_indices = make_subset_indices(len(train_hf), TRAIN_FRACTION, seed=SEED)

# train_ds = HFDatasetTorch(train_hf, transform=transform, indices=train_indices)

# train_loader = DataLoader(
#     train_ds,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=EFFECTIVE_NUM_WORKERS,
#     pin_memory=torch.cuda.is_available(),
# )

# print("Train dataset length (after fraction):", len(train_ds))
# print("Train batches                        :", len(train_loader))

import csv

#warning if using colab kernel on vscode you need to put the files on your google drive and link this notebook to it.
TRAINING_CSV_PATH = Path('student_start_pack/training_20_percent.csv')
INDEX_COLUMN = 'train_id_original'  # recommended 

TEST_CSV_PATH = Path('student_start_pack/test_5000_samples.csv')
TEST_INDEX_COLUMN = 'test_id_original'


def load_ids_from_csv(csv_path: Path, index_column: str = "train_id_original") -> list[int]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"training.csv not found: {csv_path}\n"
        )

    ids = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        r = csv.DictReader(f)
        if index_column not in (r.fieldnames or []):
            raise ValueError(
                f"Column {index_column!r} not present in {csv_path}. "
                f"Available: {r.fieldnames}"
            )
        for row in r:
            v = str(row.get(index_column, "")).strip()
            if v == "":
                continue
            ids.append(int(v))

    if len(ids) == 0:
        raise ValueError(f"No ids found in {csv_path} column {index_column!r}")
    return ids



# print('Loaded ids:', len(train_ids_from_csv))
# print('First 10 ids:', train_ids_from_csv[:10])

def get_train_loader_from_csv(batch_size=BATCH_SIZE):
    train = load_train_split()
    train_ids_from_csv = load_ids_from_csv(TRAINING_CSV_PATH, index_column=INDEX_COLUMN)

    dataset = HFDatasetTorch(train, transform=transform, indices=train_ids_from_csv)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=EFFECTIVE_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

def get_test_loader_from_csv(batch_size=BATCH_SIZE):
    test = load_test_split()

    test_ids_from_csv = load_ids_from_csv(
        TEST_CSV_PATH,
        index_column=TEST_INDEX_COLUMN
    )

    dataset = HFDatasetTorch(test, transform=transform, indices=test_ids_from_csv)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=EFFECTIVE_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

def get_all_images_from_loader(loader: DataLoader, device=None):
    images = []

    for x, _, _ in loader:
        if device is not None:
            x = x.to(device)
        images.append(x)

    return torch.cat(images, dim=0)

def get_test_images_tensor(device=None, batch_size=BATCH_SIZE):
    loader = get_test_loader_from_csv(batch_size=batch_size)
    images = get_all_images_from_loader(loader, device=device)
    return images

# Build a train dataset/loader using exactly those IDs
# train_ds_from_csv = HFDatasetTorch(train_hf, transform=transform, indices=train_ids_from_csv)
# train_loader_from_csv = DataLoader(
#     train_ds_from_csv,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=EFFECTIVE_NUM_WORKERS,
#     pin_memory=torch.cuda.is_available(),
# )

# print('Subset train dataset length:', len(train_ds_from_csv))
# print('Subset train batches      :', len(train_loader_from_csv))

def show_batch_grid(loader, class_names, n_images=36, nrow=6, title='Sample Grid'):
    x, y, idx = next(iter(loader))
    x = x[:n_images]
    y = y[:n_images]

    grid = make_grid(x, nrow=nrow, padding=2)
    np_img = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(np_img)
    plt.axis('off')
    plt.title(title)
    plt.show()

    # Print labels for quick inspection
    labels_str = [class_names[int(v)] for v in y]
    print('Labels:', labels_str)


# show_batch_grid(train_loader, class_names, n_images=36, nrow=6, title='ArtBench-10 Train Samples')

import csv


def export_split_to_folder(
    loader: DataLoader,
    class_names: list[str],
    out_dir: Path,
    max_images: int | None = 500,
):
    out_dir = Path(out_dir)
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    saved = 0

    for x, y, idx in loader:
        b = x.shape[0]
        for i in range(b):
            if max_images is not None and saved >= max_images:
                break

            label_id = int(y[i].item())
            label_name = class_names[label_id]
            src_idx = int(idx[i].item())

            file_name = f"img_{saved:06d}_label{label_id:02d}_idx{src_idx:06d}.png"
            path = img_dir / file_name
            save_image(x[i], path)

            rows.append({
                'file_name': file_name,
                'label_id': label_id,
                'label_name': label_name,
                'source_index': src_idx,
            })
            saved += 1

        if max_images is not None and saved >= max_images:
            break

    csv_path = out_dir / 'metadata.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file_name', 'label_id', 'label_name', 'source_index'])
        w.writeheader()
        w.writerows(rows)

    print(f'Exported {saved} images to: {img_dir}')
    print(f'Metadata CSV: {csv_path}')


EXPORT_ROOT = Path('exported_data')
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

# export_split_to_folder(train_loader, class_names, EXPORT_ROOT / 'train_subset', max_images=500)

def save_sample_grid(samples: torch.Tensor, out_path: str, nrow: int = 6):
    """
    Saves a 6x6 grid (or configurable) of generated samples.
    
    Args:
        samples: Tensor [N, C, H, W] in [0,1]
        out_path: file path to save image
        nrow: number of images per row (default 6 → 6x6 grid)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Take first 36 images
    grid = make_grid(samples[: nrow * nrow], nrow=nrow, padding=2)

    save_image(grid, out_path)

    print(f"[Saved] sample grid → {out_path}")
