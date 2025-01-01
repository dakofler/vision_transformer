"""Vision transformer training script"""

import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from vision_transformer import VisionTransformer

PATCH_SIZE = 4
EMBED_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
NUM_CLASSES = 10
DROPOUT = 0.5

BATCH_SIZE = 256
EPOCHS = 50
CKPT_INTEVAL = 25
LR = 3e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PATCHES = (32 // PATCH_SIZE) ** 2


# load and preprocessdata
def unpickle(file: str) -> dict:
    """Unpickles files"""
    with open(file, "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    return d


train_batches = [
    "data/data_batch_1",
    "data/data_batch_2",
    "data/data_batch_3",
    "data/data_batch_4",
    "data/data_batch_5",
]
X_train = (
    torch.concat(
        [torch.tensor(unpickle(p)[b"data"]) for p in train_batches], dim=0
    ).view(-1, 3, 32, 32)
    / 255.0
)
y_train = torch.concat(
    [torch.tensor(unpickle(p)[b"labels"]) for p in train_batches], dim=0
)
X_test = torch.tensor(unpickle("data/test_batch")[b"data"]).view(-1, 3, 32, 32) / 255.0
y_test = torch.tensor(unpickle("data/test_batch")[b"labels"])

labels = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

rand_idx = torch.randperm(len(X_train))
X_train_shuffled, y_train_shuffled = X_train[rand_idx], y_train[rand_idx]
n = int(0.8 * len(X_train))
X_train = X_train_shuffled[:n]
y_train = y_train_shuffled[:n]
X_val = X_train_shuffled[n:]
y_val = y_train_shuffled[n:]

train_dl = DataLoader(TensorDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
val_dl = DataLoader(TensorDataset(X_val, y_val), BATCH_SIZE, shuffle=False)

# create model instance
vit = VisionTransformer(
    in_channels=3,
    num_patches=NUM_PATCHES,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
).to(DEVICE)

# training
optim = torch.optim.AdamW(vit.parameters(), lr=LR)
train_steps = len(train_dl)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, T_max=EPOCHS * train_steps, eta_min=LR / 10
)


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy score"""
    with torch.no_grad():
        return (F.softmax(logits, dim=-1).argmax(-1) == target).float().mean().item()


for e in range(1, EPOCHS + 1):

    # training
    vit.train()
    train_loss = train_accuracy = 0.0
    for step, batch in enumerate(train_dl):
        print(f"step {step}/{train_steps}", end="\r")
        X = batch[0].to(DEVICE)
        y = batch[1].to(DEVICE)

        logits = vit(X)
        loss = F.cross_entropy(logits, y)
        train_loss += loss.item()
        train_accuracy += accuracy(logits, y)

        loss.backward()
        optim.step()
        optim.zero_grad()
        scheduler.step()

    train_loss /= len(train_dl)
    train_accuracy /= len(train_dl)

    # validation
    vit.eval()
    val_loss = val_accuracy = 0.0
    with torch.no_grad():
        for batch in val_dl:
            X = batch[0].to(DEVICE)
            y = batch[1].to(DEVICE)
            logits = vit(X)
            val_loss += F.cross_entropy(logits, y).item()
            val_accuracy += accuracy(logits, y)

    val_loss /= len(val_dl)
    val_accuracy /= len(val_dl)

    # save checkpoints
    if e > 1 and e % CKPT_INTEVAL == 0:
        sd = {"model": vit.state_dict(), "optim": optim.state_dict(), "epoch": e}
        torch.save(sd, f"vit_{e}.pt")

    print(
        f"epoch {e}/{EPOCHS} | train_loss {train_loss:.4f} | train_acc {train_accuracy:.4f} | val_loss {val_loss:.4f} | val_acc {val_accuracy:.4f}"
    )
