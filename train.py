"""Vision transformer training script"""

import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from vision_transformer import VisionTransformer

torch.manual_seed(42)


# hyperparameters
PATCH_SIZE = 4
EMBED_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
NUM_CLASSES = 10
DROPOUT = 0.25
BATCH_SIZE = 256
EPOCHS = 15
CKPT_INTEVAL = 5
LR = 3e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PATCHES = (32 // PATCH_SIZE) ** 2


# load data
def unpickle(file: str) -> dict:
    """Unpickles files"""
    with open(file, "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    return d


train_batches = [f"data/data_batch_{i}" for i in range(1, 6)]
train_data = [torch.tensor(unpickle(p)[b"data"]) for p in train_batches]
train_labels = [torch.tensor(unpickle(p)[b"labels"]) for p in train_batches]
X_train = torch.concat(train_data).view(-1, 3, 32, 32)
y_train = torch.concat(train_labels)
X_test = torch.tensor(unpickle("data/test_batch")[b"data"]).view(-1, 3, 32, 32)
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

# shuffle and split data
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


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes the accuracy score"""
    with torch.no_grad():
        return (F.softmax(logits, dim=-1).argmax(-1) == targets).float().mean().item()


run_label = f"P{PATCH_SIZE}_D{EMBED_DIM}_H{NUM_HEADS}_L{NUM_LAYERS}_DO{DROPOUT}_E{EPOCHS}_LR{LR}"
writer = SummaryWriter(log_dir=f"runs/{run_label}")


# training loop
for e in range(1, EPOCHS + 1):

    # training
    vit.train()
    for step, batch in enumerate(train_dl):
        print(f"epoch {e}/{EPOCHS} | step {step}/{train_steps}", end="\r")
        X = batch[0].to(DEVICE) / 255.0  # scale and cast to float32
        y = batch[1].to(DEVICE).long()  # cast to int64

        logits = vit(X)
        loss = F.cross_entropy(logits, y)

        # log stats
        global_step = (e - 1) * train_steps + step + 1
        writer.add_scalar("train/loss", loss.item(), global_step)
        writer.add_scalar("train/accuracy", accuracy(logits, y), global_step)

        loss.backward()
        optim.step()
        optim.zero_grad()
        scheduler.step()

    # validation
    vit.eval()
    val_loss = val_accuracy = 0.0
    with torch.no_grad():
        for batch in val_dl:
            X = batch[0].to(DEVICE) / 255.0
            y = batch[1].to(DEVICE).long()
            logits = vit(X)
            val_loss += F.cross_entropy(logits, y).item()
            val_accuracy += accuracy(logits, y)

    val_loss /= len(val_dl)
    val_accuracy /= len(val_dl)
    writer.add_scalar("validation/loss", val_loss, e)
    writer.add_scalar("validation/accuracy", val_accuracy, e)

    # save checkpoints
    if e > 1 and e % CKPT_INTEVAL == 0:
        sd = {"model": vit.state_dict(), "optim": optim.state_dict(), "epoch": e}
        torch.save(sd, f"vit_{e}.pt")
