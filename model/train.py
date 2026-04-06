import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rag"))
from rag_config import faiss_path, NPROBE, parse_gpu_args

from model_config import MODEL_NAME, PQ_NAME, MAX_LENGTH, LR, EPOCHS, BATCH_SIZE, NUM_LABELS, TOP_K, NUM_WORKERS, PREFETCH
from dataset import ClaimDataset, collate_fn
from model import FactChecker

args, device = parse_gpu_args()
print(f"Device: {device}")

index = faiss.read_index(faiss_path(PQ_NAME))
faiss.extract_index_ivf(index).nprobe = NPROBE
print(f"FAISS index: {index.ntotal:,} vectors")

train_ds = ClaimDataset("train", MODEL_NAME, index, TOP_K, MAX_LENGTH, device)
test_ds = ClaimDataset("test", MODEL_NAME, index, TOP_K, MAX_LENGTH, device)

dl_kwargs = dict(num_workers=NUM_WORKERS, prefetch_factor=PREFETCH, persistent_workers=True, collate_fn=collate_fn)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **dl_kwargs)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, **dl_kwargs)

model = FactChecker(MODEL_NAME, NUM_LABELS).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

history = {"train_loss_steps": [], "val_loss": [], "val_acc": []}

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")
    t_data, t_fwd, t_bwd, t_log = 0, 0, 0, 0
    sync = torch.cuda.synchronize if device == "cuda" else lambda: None
    t0 = time.perf_counter()
    for input_ids, attention_mask, labels in pbar:
        t_data += time.perf_counter() - t0

        t0 = time.perf_counter()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        sync()
        t_fwd += time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sync()
        t_bwd += time.perf_counter() - t0

        t0 = time.perf_counter()
        total_loss += loss.item()
        history["train_loss_steps"].append(loss.item())
        avg = total_loss / (pbar.n + 1)
        n = pbar.n + 1
        pbar.set_description(
            f"Epoch {epoch+1}/{EPOCHS} loss={avg:.4f} "
            f"data={t_data/n:.3f}s fwd={t_fwd/n:.3f}s bwd={t_bwd/n:.3f}s log={t_log/n:.3f}s"
        )
        t_log += time.perf_counter() - t0
        t0 = time.perf_counter()

    avg_train_loss = total_loss / len(train_dl)

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(test_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [val]"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            val_loss += criterion(logits, labels).item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(test_dl)
    accuracy = correct / total

    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(accuracy)
    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  val_acc={accuracy:.4f}")

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/fact_checker.pt")
print("Saved to checkpoints/fact_checker.pt")

# plot
epochs_range = range(1, EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_loss_steps"], alpha=0.3, label="train (step)")
steps_per_epoch = len(train_dl)
val_x = [(i + 1) * steps_per_epoch - 1 for i in range(EPOCHS)]
ax1.plot(val_x, history["val_loss"], marker="o", label="val (epoch)")
ax1.set_xlabel("Step")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.set_title("Loss")

ax2.plot(epochs_range, history["val_acc"], label="val accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.set_title("Val Accuracy")

plt.tight_layout()
plt.savefig("checkpoints/training_curves.png", dpi=150)
print("Plot saved to checkpoints/training_curves.png")
