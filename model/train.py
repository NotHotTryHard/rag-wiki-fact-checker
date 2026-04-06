import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rag"))
from rag_config import faiss_path, NPROBE, parse_gpu_args

from model_config import MODEL_NAME, PQ_NAME, MAX_LENGTH, LR, EPOCHS, BATCH_SIZE, NUM_LABELS, TOP_K
from dataset import ClaimDataset
from model import FactChecker

args, device = parse_gpu_args()
print(f"Device: {device}")

index = faiss.read_index(faiss_path(PQ_NAME))
faiss.extract_index_ivf(index).nprobe = NPROBE
print(f"FAISS index: {index.ntotal:,} vectors")

train_ds = ClaimDataset("train", MODEL_NAME, index, TOP_K, MAX_LENGTH, device)
test_ds = ClaimDataset("test", MODEL_NAME, index, TOP_K, MAX_LENGTH, device)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

model = FactChecker(MODEL_NAME, NUM_LABELS).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")
    for input_ids, attention_mask, labels in pbar:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

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

    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  val_acc={accuracy:.4f}")

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/fact_checker.pt")
print("Saved to checkpoints/fact_checker.pt")
