"""Training / evaluation loop with per-epoch logging."""
from __future__ import annotations
import os, time
from typing import Dict, List
import torch, torch.nn as nn
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


def _opt(model, cfg):
    if cfg["optimizer"].lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                               weight_decay=cfg["weight_decay"])
    return torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])


@torch.no_grad()
def evaluate(model, loader, device) -> Dict:
    model.eval(); correct = 0; total = 0; loss_sum = 0.0; crit = nn.CrossEntropyLoss()
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        o = model(x); loss_sum += crit(o, y).item() * x.size(0)
        p = o.argmax(1); correct += (p == y).sum().item(); total += x.size(0)
        ys.append(y.cpu().numpy()); ps.append(p.cpu().numpy())
    return {"acc": correct / total, "loss": loss_sum / total,
            "y_true": np.concatenate(ys), "y_pred": np.concatenate(ps)}


def train(model, train_loader, test_loader, device, cfg,
          log_path: str | None = None) -> pd.DataFrame:
    model.to(device); opt = _opt(model, cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    crit = nn.CrossEntropyLoss()
    logs: List[dict] = []
    best_acc = 0.0

    for ep in range(1, cfg["epochs"] + 1):
        model.train(); t0 = time.time()
        run_loss = 0.0; run_n = 0; run_correct = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{cfg['epochs']}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); o = model(x); loss = crit(o, y)
            loss.backward(); opt.step()
            run_loss += loss.item() * x.size(0); run_n += x.size(0)
            run_correct += (o.argmax(1) == y).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()
        tr_acc, tr_loss = run_correct / run_n, run_loss / run_n
        ev = evaluate(model, test_loader, device)
        row = {"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
               "test_loss": ev["loss"], "test_acc": ev["acc"],
               "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0}
        logs.append(row)
        best_acc = max(best_acc, ev["acc"])
        print(f"[E{ep:03d}] tr_acc={tr_acc:.4f} te_acc={ev['acc']:.4f} "
              f"tr_loss={tr_loss:.4f} te_loss={ev['loss']:.4f}")
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            pd.DataFrame(logs).to_csv(log_path, index=False)
    df = pd.DataFrame(logs); df.attrs["best_acc"] = best_acc
    return df


def final_eval_with_confusion(model, loader, device, n_cls: int):
    ev = evaluate(model, loader, device)
    cm = confusion_matrix(ev["y_true"], ev["y_pred"], labels=list(range(n_cls)))
    return ev, cm
