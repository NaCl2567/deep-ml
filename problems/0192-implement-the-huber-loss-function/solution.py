import torch
import torch.nn.functional as F

def huber_loss(y_true, y_pred, delta=1.0) -> float:
    y_true = torch.as_tensor(y_true, dtype=torch.float32)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32)
    return F.huber_loss(y_pred, y_true, delta=delta, reduction='mean').item()