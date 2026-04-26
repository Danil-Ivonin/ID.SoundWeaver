import torch
import time

while True:
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2

    print(f"allocated={allocated:.1f}MB reserved={reserved:.1f}MB")
    time.sleep(1)