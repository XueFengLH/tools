from tqdm import tqdm
import time
from tqdm import tqdm

# 外层循环
from tqdm import tqdm

epochs = 30
num_batch=500
for epoch_step in range(epochs):
    with tqdm(range(num_batch)) as pbar:
        for batch_step in pbar:
            time.sleep(0.001)
            pbar.set_description(f"Epochs {epoch_step+1}/{epochs}")
            # pbar.set_postfix({'batch loss': 'This is batch loss' })
        pbar.set_postfix({'epoch loss': 'This is epoch loss'})