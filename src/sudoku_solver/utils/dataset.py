import numpy as np
import time
from tqdm import tqdm

def print_pipeline_performance(dataset):
    limit = 1_000
    times = []
    start = time.time()
    for i, (x, y) in tqdm(enumerate(dataset.take(limit)), total=limit, desc="Measuring pipeline performance..."):
        if i % 10 == 0:
            times.append(time.time() - start)
        start = time.time()
    print(f"Average batch fetch time is: {np.mean(times)}")