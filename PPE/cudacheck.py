import torch

def gpu_test():
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = (x @ y).mean()
        print(f"GPU test successful! Result: {z:.4f}")
        print(f"Allocated memory: {torch.cuda.memory_allocated()/1e6:.2f} MB")
        print(f"Cached memory: {torch.cuda.memory_reserved()/1e6:.2f} MB")
    else:
        print("GPU not available")

gpu_test()