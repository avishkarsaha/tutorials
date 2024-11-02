import torch
import torch.nn as nn
import layernorm_cuda
import time

class CustomLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layernorm_cuda.forward(x, self.weight, self.bias, self.eps)

def benchmark_config(batch_size: int, hidden_size: int, dtype: torch.dtype, 
                    num_warmup: int = 50, num_iters: int = 1000):
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    # Initialize layers
    pytorch_ln = torch.nn.LayerNorm(hidden_size).to(device).to(dtype)
    custom_ln = CustomLayerNorm(hidden_size).to(device).to(dtype)
    
    # Create input tensor
    x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(num_warmup):
        _ = pytorch_ln(x)
        _ = custom_ln(x)
        torch.cuda.synchronize()
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = pytorch_ln(x)
    torch.cuda.synchronize()
    pt_time = (time.perf_counter() - start) / num_iters * 1000  # ms
    
    # Benchmark Custom
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = custom_ln(x)
    torch.cuda.synchronize()
    custom_time = (time.perf_counter() - start) / num_iters * 1000  # ms
    
    # Verify correctness
    out_pytorch = pytorch_ln(x)
    out_custom = custom_ln(x)
    max_diff = torch.max(torch.abs(out_pytorch - out_custom)).item()
    
    return pt_time, custom_time, max_diff

def print_header(text):
    print(f"\n{'-' * 80}")
    print(text)
    print('-' * 80)

def format_row(batch_size, hidden_size, pt_time, custom_time, max_diff):
    speedup = pt_time / custom_time
    return f"B={batch_size:<4} H={hidden_size:<5} | PyTorch: {pt_time:>6.3f}ms | Custom: {custom_time:>6.3f}ms | " \
           f"Speedup: {speedup:>5.2f}x | MaxDiff: {max_diff:.2e}"

def main():
    print("\nLayerNorm Benchmark")
    print("Device:", torch.cuda.get_device_name())
    print("CUDA:", torch.version.cuda)
    print("PyTorch:", torch.__version__)
    
    # Test configurations
    batch_sizes = [1, 8, 32, 128, 512]
    hidden_sizes = [256, 512, 1024, 2048, 4096]
    dtypes = [torch.float32, torch.float16]
    
    for dtype in dtypes:
        print_header(f"Testing {dtype}")
        print(f"{'Configuration':<20} | {'PyTorch':>15} | {'Custom':>15} | {'Speedup':>10} | {'Max Diff'}")
        print('-' * 80)
        
        results = []
        for batch_size in batch_sizes:
            for hidden_size in hidden_sizes:
                try:
                    pt_time, custom_time, max_diff = benchmark_config(
                        batch_size=batch_size,
                        hidden_size=hidden_size,
                        dtype=dtype
                    )
                    
                    print(format_row(batch_size, hidden_size, pt_time, custom_time, max_diff))
                    results.append((batch_size, hidden_size, pt_time, custom_time, max_diff))
                except Exception as e:
                    print(f"Error with B={batch_size}, H={hidden_size}: {str(e)}")
        
        # Print summary
        if results:
            speedups = [pt_time / custom_time for _, _, pt_time, custom_time, _ in results]
            max_diffs = [diff for _, _, _, _, diff in results]
            
            print('\nSummary:')
            print(f"Average Speedup: {sum(speedups)/len(speedups):.2f}x")
            print(f"Best Speedup: {max(speedups):.2f}x")
            print(f"Worst Speedup: {min(speedups):.2f}x")
            print(f"Max Numerical Difference: {max(max_diffs):.2e}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
