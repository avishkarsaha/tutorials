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

def test_single_forward():
    print("Testing single forward pass...")
    device = torch.device('cuda')
    batch_size = 1
    hidden_size = 256
    
    # Initialize layers
    print("Initializing layers...")
    pytorch_ln = torch.nn.LayerNorm(hidden_size).to(device)
    custom_ln = CustomLayerNorm(hidden_size).to(device)
    
    # Create input tensor
    print("Creating input tensor...")
    x = torch.randn(batch_size, hidden_size, device=device)
    
    # Run forward passes
    print("Running PyTorch LayerNorm...")
    out_pytorch = pytorch_ln(x)
    torch.cuda.synchronize()
    print("PyTorch LayerNorm completed")
    
    print("Running Custom LayerNorm...")
    out_custom = custom_ln(x)
    torch.cuda.synchronize()
    print("Custom LayerNorm completed")
    
    # Check results
    max_diff = torch.max(torch.abs(out_pytorch - out_custom)).item()
    print(f"Max difference between PyTorch and Custom: {max_diff}")
    
    return max_diff < 1e-5

if __name__ == "__main__":
    print("\nTesting LayerNorm Implementation...")
    print("Device:", torch.cuda.get_device_name())
    print("CUDA Version:", torch.version.cuda)
    print("PyTorch Version:", torch.__version__)
    
    try:
        success = test_single_forward()
        print("\nTest result:", "PASSED" if success else "FAILED")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
    finally:
        torch.cuda.empty_cache()
