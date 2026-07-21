"""Phase 0 gate: torch sees the GPU and Blackwell (sm_120) kernels actually run."""
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
a = torch.randn(512, 512, device="cuda")
b = torch.randn(512, 512, device="cuda")
c = (a @ b).sum()
torch.cuda.synchronize()
print("matmul ok, sum=", float(c))
x = torch.randn(2, 3, 8, 224, 224, device="cuda", dtype=torch.bfloat16)
w = torch.nn.Conv3d(3, 8, 3, device="cuda", dtype=torch.bfloat16)
y = w(x).float().mean()
torch.cuda.synchronize()
print("bf16 conv3d ok, mean=", float(y))
print("GATE PASSED: sm_120 works in this container")
