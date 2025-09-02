import torch
import torch.nn as nn
import coremltools as ct
import os

MODEL_CONFIGS = {
    # (N, K, M)
    # A(N, K) @ B(K, M)
    "matmul_fp16_64x2048": (64, 2048, 2048),
    "matmul_fp16_256x2048": (256, 2048, 2048),
    "matmul_fp16_1024x2048": (1024, 2048, 2048),
    "matmul_fp16_2048x2048": (2048, 2048, 2048),
    "matmul_fp16_4096x2048": (4096, 2048, 2048),
    "matmul_fp16_tall_8192x512": (8192, 512, 2048),
    "matmul_fp16_fat_512x8192": (512, 8192, 2048),
}

OUTPUT_DIR = "../models/"


class StackedMM(nn.Module):
    def __init__(self):
        super(StackedMM, self).__init__()

    def forward(self, a_B_N_K: torch.tensor, b_B_K_M: torch.tensor) -> torch.tensor:
        return torch.matmul(a_B_N_K, b_B_K_M)


def generate_model(name: str, N: int, K: int, M: int):
    torch_model = StackedMM().eval()
    tensor_a = torch.rand(1, N, K)
    tensor_b = torch.rand(1, K, M)

    traced_model = torch.jit.trace(torch_model, (tensor_a, tensor_b))
    mlmodel = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="A", shape=tensor_a.shape),
            ct.TensorType(name="B", shape=tensor_b.shape),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )

    output_path = os.path.join(OUTPUT_DIR, f"{name}.mlpackage")
    mlmodel.save(output_path)
    print(f"Generated model: {name}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, (N, K, M) in MODEL_CONFIGS.items():
        generate_model(name, N, K, M)
