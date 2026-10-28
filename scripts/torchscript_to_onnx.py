"""Convert the bundled G1 TorchScript policy to an ONNX runtime model."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "third_party" / "unitree_rl_gym" / "deploy" / "pre_train" / "g1" / "motion.pt"
DEFAULT_OUTPUT = DEFAULT_INPUT.with_suffix(".onnx")


def convert(source: Path, destination: Path) -> None:
    try:
        import onnx
        import torch
    except ImportError as exc:
        raise RuntimeError("Conversion requires torch and onnx on the development machine") from exc

    policy = torch.jit.load(str(source), map_location="cpu").eval()
    sample = torch.zeros((1, 47), dtype=torch.float32)
    with torch.inference_mode():
        output = policy(sample)
    if output.shape != (1, 12) or not torch.isfinite(output).all():
        raise ValueError(f"expected a finite [1, 12] policy output, got {tuple(output.shape)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        policy,
        sample,
        str(destination),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(destination))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    convert(args.input, args.output)
    print(f"Converted {args.input} to {args.output}")
