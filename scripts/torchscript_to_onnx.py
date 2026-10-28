"""Convert the bundled G1 TorchScript policy to an ONNX runtime model."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "third_party" / "unitree_rl_gym" / "deploy" / "pre_train" / "g1" / "motion.pt"
DEFAULT_OUTPUT = DEFAULT_INPUT.with_suffix(".onnx")


def _exportable_policy(torch, policy):
    class RecurrentPolicy(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            self.memory = torch.nn.LSTM(input_size=47, hidden_size=64, num_layers=1)
            self.memory.load_state_dict(source.memory.state_dict())
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(64, 32),
                torch.nn.ELU(),
                torch.nn.Linear(32, 12),
            )
            self.actor.load_state_dict(source.actor.state_dict())

        def forward(self, observation, hidden_state, cell_state):
            memory_output, (hidden_state_out, cell_state_out) = self.memory(
                observation.unsqueeze(0), (hidden_state, cell_state)
            )
            action = self.actor(memory_output.squeeze(0))
            return action, hidden_state_out, cell_state_out

    return RecurrentPolicy(policy).eval()


def convert(source: Path, destination: Path) -> None:
    try:
        import onnx
        import torch
    except ImportError as exc:
        raise RuntimeError("Conversion requires torch and onnx on the development machine") from exc

    policy = torch.jit.load(str(source), map_location="cpu").eval()
    exportable_policy = _exportable_policy(torch, policy)
    sample = torch.zeros((1, 47), dtype=torch.float32)
    hidden_state = torch.zeros((1, 1, 64), dtype=torch.float32)
    cell_state = torch.zeros((1, 1, 64), dtype=torch.float32)
    with torch.inference_mode():
        action, next_hidden, next_cell = exportable_policy(sample, hidden_state, cell_state)
        policy.hidden_state.zero_()
        policy.cell_state.zero_()
        reference_action = policy(sample)
    if action.shape != (1, 12) or not torch.isfinite(action).all():
        raise ValueError(f"expected a finite [1, 12] policy output, got {tuple(action.shape)}")
    if next_hidden.shape != hidden_state.shape or next_cell.shape != cell_state.shape:
        raise ValueError("policy recurrent state shape changed during inference")
    if not torch.allclose(action, reference_action, rtol=1e-5, atol=1e-6):
        raise ValueError("stateless ONNX wrapper does not match the TorchScript policy output")

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        exportable_policy,
        (sample, hidden_state, cell_state),
        str(destination),
        input_names=["observation", "hidden_state", "cell_state"],
        output_names=["action", "hidden_state_out", "cell_state_out"],
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
