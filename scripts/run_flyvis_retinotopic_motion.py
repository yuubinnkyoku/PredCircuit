from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
)
from predcircuit.model import PredictiveCodingGraph

DIRECTIONS = (0.0, 90.0, 180.0, 270.0)
KNOWN_PREFERRED_DIRECTION = {
    "T4a": 180.0,
    "T4b": 0.0,
    "T4c": 90.0,
    "T4d": 270.0,
    "T5a": 180.0,
    "T5b": 0.0,
    "T5c": 90.0,
    "T5d": 270.0,
}


def _input_xy(circuit: RetinotopicFlyVisCircuit) -> torch.Tensor:
    indices = torch.tensor(circuit.input_nodes, dtype=torch.long)
    u = circuit.node_u[indices].float()
    v = circuit.node_v[indices].float()
    x = u + 0.5 * v
    y = (math.sqrt(3.0) / 2.0) * v
    return torch.stack([x, y], dim=1)


def moving_bar_frames(
    circuit: RetinotopicFlyVisCircuit,
    *,
    direction_deg: float,
    contrast: str,
    frames: int,
    width: float,
) -> torch.Tensor:
    """Render a Gaussian bar moving over the explicit FlyVis input lattice."""
    if contrast not in {"on", "off"}:
        raise ValueError("contrast must be 'on' or 'off'")
    xy = _input_xy(circuit)
    theta = math.radians(direction_deg)
    direction = torch.tensor([math.cos(theta), math.sin(theta)], dtype=torch.float32)
    projection = xy @ direction
    span = float(projection.abs().max()) + 1.0
    centers = torch.linspace(-span, span, frames)
    rendered = []
    for center in centers:
        bar = torch.exp(-0.5 * ((projection - center) / width).square())
        rendered.append(bar if contrast == "on" else 1.0 - bar)
    return torch.stack(rendered)


def sequence_responses(
    circuit: RetinotopicFlyVisCircuit,
    *,
    model: PredictiveCodingGraph,
    direction_deg: float,
    contrast: str,
    frames: int,
    width: float,
    inference_steps: int,
    step_size: float,
    warm_state: bool,
) -> dict[str, float]:
    stimulus = moving_bar_frames(
        circuit,
        direction_deg=direction_deg,
        contrast=contrast,
        frames=frames,
        width=width,
    )
    state = torch.zeros(1, circuit.graph.num_nodes, dtype=torch.float32)
    clamp_mask = torch.zeros(circuit.graph.num_nodes, dtype=torch.bool)
    clamp_mask[list(circuit.input_nodes)] = True
    baseline = 0.0 if contrast == "on" else 1.0

    # Establish a reproducible pre-stimulus state. Keeping this state between frames creates
    # a transient temporal response; resetting below is the matched static control.
    clamp_values = torch.zeros_like(state)
    clamp_values[:, list(circuit.input_nodes)] = baseline
    state, _ = model.infer(
        state,
        clamp_mask=clamp_mask,
        clamp_values=clamp_values,
        steps=max(inference_steps * 2, 1),
        step_size=step_size,
    )
    baseline_state = state.clone()

    target_types = tuple(KNOWN_PREFERRED_DIRECTION)
    central_nodes = {cell_type: circuit.central_node(cell_type) for cell_type in target_types}
    traces: dict[str, list[float]] = {cell_type: [] for cell_type in target_types}

    for frame in stimulus:
        if not warm_state:
            state = baseline_state.clone()
        clamp_values = torch.zeros_like(state)
        clamp_values[:, list(circuit.input_nodes)] = frame
        state, _ = model.infer(
            state,
            clamp_mask=clamp_mask,
            clamp_values=clamp_values,
            steps=inference_steps,
            step_size=step_size,
        )
        if not torch.isfinite(state).all():
            raise RuntimeError("non-finite state in retinotopic motion pilot")
        for cell_type, node in central_nodes.items():
            traces[cell_type].append(float(state[0, node]))

    # T4 is the ON pathway and T5 the OFF pathway. Squared transient activity makes the
    # tuning statistic non-negative without assuming our simplified state sign matches firing.
    return {
        cell_type: sum(value * value for value in values) / max(len(values), 1)
        for cell_type, values in traces.items()
        if (cell_type.startswith("T4") and contrast == "on")
        or (cell_type.startswith("T5") and contrast == "off")
    }


def circular_tuning(responses: dict[float, float]) -> tuple[float, float]:
    total = sum(max(value, 0.0) for value in responses.values())
    if total <= 1e-12:
        return float("nan"), 0.0
    real = 0.0
    imag = 0.0
    for angle, value in responses.items():
        radians = math.radians(angle)
        real += value * math.cos(radians)
        imag += value * math.sin(radians)
    strength = math.hypot(real, imag) / total
    preferred = math.degrees(math.atan2(imag, real)) % 360.0
    return preferred, strength


def angular_error(a: float, b: float) -> float:
    if not math.isfinite(a):
        return float("nan")
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def evaluate_circuit(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    scramble_seed: int,
    frames: int,
    width: float,
    inference_steps: int,
    step_size: float,
    init_scale: float,
) -> list[dict[str, float | int | str | bool]]:
    model = PredictiveCodingGraph(
        circuit.graph,
        use_biological_strength=True,
        init_scale=init_scale,
        seed=0,
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for mode, warm_state in (("sequential", True), ("reset_each_frame", False)):
        per_type: dict[str, dict[float, float]] = {
            cell_type: {} for cell_type in KNOWN_PREFERRED_DIRECTION
        }
        for direction in DIRECTIONS:
            on = sequence_responses(
                circuit,
                model=model,
                direction_deg=direction,
                contrast="on",
                frames=frames,
                width=width,
                inference_steps=inference_steps,
                step_size=step_size,
                warm_state=warm_state,
            )
            off = sequence_responses(
                circuit,
                model=model,
                direction_deg=direction,
                contrast="off",
                frames=frames,
                width=width,
                inference_steps=inference_steps,
                step_size=step_size,
                warm_state=warm_state,
            )
            for cell_type, value in {**on, **off}.items():
                per_type[cell_type][direction] = value

        for cell_type, known in KNOWN_PREFERRED_DIRECTION.items():
            preferred, strength = circular_tuning(per_type[cell_type])
            row: dict[str, float | int | str | bool] = {
                "topology": topology,
                "scramble_seed": scramble_seed,
                "mode": mode,
                "cell_type": cell_type,
                "known_preferred_deg": known,
                "preferred_deg": preferred,
                "angular_error_deg": angular_error(preferred, known),
                "vector_strength": strength,
                "nodes": circuit.graph.num_nodes,
                "edges": circuit.graph.num_edges,
                "finite": math.isfinite(preferred) and math.isfinite(strength),
            }
            for direction in DIRECTIONS:
                row[f"response_{int(direction)}"] = per_type[cell_type][direction]
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retinotopic FlyVis moving-bar transient-direction pilot"
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--scramble-seeds", type=int, default=8)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--bar-width", type=float, default=0.55)
    parser.add_argument("--inference-steps", type=int, default=5)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--init-scale", type=float, default=0.08)
    parser.add_argument(
        "--out", type=Path, default=Path("results/generated/flyvis_retinotopic_motion.csv")
    )
    args = parser.parse_args()

    spec = load_flyvis_spec()
    biological = graph_from_flyvis_retinotopy(spec, extent=args.extent)
    rows = evaluate_circuit(
        biological,
        topology="biological",
        scramble_seed=-1,
        frames=args.frames,
        width=args.bar_width,
        inference_steps=args.inference_steps,
        step_size=args.step_size,
        init_scale=args.init_scale,
    )
    for seed in range(args.scramble_seeds):
        scrambled = graph_from_flyvis_retinotopy(
            spec,
            extent=args.extent,
            pair_rotation_seed=10_000 + seed,
        )
        rows.extend(
            evaluate_circuit(
                scrambled,
                topology="pair_rotation_scramble",
                scramble_seed=seed,
                frames=args.frames,
                width=args.bar_width,
                inference_steps=args.inference_steps,
                step_size=args.step_size,
                init_scale=args.init_scale,
            )
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = (
        frame.groupby(["topology", "mode"])[["angular_error_deg", "vector_strength"]]
        .agg(["mean", "median", "std"])
        .sort_index()
    )
    edge_counts = frame.groupby("topology")["edges"].agg(["min", "max", "mean"])
    print(
        f"Retinotopic FlyVis crop: extent={args.extent}, "
        f"{biological.graph.num_nodes} nodes, {biological.graph.num_edges} biological edges"
    )
    print("\nDirection-tuning summary across T4/T5 types:")
    print(summary.to_string())
    print("\nEdge-count check:")
    print(edge_counts.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
