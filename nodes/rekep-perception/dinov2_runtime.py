from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def bounded_kmeans(features: Any, clusters: int, *, max_iterations: int = 100, tolerance: float = 1e-4) -> tuple[Any, Any]:
    """Bounded k-means used instead of the upstream unbounded convergence loop."""
    import torch

    values = features.float()
    if values.ndim != 2 or not values.shape[0]:
        raise RuntimeError("k-means features must be a non-empty matrix")
    if not torch.isfinite(values).all():
        raise RuntimeError("k-means features contain non-finite values")
    count = min(max(1, int(clusters)), int(values.shape[0]))
    indices = torch.linspace(0, values.shape[0] - 1, count, device=values.device).round().long()
    centers = values.index_select(0, indices).clone()
    assignments = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
    for _ in range(max(1, int(max_iterations))):
        distances = torch.cdist(values, centers)
        assignments = distances.argmin(dim=1)
        previous = centers.clone()
        for cluster in range(count):
            members = values[assignments == cluster]
            if members.numel():
                centers[cluster] = members.mean(dim=0)
        shift = torch.linalg.vector_norm(centers - previous)
        if torch.isfinite(shift) and float(shift) <= tolerance:
            break
    return assignments.cpu(), centers.cpu()


class DinoV2:
    def __init__(self, config_path: Path):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid DINOv2 model config {config_path}: {exc}") from exc
        for field in ("source_path", "weights_path", "model_name", "device"):
            if not isinstance(config.get(field), str) or not config[field]:
                raise RuntimeError(f"DINOv2 model config requires {field}")
        self.config = config
        self._model: Any = None
        weights = Path(config["weights_path"]).expanduser()
        if not weights.is_file():
            raise RuntimeError(f"DINOv2 weights are missing: {weights}")
        self.weights_digest = "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest()

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        import torch

        source = Path(self.config["source_path"]).expanduser()
        model = torch.hub.load(str(source), self.config["model_name"], source="local", pretrained=False)
        checkpoint = torch.load(self.config["weights_path"], map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict):
            checkpoint = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if not isinstance(checkpoint, dict):
            raise RuntimeError("DINOv2 checkpoint does not contain a state dictionary")
        cleaned = {str(key).removeprefix("module.").removeprefix("backbone."): value for key, value in checkpoint.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if len(missing) > 32 or len(unexpected) > 32:
            raise RuntimeError(f"DINOv2 checkpoint mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
        device = self.config["device"]
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"configured DINOv2 device is unavailable: {device}")
        self._model = model.eval().to(device)
        return self._model

    def feature_map(self, rgb: Any) -> Any:
        import torch
        import torch.nn.functional as functional

        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        tensor = functional.interpolate(tensor, size=(476, 630), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).to(self.config["device"])
        with torch.inference_mode():
            tokens = self._load().forward_features(tensor)["x_norm_patchtokens"][0].float().cpu()
        grid_height = max(1, round((tokens.shape[0] * 476 / 630) ** 0.5))
        grid_width = max(1, tokens.shape[0] // grid_height)
        return tokens[: grid_height * grid_width].reshape(grid_height, grid_width, -1)

    def propose(self, feature_map: Any, mask: Any, count: int) -> list[tuple[int, int, float]]:
        import numpy as np
        import torch

        rows, columns = np.where(mask)
        grid_height, grid_width = feature_map.shape[:2]
        patch_rows = np.clip((rows * grid_height / mask.shape[0]).astype(int), 0, grid_height - 1)
        patch_columns = np.clip((columns * grid_width / mask.shape[1]).astype(int), 0, grid_width - 1)
        pairs = np.unique(np.stack([patch_rows, patch_columns], axis=1), axis=0)
        vectors = feature_map[torch.from_numpy(pairs[:, 0]), torch.from_numpy(pairs[:, 1])]
        assignments, centers = bounded_kmeans(vectors, count)
        result = []
        for cluster in range(centers.shape[0]):
            members = torch.nonzero(assignments == cluster).squeeze(1)
            if not members.numel():
                continue
            member_vectors = vectors.index_select(0, members)
            local = torch.linalg.vector_norm(member_vectors - centers[cluster], dim=1).argmin()
            pair = pairs[int(members[int(local)])]
            v = min(mask.shape[0] - 1, int((pair[0] + 0.5) * mask.shape[0] / grid_height))
            u = min(mask.shape[1] - 1, int((pair[1] + 0.5) * mask.shape[1] / grid_width))
            confidence = float(torch.sigmoid(torch.linalg.vector_norm(centers[cluster])).item())
            result.append((u, v, confidence))
        return result
