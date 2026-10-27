"""
PatchCore Anomaly Detection Architecture.
Implements patch feature embedding, greedy k-Center coreset subsampling,
fast vector nearest-neighbor search, and spatial anomaly map generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    from sklearn.neighbors import NearestNeighbors


class PatchCoreDetector(nn.Module):
    """
    PatchCore Unsupervised Anomaly Detection model.
    Learns distribution of normal local patch features and identifies
    unseen defects via distance to the normal feature memory bank.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        layers: Optional[List[str]] = None,
        coreset_ratio: float = 0.10,
        n_neighbors: int = 9,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.layers = layers or ["layer2", "layer3"]
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors
        self.device = device or torch.device("cpu")

        self.memory_bank: Optional[np.ndarray] = None
        self.faiss_index = None
        self.sk_nn = None
        self.score_min: float = 0.0
        self.score_max: float = 1.0
        self.calibrated_threshold: float = 0.5

        # Initialize backbone and hooks
        self._init_backbone()

    def _init_backbone(self) -> None:
        """Initializes torchvision backbone and registers forward hooks."""
        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
        elif self.backbone_name == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.DEFAULT
            self.backbone = models.wide_resnet50_2(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}. Choose 'resnet18' or 'wide_resnet50_2'.")

        self.backbone.to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.features: Dict[str, torch.Tensor] = {}
        self.hooks = []
        for layer_name in self.layers:
            layer = getattr(self.backbone, layer_name)
            hook = layer.register_forward_hook(self._get_hook(layer_name))
            self.hooks.append(hook)

    def _get_hook(self, layer_name: str):
        def hook(module, input_tensor, output_tensor):
            self.features[layer_name] = output_tensor
        return hook

    def extract_patch_features(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extracts concatenated multiscale spatial patch embeddings from specified intermediate layers.
        tensor: [B, C, H, W]
        Returns:
          features: [B * H_patch * W_patch, D]
          patch_grid_shape: (H_patch, W_patch)
        """
        tensor = tensor.to(self.device)
        self.features.clear()
        with torch.no_grad():
            _ = self.backbone(tensor)

        layer_feats = [self.features[layer] for layer in self.layers]
        target_h, target_w = layer_feats[0].shape[-2:]

        # Harmonize feature map resolutions to target_h, target_w
        resized_feats = []
        for feat in layer_feats:
            if feat.shape[-2:] != (target_h, target_w):
                feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            # Local average pooling (3x3 receptive field aggregation)
            feat = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
            resized_feats.append(feat)

        # Concatenate along channel dimension [B, D_total, H_patch, W_patch]
        concat_feats = torch.cat(resized_feats, dim=1)
        b, d, hp, wp = concat_feats.shape
        # Permute and reshape to [B, H_patch, W_patch, D_total] -> [B * H_patch * W_patch, D_total]
        reshaped = concat_feats.permute(0, 2, 3, 1).reshape(-1, d)

        return reshaped, (hp, wp)

    @staticmethod
    def kcenter_greedy(features: np.ndarray, sampling_ratio: float, seed: int = 42) -> np.ndarray:
        """
        Greedy k-Center Coreset Selection:
        Iteratively selects the point furthest from the currently selected centers
        to uniformly cover the normal feature space without redundancy.
        """
        n_samples = features.shape[0]
        n_coreset = max(1, int(n_samples * sampling_ratio))
        if n_coreset >= n_samples:
            return features

        np.random.seed(seed)
        first_idx = np.random.randint(0, n_samples)
        selected_indices = [first_idx]

        # Initialize minimum distance array
        dists = np.linalg.norm(features - features[first_idx], axis=1)

        # Iteratively pick points with maximum distance to selected set
        # For efficiency, if n_coreset is large, subsample batches or step
        step_limit = min(n_coreset, 2000)
        for _ in range(1, step_limit):
            farthest_idx = int(np.argmax(dists))
            selected_indices.append(farthest_idx)
            new_dists = np.linalg.norm(features - features[farthest_idx], axis=1)
            dists = np.minimum(dists, new_dists)

        return features[selected_indices]

    def fit(self, train_loader: torch.utils.data.DataLoader) -> None:
        """
        Builds the memory bank from normal training images.
        """
        self.backbone.eval()
        feature_list = []

        with torch.no_grad():
            for batch in train_loader:
                imgs = batch["tensor"].to(self.device)
                patch_feats, _ = self.extract_patch_features(imgs)
                feature_list.append(patch_feats.cpu().numpy())

        all_features = np.concatenate(feature_list, axis=0)

        # Apply coreset subsampling
        if self.coreset_ratio < 1.0:
            self.memory_bank = self.kcenter_greedy(all_features, self.coreset_ratio)
        else:
            self.memory_bank = all_features

        # Build nearest neighbor search structure
        self.memory_bank = np.ascontiguousarray(self.memory_bank, dtype=np.float32)
        if FAISS_AVAILABLE:
            dim = self.memory_bank.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(self.memory_bank)
            self.faiss_index = index
        else:
            self.sk_nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean", algorithm="auto")
            self.sk_nn.fit(self.memory_bank)

    def predict_anomaly_map(self, tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Infers pixel anomaly map and image-level anomaly score for a single image tensor [1, C, H, W].
        Returns:
          anomaly_map: 2D numpy array [H, W] normalized to [0, 1]
          image_score: scalar anomaly score
        """
        if self.memory_bank is None:
            raise RuntimeError("Model has not been trained. Fit memory bank before running inference.")

        b, c, h, w = tensor.shape
        patch_feats, (hp, wp) = self.extract_patch_features(tensor)
        query = np.ascontiguousarray(patch_feats.cpu().numpy(), dtype=np.float32)

        if FAISS_AVAILABLE and self.faiss_index is not None:
            # Query nearest neighbor distances
            distances, _ = self.faiss_index.search(query, k=1)
            # FAISS returns squared L2 distances
            distances = np.sqrt(np.maximum(distances, 0.0)).reshape(hp, wp)
        elif self.sk_nn is not None:
            distances, _ = self.sk_nn.kneighbors(query, n_neighbors=1)
            distances = distances.reshape(hp, wp)
        else:
            # Fallback simple brute force
            dists = np.linalg.norm(query[:, None, :] - self.memory_bank[None, :, :], axis=2)
            distances = np.min(dists, axis=1).reshape(hp, wp)

        # Bilinear interpolation back to input image size [H, W]
        dist_tensor = torch.from_numpy(distances).unsqueeze(0).unsqueeze(0).float()
        interpolated = F.interpolate(dist_tensor, size=(h, w), mode="bilinear", align_corners=False)
        anomaly_map = interpolated.squeeze().numpy()

        # Image-level anomaly score: maximum patch distance
        image_score = float(np.max(distances))

        # Dynamic or calibrated score normalization
        if self.score_max > self.score_min:
            norm_map = np.clip((anomaly_map - self.score_min) / (self.score_max - self.score_min + 1e-8), 0.0, 1.0)
            norm_score = float(np.clip((image_score - self.score_min) / (self.score_max - self.score_min + 1e-8), 0.0, 1.0))
        else:
            norm_map = anomaly_map
            norm_score = image_score

        return norm_map, norm_score

    def calibrate_scores(self, normal_scores: List[float], defective_scores: Optional[List[float]] = None) -> None:
        """
        Updates min/max reference bounds for robust visualization normalization.
        """
        if normal_scores:
            self.score_min = float(np.min(normal_scores))
            if defective_scores:
                self.score_max = float(np.max(defective_scores))
            else:
                self.score_max = float(np.max(normal_scores) * 1.5)

    def save_checkpoint(self, filepath: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Saves PatchCore memory bank, configuration, and calibration parameters."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_type": "patchcore",
            "backbone_name": self.backbone_name,
            "layers": self.layers,
            "coreset_ratio": self.coreset_ratio,
            "n_neighbors": self.n_neighbors,
            "memory_bank": self.memory_bank,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "calibrated_threshold": self.calibrated_threshold,
            "metadata": metadata or {}
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path], device: Optional[torch.device] = None) -> "PatchCoreDetector":
        """Loads a saved PatchCore checkpoint and reconstructs the search index."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        dev = device or torch.device("cpu")
        model = cls(
            backbone_name=data["backbone_name"],
            layers=data["layers"],
            coreset_ratio=data["coreset_ratio"],
            n_neighbors=data["n_neighbors"],
            device=dev
        )
        model.memory_bank = data["memory_bank"]
        model.score_min = data.get("score_min", 0.0)
        model.score_max = data.get("score_max", 1.0)
        model.calibrated_threshold = data.get("calibrated_threshold", 0.5)

        # Rebuild nearest neighbor search structure
        if model.memory_bank is not None:
            model.memory_bank = np.ascontiguousarray(model.memory_bank, dtype=np.float32)
            if FAISS_AVAILABLE:
                dim = model.memory_bank.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(model.memory_bank)
                model.faiss_index = index
            else:
                model.sk_nn = NearestNeighbors(n_neighbors=model.n_neighbors, metric="euclidean", algorithm="auto")
                model.sk_nn.fit(model.memory_bank)

        return model
