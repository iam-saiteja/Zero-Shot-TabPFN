from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

class RandomMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, output_dim: int = 1):
        super().__init__()
        if input_dim == 0:
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, output_dim)
            )
            # Initialize weights to generate smooth but active functions
            with torch.no_grad():
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=0.5)
                        nn.init.normal_(m.bias, std=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        return self.net(x)

def generate_scm_dataset(
    num_samples: int,
    num_features: int,
    task_type: str = "classification",
    num_classes: int = 2,
    parent_prob: float = 0.25,
    noise_scale: float = 0.1,
    random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic tabular data using a random Structural Causal Model (SCM)."""
    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    # We will generate features sequentially using PyTorch for the MLPs
    X = torch.zeros(num_samples, num_features, dtype=torch.float32)
    
    # Store MLPs for each feature
    mlps = []
    parent_lists = []
    
    for i in range(num_features):
        # Sample parents from preceding features
        parents = []
        if i > 0:
            for j in range(i):
                if np.random.rand() < parent_prob:
                    parents.append(j)
        
        parent_lists.append(parents)
        mlp = RandomMLP(input_dim=len(parents), output_dim=1)
        mlps.append(mlp)
        
        # Compute feature value
        with torch.no_grad():
            if len(parents) == 0:
                # Root node: standard normal noise
                val = torch.randn(num_samples, 1)
            else:
                parent_data = X[:, parents]
                val = mlp(parent_data) + torch.randn(num_samples, 1) * noise_scale
            
            X[:, i] = val.squeeze(-1)
            
    # Compute target y
    # Select a random subset of features to be the direct causes of the target
    num_causes = min(max(3, num_features // 4), num_features)
    causes = np.random.choice(num_features, size=num_causes, replace=False).tolist()
    
    if task_type == "classification":
        mlp_y = RandomMLP(input_dim=num_causes, output_dim=num_classes)
        with torch.no_grad():
            logits = mlp_y(X[:, causes])
            # Apply softmax to get class probabilities
            probs = torch.softmax(logits, dim=-1).numpy()
            
        # Sample labels from probabilities
        y = np.array([np.random.choice(num_classes, p=p) for p in probs])
    else:
        # Regression
        mlp_y = RandomMLP(input_dim=num_causes, output_dim=1)
        with torch.no_grad():
            y_val = mlp_y(X[:, causes]) + torch.randn(num_samples, 1) * noise_scale
            y = y_val.squeeze(-1).numpy()
            
    return X.numpy(), y

def main():
    print("Testing SCM Data Generator...")
    X_cls, y_cls = generate_scm_dataset(1000, 10, task_type="classification", num_classes=3, random_state=42)
    print(f"Classification Shapes: X={X_cls.shape}, y={y_cls.shape}")
    print(f"Class distribution: {np.bincount(y_cls)}")
    
    X_reg, y_reg = generate_scm_dataset(1000, 10, task_type="regression", random_state=42)
    print(f"Regression Shapes: X={X_reg.shape}, y={y_reg.shape}")
    print(f"y mean={y_reg.mean():.4f}, std={y_reg.std():.4f}")

if __name__ == "__main__":
    main()
