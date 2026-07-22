from __future__ import annotations

import numpy as np
import torch


class _CNNClassifier(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(n_features, hidden, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.output = torch.nn.Linear(hidden, 1)

    def forward(self, sequence):
        encoded = self.net(sequence.transpose(1, 2)).squeeze(-1)
        return self.output(encoded)


class _LSTMClassifier(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.recurrent = torch.nn.LSTM(n_features, hidden, batch_first=True)
        self.output = torch.nn.Linear(hidden, 1)

    def forward(self, sequence):
        encoded, _ = self.recurrent(sequence)
        return self.output(encoded[:, -1])


class _TransformerClassifier(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.input = torch.nn.Linear(n_features, hidden)
        layer = torch.nn.TransformerEncoderLayer(
            hidden, nhead=2, dim_feedforward=2 * hidden, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=1)
        self.output = torch.nn.Linear(hidden, 1)

    def forward(self, sequence):
        return self.output(self.encoder(self.input(sequence)).mean(dim=1))


def _classifier_core(architecture: str, n_features: int, hidden: int):
    constructors = {
        "cnn": _CNNClassifier,
        "lstm": _LSTMClassifier,
        "transformer": _TransformerClassifier,
    }
    if architecture not in constructors:
        raise ValueError(f"unknown direct classifier architecture: {architecture}")
    return constructors[architecture](n_features, hidden).double()


class DirectPrefixClassifier:
    """Supplementary fixed-prefix classifier with an explicit architecture."""

    def __init__(self, architecture: str, n_features: int, hidden: int = 24):
        self.architecture = architecture
        self.n_features = n_features
        self.hidden = hidden
        self.net = _classifier_core(architecture, n_features, hidden)
        self.train_case_ids: tuple[str, ...] = ()
        self.val_case_ids: tuple[str, ...] = ()

    @staticmethod
    def sequence(sample):
        context = sample["model_inputs"].get("observed_context")
        if context is None:
            raise ValueError("direct classifiers require a fixed prefix request")
        return np.concatenate(
            (context["delta"], context["omega"], context["Pm"]), axis=1
        )

    def fit(self, train, val, epochs: int = 2, seed: int = 324):
        self.train_case_ids = tuple(sample["event_id"] for sample in train)
        self.val_case_ids = tuple(sample["event_id"] for sample in val)
        if set(self.train_case_ids) & set(self.val_case_ids):
            raise ValueError("train/val lineage overlap")
        sequences = [self.sequence(sample) for sample in train]
        if len({sequence.shape for sequence in sequences}) != 1:
            raise ValueError("direct classifiers require one fixed prefix length")
        labels = np.asarray(
            [
                [1.0 if sample["training_context"]["label"] == "unstable" else 0.0]
                for sample in train
            ]
        )
        if len(np.unique(labels)) < 2:
            raise ValueError("classifier train cohort must contain both classes")
        torch.manual_seed(seed)
        inputs = torch.as_tensor(np.asarray(sequences), dtype=torch.float64)
        targets = torch.as_tensor(labels, dtype=torch.float64)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        for _ in range(max(1, epochs)):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                self.net(inputs), targets
            )
            loss.backward()
            optimizer.step()

    def predict_probability(self, sample):
        sequence = torch.as_tensor(
            self.sequence(sample), dtype=torch.float64
        ).unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(self.net(sequence)).item())


class PrefixClassifier:
    """Supplementary prospective prefix classifier; never a dynamics model."""

    def __init__(self, input_size: int, hidden: int = 24):
        self.net = torch.nn.Sequential(torch.nn.Linear(input_size, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)).double()
        self.train_case_ids: tuple[str, ...] = ()
        self.val_case_ids: tuple[str, ...] = ()

    @staticmethod
    def vector(sample):
        context = sample["model_inputs"]["observed_context"]
        return np.concatenate((context["delta"].reshape(-1), context["omega"].reshape(-1), context["Pm"].reshape(-1)))

    def fit(self, train, val, epochs: int = 2, seed: int = 324):
        torch.manual_seed(seed)
        self.train_case_ids = tuple(s["event_id"] for s in train); self.val_case_ids = tuple(s["event_id"] for s in val)
        if set(self.train_case_ids) & set(self.val_case_ids): raise ValueError("train/val lineage overlap")
        x = torch.as_tensor(np.asarray([self.vector(s) for s in train]), dtype=torch.float64)
        y = torch.as_tensor([[1.0 if s["training_context"]["label"] == "unstable" else 0.0] for s in train], dtype=torch.float64)
        if len(torch.unique(y)) < 2: raise ValueError("classifier train cohort must contain both classes")
        optimizer=torch.optim.Adam(self.net.parameters(),lr=1e-3)
        for _ in range(max(1,epochs)):
            optimizer.zero_grad(); loss=torch.nn.functional.binary_cross_entropy_with_logits(self.net(x),y); loss.backward(); optimizer.step()

    def predict_probability(self, sample):
        with torch.no_grad(): return float(torch.sigmoid(self.net(torch.as_tensor(self.vector(sample),dtype=torch.float64))).item())
