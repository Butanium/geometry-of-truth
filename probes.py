import torch as th
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


class LRProbe(nn.Module):
    def __init__(self, d_in, bias=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 1, bias=bias), nn.Sigmoid())

    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)

    def pred(self, x, iid=None):
        return self(x).round()

    def from_data(
        acts,
        labels,
        bias=False,
        lr=0.001,
        weight_decay=0.1,
        epochs=10_000,
        device="cpu",
        algo="sklearn",
    ):
        probe = LRProbe(acts.shape[-1], bias=bias).to(device)
        probe.fit(
            acts,
            labels,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            device=device,
            algo=algo,
        )
        return probe

    def fit(
        self,
        acts,
        labels,
        lr=0.001,
        weight_decay=0.1,
        epochs=10_000,
        device="cpu",
        algo="sklearn",
    ):
        if algo == "sklearn":
            classifier = LogisticRegression(
                class_weight="balanced", solver="newton-cholesky"
            )
            classifier.fit(acts.cpu(), labels.cpu())
            self.net[0].weight.data.copy_(th.tensor(classifier.coef_, device=device))
            self.net[0].bias.data.copy_(th.tensor(classifier.intercept_, device=device))
        elif algo == "sgd":
            acts, labels = acts.to(device), labels.to(device)
            opt = th.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
            for _ in range(epochs):
                opt.zero_grad()
                loss = th.nn.BCELoss()(self(acts), labels)
                loss.backward()
                opt.step()
        elif algo == "elk":
            try:
                from elk.training.classifier import Classifier
            except ImportError:
                raise ImportError("Please install elk using pip install eleuther-elk")
            classifier = Classifier(acts.shape[-1], num_classes=1, device=device)
            classifier.fit(acts, labels, l2_penalty=weight_decay, max_iter=epochs)
            self.net[0].weight.data.copy_(classifier.linear.weight.data)
            self.net[0].bias.data.copy_(classifier.linear.bias.data)
        else:
            raise ValueError(
                f"Unknown algo {algo}. Must be one of 'sklearn', 'sgd', or 'elk'"
            )
    
    @th.no_grad()
    def accuracy(self, acts, labels):
        return (self.pred(acts) == labels).float().mean()

    @property
    def direction(self):
        return self.net[0].weight.data[0]


class MMProbe(nn.Module):
    def __init__(self, direction, covariance=None, inv=None, atol=1e-3):
        super().__init__()
        self.direction = nn.Parameter(direction, requires_grad=False)
        if inv is None:
            self.inv = nn.Parameter(
                th.linalg.pinv(covariance, hermitian=True, atol=atol),
                requires_grad=False,
            )
        else:
            self.inv = nn.Parameter(inv, requires_grad=False)

    def forward(self, x, iid=False):
        if iid:
            return nn.Sigmoid()(x @ self.inv @ self.direction)
        else:
            return nn.Sigmoid()(x @ self.direction)

    def pred(self, x, iid=False):
        return self(x, iid=iid).round()

    def from_data(acts, labels, atol=1e-3, device="cpu"):
        acts, labels
        pos_acts, neg_acts = acts[labels == 1], acts[labels == 0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean

        centered_data = th.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
        covariance = centered_data.t() @ centered_data / acts.shape[0]

        probe = MMProbe(direction, covariance=covariance).to(device)

        return probe


def ccs_loss(probe, acts, neg_acts):
    p_pos = probe(acts)
    p_neg = probe(neg_acts)
    consistency_losses = (p_pos - (1 - p_neg)) ** 2
    confidence_losses = th.min(th.stack((p_pos, p_neg), dim=-1), dim=-1).values ** 2
    return th.mean(consistency_losses + confidence_losses)


class CCSProbe(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 1, bias=False), nn.Sigmoid())

    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)

    def pred(self, acts, iid=None):
        return self(acts).round()

    def from_data(
        acts,
        neg_acts,
        labels=None,
        lr=0.001,
        weight_decay=0.1,
        epochs=1000,
        device="cpu",
    ):
        acts, neg_acts = acts.to(device), neg_acts.to(device)
        probe = CCSProbe(acts.shape[-1]).to(device)

        opt = th.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = ccs_loss(probe, acts, neg_acts)
            loss.backward()
            opt.step()

        if labels is not None:  # flip direction if needed
            acc = (probe.pred(acts) == labels).float().mean()
            if acc < 0.5:
                probe.net[0].weight.data *= -1

        return probe

    @property
    def direction(self):
        return self.net[0].weight.data[0]
