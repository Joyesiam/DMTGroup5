"""
Model definitions for all iterations.
Every model returns an sklearn-compatible interface (fit/predict).
PyTorch models are wrapped in sklearn-compatible classes.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LassoCV, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import RANDOM_SEED, VERBOSE, N_JOBS

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# === Tabular Models ===

def get_random_forest(task="classification", **kwargs):
    defaults = dict(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_SEED, n_jobs=N_JOBS, verbose=VERBOSE,
    )
    defaults.update(kwargs)
    if task == "classification":
        defaults.setdefault("class_weight", "balanced")
        return RandomForestClassifier(**defaults)
    return RandomForestRegressor(**defaults)


def get_xgboost(task="classification", **kwargs):
    if not HAS_XGBOOST:
        raise ImportError("xgboost not installed")
    defaults = dict(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, verbosity=0,
    )
    defaults.update(kwargs)
    if task == "classification":
        defaults.setdefault("eval_metric", "mlogloss")
        return XGBClassifier(**defaults)
    defaults.setdefault("eval_metric", "rmse")
    return XGBRegressor(**defaults)


def get_gradient_boosting(task="classification", **kwargs):
    defaults = dict(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_SEED, verbose=VERBOSE,
    )
    defaults.update(kwargs)
    if task == "classification":
        return GradientBoostingClassifier(**defaults)
    return GradientBoostingRegressor(**defaults)


def get_svm(task="classification", **kwargs):
    defaults = dict(kernel="rbf", random_state=RANDOM_SEED)
    defaults.update(kwargs)
    if task == "classification":
        defaults.setdefault("class_weight", "balanced")
        defaults.setdefault("probability", True)
        return SVC(**defaults)
    return SVR(**defaults)


def get_knn(task="classification", **kwargs):
    defaults = dict(n_neighbors=5, weights="distance", n_jobs=N_JOBS)
    defaults.update(kwargs)
    if task == "classification":
        return KNeighborsClassifier(**defaults)
    return KNeighborsRegressor(**defaults)


def get_naive_bayes(task="classification", **kwargs):
    if task != "classification":
        raise ValueError("Naive Bayes only supports classification")
    defaults = dict()
    defaults.update(kwargs)
    return GaussianNB(**defaults)


def get_decision_tree(task="classification", **kwargs):
    defaults = dict(
        max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_SEED,
    )
    defaults.update(kwargs)
    if task == "classification":
        defaults.setdefault("class_weight", "balanced")
        return DecisionTreeClassifier(**defaults)
    return DecisionTreeRegressor(**defaults)


def get_mlp(task="classification", **kwargs):
    defaults = dict(
        hidden_layer_sizes=(64, 32), max_iter=500,
        early_stopping=True, validation_fraction=0.15,
        random_state=RANDOM_SEED, verbose=False,
    )
    defaults.update(kwargs)
    if task == "classification":
        return MLPClassifier(**defaults)
    return MLPRegressor(**defaults)


def get_lasso(task="regression", **kwargs):
    """Iter 132: LASSO regression (L1 regularization)."""
    if task != "regression":
        raise ValueError("LASSO only supports regression")
    defaults = dict(max_iter=10000)
    defaults.update(kwargs)
    # Remove random_state since LassoCV doesn't use it for fitting
    defaults.pop("random_state", None)
    return LassoCV(cv=5, **defaults)


def get_ridge(task="regression", **kwargs):
    """Iter 133: Ridge regression (L2 regularization)."""
    if task != "regression":
        raise ValueError("Ridge only supports regression")
    defaults = dict()
    defaults.update(kwargs)
    defaults.pop("random_state", None)
    return RidgeCV(cv=5, alphas=[0.01, 0.1, 1.0, 10.0, 100.0])


def get_elasticnet(task="regression", **kwargs):
    """Iter 133: ElasticNet regression (L1+L2)."""
    if task != "regression":
        raise ValueError("ElasticNet only supports regression")
    defaults = dict(max_iter=10000)
    defaults.update(kwargs)
    defaults.pop("random_state", None)
    return ElasticNetCV(cv=5, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], **defaults)


class TransformerModel(nn.Module):
    """Iter 140: Simple Transformer for mood prediction."""
    def __init__(self, input_dim, d_model=64, nhead=2, num_layers=2,
                 dim_feedforward=128, dropout=0.1, output_dim=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling over sequence
        x = self.dropout(x)
        return self.fc(x)


class LSTMWithEmbedding(nn.Module):
    """Iter 134: LSTM with learned patient embeddings."""
    def __init__(self, input_dim, hidden_dim=32, n_patients=27,
                 embed_dim=8, dropout=0.2, output_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(n_patients, embed_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim + embed_dim, 16)
        self.fc2 = nn.Linear(16, output_dim)

    def forward(self, x, patient_ids=None):
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])
        if patient_ids is not None:
            emb = self.embedding(patient_ids)
            h = torch.cat([h, emb], dim=1)
            h = torch.relu(self.fc1(h))
            h = self.dropout(h)
        return self.fc2(h)


def get_transformer(input_dim, task="regression", **kwargs):
    """Get Transformer temporal model."""
    model_kwargs = dict(input_dim=input_dim, d_model=64, nhead=2, num_layers=2,
                        dim_feedforward=128, dropout=0.1)
    model_kwargs.update({k: v for k, v in kwargs.items()
                         if k in ["d_model", "nhead", "num_layers", "dim_feedforward", "dropout"]})
    wrapper_kwargs = {k: v for k, v in kwargs.items()
                      if k in ["lr", "epochs", "patience", "batch_size"]}
    return TemporalModelWrapper(TransformerModel, model_kwargs, task=task, **wrapper_kwargs)


def get_stacking_classifier(estimators=None, **kwargs):
    if estimators is None:
        estimators = [
            ("rf", get_random_forest("classification")),
            ("svm", get_svm("classification")),
        ]
        if HAS_XGBOOST:
            estimators.append(("xgb", get_xgboost("classification")))
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        cv=3, n_jobs=N_JOBS, verbose=VERBOSE,
        **kwargs,
    )


def get_stacking_regressor(estimators=None, **kwargs):
    if estimators is None:
        estimators = [
            ("rf", get_random_forest("regression")),
        ]
        if HAS_XGBOOST:
            estimators.append(("xgb", get_xgboost("regression")))
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3, n_jobs=N_JOBS, verbose=VERBOSE,
        **kwargs,
    )


# === PyTorch Temporal Models ===

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, n_layers=1, dropout=0.3, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.fc(out)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, n_layers=1, dropout=0.3,
                 output_dim=1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0,
                          bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        fc_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_dim, output_dim)

    def forward(self, x):
        _, h_n = self.gru(x)
        if self.bidirectional:
            out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            out = h_n[-1]
        out = self.dropout(out)
        return self.fc(out)


class CNN1DModel(nn.Module):
    def __init__(self, input_dim, n_filters=32, kernel_size=3, dropout=0.3, output_dim=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.mean(dim=2)  # Global average pooling
        x = self.dropout(x)
        return self.fc(x)


class TemporalModelWrapper(BaseEstimator):
    """
    Sklearn-compatible wrapper for PyTorch temporal models.
    Handles training with early stopping and prediction.
    """
    def __init__(self, model_class, model_kwargs, task="regression",
                 lr=0.001, epochs=100, patience=15, batch_size=32,
                 weight_decay=0.0):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.task = task
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.model_ = None
        self.train_losses_ = []
        self.val_losses_ = []

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        n_outputs = len(np.unique(y)) if self.task == "classification" else 1
        kwargs = dict(self.model_kwargs)
        kwargs["output_dim"] = n_outputs

        self.model_ = self.model_class(**kwargs)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)

        if self.task == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        X_t = torch.FloatTensor(X)
        if self.task == "classification":
            y_t = torch.LongTensor(y)
        else:
            y_t = torch.FloatTensor(y).unsqueeze(1)

        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val)
            if self.task == "classification":
                y_val_t = torch.LongTensor(y_val)
            else:
                y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.train_losses_ = []
        self.val_losses_ = []

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.model_(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)

            avg_train_loss = epoch_loss / len(X_t)
            self.train_losses_.append(avg_train_loss)

            # Validation for early stopping
            if X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_output = self.model_(X_val_t)
                    val_loss = criterion(val_output, y_val_t).item()
                self.val_losses_.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict(self, X):
        self.model_.eval()
        X_t = torch.FloatTensor(X)
        with torch.no_grad():
            output = self.model_(X_t)
            if self.task == "classification":
                return output.argmax(dim=1).numpy()
            return output.squeeze(1).numpy()

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba only for classification")
        self.model_.eval()
        X_t = torch.FloatTensor(X)
        with torch.no_grad():
            output = self.model_(X_t)
            return torch.softmax(output, dim=1).numpy()


def get_lstm(input_dim, task="regression", **kwargs):
    model_kwargs = dict(input_dim=input_dim, hidden_dim=32, n_layers=1, dropout=0.3)
    model_kwargs.update({k: v for k, v in kwargs.items()
                         if k in ["hidden_dim", "n_layers", "dropout"]})
    wrapper_kwargs = {k: v for k, v in kwargs.items()
                      if k in ["lr", "epochs", "patience", "batch_size"]}
    return TemporalModelWrapper(LSTMModel, model_kwargs, task=task, **wrapper_kwargs)


def get_gru(input_dim, task="regression", **kwargs):
    model_kwargs = dict(input_dim=input_dim, hidden_dim=32, n_layers=1, dropout=0.3,
                        bidirectional=False)
    model_kwargs.update({k: v for k, v in kwargs.items()
                         if k in ["hidden_dim", "n_layers", "dropout", "bidirectional"]})
    wrapper_kwargs = {k: v for k, v in kwargs.items()
                      if k in ["lr", "epochs", "patience", "batch_size", "weight_decay"]}
    return TemporalModelWrapper(GRUModel, model_kwargs, task=task, **wrapper_kwargs)


def get_cnn1d(input_dim, task="regression", **kwargs):
    model_kwargs = dict(input_dim=input_dim, n_filters=32, kernel_size=3, dropout=0.3)
    model_kwargs.update({k: v for k, v in kwargs.items()
                         if k in ["n_filters", "kernel_size", "dropout"]})
    wrapper_kwargs = {k: v for k, v in kwargs.items()
                      if k in ["lr", "epochs", "patience", "batch_size"]}
    return TemporalModelWrapper(CNN1DModel, model_kwargs, task=task, **wrapper_kwargs)
