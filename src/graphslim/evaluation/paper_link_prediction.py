"""Paper-aligned link-prediction evaluation for HyDRO.

This module implements the protocol stated in the paper:

* 70/10/20 edge split;
* two-layer GCN encoder with 128 hidden units;
* Adam with learning rate 0.001 for 100 epochs;
* balanced positive/negative link examples; and
* mean and population standard deviation over ten seeded runs.

The public repository does not disclose how its dense, weighted synthetic
adjacency was converted into binary LP supervision. Candidate cutoffs are
therefore selected exclusively by validation F1. The test set is evaluated
only once after the winning cutoff and epoch have been fixed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import dense_to_sparse, remove_self_loops, to_undirected
from torch_sparse import SparseTensor

from graphslim.dataset.utils import get_syn_data
from graphslim.utils import seed_everything


class InvalidSyntheticThresholdError(ValueError):
    """A cutoff cannot form balanced synthetic link supervision."""


class PaperGCNLinkPredictor(nn.Module):
    """Two-layer GCN encoder with a dot-product decoder."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def encode(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        return self.conv2(x, edge_index, edge_weight=edge_weight)

    @staticmethod
    def decode(z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)


class PaperLinkPredictionEvaluator:
    """Evaluate an original or condensed graph with the paper LP protocol."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

    def evaluate(
        self,
        data,
        reduced=True,
        verbose=None,
        validation_only=False,
    ):
        args = self.args
        verbose = args.verbose if verbose is None else verbose

        if reduced and (
            not hasattr(data, 'feat_syn') or not hasattr(data, 'adj_syn')
        ):
            data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(
                data,
                args,
                model_type='GCN',
                verbose=verbose,
            )

        thresholds = self._parse_thresholds(data) if reduced else [None]
        val_scores = []
        test_scores = []
        selected_thresholds = []

        for run_idx in range(args.run_eval):
            run_seed = args.seed + run_idx
            seed_everything(run_seed)
            split = self._make_original_split(data)

            best = None
            for threshold in thresholds:
                seed_everything(run_seed)
                try:
                    candidate = self._train_candidate(
                        data,
                        split,
                        reduced,
                        threshold,
                    )
                except InvalidSyntheticThresholdError as error:
                    args.logger.warning(
                        'Skipping synthetic threshold %s: %s',
                        threshold,
                        error,
                    )
                    continue
                if best is None or candidate['val_f1'] > best['val_f1']:
                    best = candidate

            if best is None:
                raise RuntimeError(
                    'No synthetic adjacency threshold produced both positive '
                    'and negative training pairs.'
                )

            test_f1 = None
            if not validation_only:
                test_f1 = self._test_best_state(split, best)
                test_scores.append(test_f1)

            val_scores.append(best['val_f1'])
            selected_thresholds.append(best['synthetic_threshold'])

            if verbose:
                threshold_text = (
                    'whole'
                    if best['synthetic_threshold'] is None
                    else f"{best['synthetic_threshold']:.6g}"
                )
                message = (
                    f"LP run {run_idx + 1}/{args.run_eval}: "
                    f"threshold={threshold_text}, "
                    f"epoch={best['best_epoch'] + 1}, "
                    f"val_f1={100 * best['val_f1']:.2f}"
                )
                if test_f1 is not None:
                    message += f", test_f1={100 * test_f1:.2f}"
                print(message)

        val_scores = np.asarray(val_scores, dtype=np.float64)
        test_scores = np.asarray(test_scores, dtype=np.float64)
        result = {
            'val_mean': float(val_scores.mean()),
            'val_std': float(val_scores.std(ddof=0)),
            'test_mean': (
                None if validation_only else float(test_scores.mean())
            ),
            'test_std': (
                None if validation_only else float(test_scores.std(ddof=0))
            ),
            'selected_thresholds': selected_thresholds,
        }

        if validation_only:
            args.logger.info(
                'LP validation mean F1: %.2f +/- %.2f',
                100 * result['val_mean'],
                100 * result['val_std'],
            )
        else:
            args.logger.info(
                'LP test mean F1: %.2f +/- %.2f '
                '(validation %.2f +/- %.2f)',
                100 * result['test_mean'],
                100 * result['test_std'],
                100 * result['val_mean'],
                100 * result['val_std'],
            )
        return result

    def evaluate_validation(self, data, reduced=True, run_count=None):
        old_run_eval = self.args.run_eval
        if run_count is not None:
            self.args.run_eval = run_count
        try:
            result = self.evaluate(
                data,
                reduced=reduced,
                verbose=False,
                validation_only=True,
            )
            return result['val_mean']
        finally:
            self.args.run_eval = old_run_eval

    def _make_original_split(self, data):
        x = self._as_float_tensor(data.feat_full).cpu()
        edge_index = self._original_edge_index(data, x.size(0))
        graph = Data(x=x, edge_index=edge_index, num_nodes=x.size(0))
        transform = RandomLinkSplit(
            num_val=0.10,
            num_test=0.20,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=self.args.lp_neg_ratio,
        )
        train_data, val_data, test_data = transform(graph)
        return {
            'x': train_data.x.to(self.device),
            'message_edge_index': train_data.edge_index.to(self.device),
            'train_edge_label_index': (
                train_data.edge_label_index.to(self.device)
            ),
            'train_edge_label': train_data.edge_label.float().to(self.device),
            'val_edge_label_index': val_data.edge_label_index.to(self.device),
            'val_edge_label': val_data.edge_label.float().to(self.device),
            'test_edge_label_index': test_data.edge_label_index.to(self.device),
            'test_edge_label': test_data.edge_label.float().to(self.device),
        }

    def _train_candidate(self, data, split, reduced, threshold):
        if reduced:
            train = self._synthetic_training_data(data, threshold)
        else:
            train = {
                'x': split['x'],
                'message_edge_index': split['message_edge_index'],
                'message_edge_weight': None,
                'edge_label_index': split['train_edge_label_index'],
                'edge_label': split['train_edge_label'],
            }

        model = PaperGCNLinkPredictor(
            train['x'].size(1),
            self.args.lp_hidden,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.args.lp_lr,
            weight_decay=self.args.lp_weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_f1 = -1.0
        best_epoch = -1
        best_state = None
        interval = max(int(self.args.lp_val_interval), 1)

        for epoch in range(self.args.lp_epochs):
            model.train()
            optimizer.zero_grad()
            z_train = model.encode(
                train['x'],
                train['message_edge_index'],
                train['message_edge_weight'],
            )
            logits = model.decode(z_train, train['edge_label_index'])
            loss = criterion(logits, train['edge_label'])
            loss.backward()
            optimizer.step()

            should_validate = (
                (epoch + 1) % interval == 0
                or epoch + 1 == self.args.lp_epochs
            )
            if not should_validate:
                continue

            val_f1 = self._score_original(
                model,
                split,
                split['val_edge_label_index'],
                split['val_edge_label'],
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        return {
            'state_dict': best_state,
            'val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'synthetic_threshold': threshold,
            'in_channels': train['x'].size(1),
        }

    def _test_best_state(self, split, best):
        model = PaperGCNLinkPredictor(
            best['in_channels'],
            self.args.lp_hidden,
        ).to(self.device)
        model.load_state_dict(best['state_dict'])
        return self._score_original(
            model,
            split,
            split['test_edge_label_index'],
            split['test_edge_label'],
        )

    def _score_original(self, model, split, edge_label_index, edge_label):
        model.eval()
        with torch.no_grad():
            z = model.encode(
                split['x'],
                split['message_edge_index'],
            )
            probability = model.decode(z, edge_label_index).sigmoid()
        prediction = (
            probability >= self.args.lp_decision_threshold
        ).long().cpu().numpy()
        target = edge_label.long().cpu().numpy()
        return float(
            f1_score(
                target,
                prediction,
                average=self.args.lp_f1_average,
                zero_division=0,
            )
        )

    def _parse_thresholds(self, data):
        adjacency = self._dense_adjacency(data.adj_syn).float().cpu()
        adjacency = (adjacency + adjacency.T) / 2.0
        upper = torch.triu(
            torch.ones_like(adjacency, dtype=torch.bool),
            diagonal=1,
        )
        edge_weights = adjacency[upper]
        if edge_weights.numel() == 0:
            raise ValueError('Synthetic adjacency has no off-diagonal pairs')

        tokens = [
            token.strip().lower()
            for token in str(self.args.lp_syn_thresholds).split(',')
            if token.strip()
        ]
        if not tokens:
            raise ValueError('--lp_syn_thresholds cannot be empty')

        thresholds = []
        for token in tokens:
            if token.startswith('q'):
                percentile = float(token[1:])
                if not 0.0 < percentile < 100.0:
                    raise ValueError(
                        'Synthetic adjacency quantiles must be within q0-q100'
                    )
                thresholds.append(
                    float(torch.quantile(edge_weights, percentile / 100.0))
                )
            else:
                value = float(token)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        'Synthetic adjacency thresholds must be within [0, 1]'
                    )
                thresholds.append(value)
        return sorted(set(thresholds))

    def _synthetic_training_data(self, data, threshold):
        x = self._as_float_tensor(data.feat_syn).to(self.device)
        adjacency = self._dense_adjacency(data.adj_syn).float().cpu()
        adjacency = (adjacency + adjacency.T) / 2.0
        adjacency.fill_diagonal_(0.0)
        upper = torch.triu(
            torch.ones_like(adjacency, dtype=torch.bool),
            diagonal=1,
        )

        positive = (
            (adjacency >= threshold) & upper
        ).nonzero(as_tuple=False).T.contiguous()
        negative_pool = (
            (adjacency < threshold) & upper
        ).nonzero(as_tuple=False).T.contiguous()
        positive_count = positive.size(1)
        negative_count = int(
            round(positive_count * self.args.lp_neg_ratio)
        )

        if positive_count == 0:
            raise InvalidSyntheticThresholdError('no positive links')
        if negative_pool.size(1) < negative_count:
            raise InvalidSyntheticThresholdError(
                f'{positive_count} positives but only '
                f'{negative_pool.size(1)} negative candidates'
            )

        permutation = torch.randperm(negative_pool.size(1))[:negative_count]
        negative = negative_pool[:, permutation]
        edge_label_index = torch.cat([positive, negative], dim=1)
        edge_label = torch.cat(
            [torch.ones(positive_count), torch.zeros(negative_count)],
            dim=0,
        )

        message_pairs = (
            (adjacency > 0) & upper
        ).nonzero(as_tuple=False).T.contiguous()
        if message_pairs.size(1) == 0:
            raise InvalidSyntheticThresholdError(
                'synthetic adjacency has no positive weights'
            )
        message_edge_index = torch.cat(
            [message_pairs, message_pairs.flip(0)],
            dim=1,
        )
        message_weight = adjacency[
            message_pairs[0],
            message_pairs[1],
        ]
        message_edge_weight = torch.cat(
            [message_weight, message_weight],
            dim=0,
        )

        return {
            'x': x,
            'message_edge_index': message_edge_index.to(self.device),
            'message_edge_weight': message_edge_weight.to(self.device),
            'edge_label_index': edge_label_index.to(self.device),
            'edge_label': edge_label.to(self.device),
        }

    @staticmethod
    def _as_float_tensor(value):
        if torch.is_tensor(value):
            return value.detach().float()
        return torch.as_tensor(value, dtype=torch.float32)

    @staticmethod
    def _dense_adjacency(adjacency):
        if isinstance(adjacency, SparseTensor):
            return adjacency.to_dense()
        if torch.is_tensor(adjacency):
            return adjacency.to_dense() if adjacency.is_sparse else adjacency.detach()
        if hasattr(adjacency, 'toarray'):
            return torch.as_tensor(adjacency.toarray())
        return torch.as_tensor(adjacency)

    @staticmethod
    def _original_edge_index(data, num_nodes):
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            edge_index = data.edge_index.detach().cpu().long()
        else:
            adjacency = data.adj_full
            if isinstance(adjacency, SparseTensor):
                row, col, _ = adjacency.coo()
                edge_index = torch.stack([row, col], dim=0).cpu().long()
            elif torch.is_tensor(adjacency):
                if adjacency.is_sparse:
                    edge_index = adjacency.coalesce().indices().cpu().long()
                else:
                    edge_index, _ = dense_to_sparse(adjacency.cpu())
            elif hasattr(adjacency, 'tocoo'):
                coo = adjacency.tocoo()
                edge_index = torch.as_tensor(
                    np.vstack([coo.row, coo.col]),
                    dtype=torch.long,
                )
            else:
                raise TypeError('Cannot convert original adjacency to edges')

        edge_index, _ = remove_self_loops(edge_index)
        return to_undirected(edge_index, num_nodes=num_nodes)
