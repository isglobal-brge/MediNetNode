"""
FedSVM Core Implementation - Copied from FSV project.

This module contains the core FedSVM implementation, including:
- FedSVMClientOptMD: Optimized Multiple Deltas client (primary algorithm)
- Helper classes for delta computation and kernel operations
- Privacy-preserving support vector perturbation logic

Reference: Support Vector Federation (FedSVM) paper
"""

import numpy as np
import torch
from torch import optim
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from hashlib import sha256
from collections import OrderedDict
from copy import deepcopy


# ==================== Helper Functions ====================

def h(alphas, svs):
    """
    Construct the hyperplane:
                w = sum_i alpha_i * sv_i
    where alpha_i is the dual variable and sv_i is a support vector.
    """
    res = 0
    for i in range(len(alphas)):
        res += alphas[i] * svs[i]
    return res


# ==================== Delta Strategy Mixins ====================

class FedSVM_Optim:
    """
    Mixin class that defines the fitting behavior when delta(s) vector(s)
    are found using an optimization algorithm, so that support vectors
    are shifted parallel to the decision boundary.
    """

    def fit(self, device, svs_to_opt) -> torch.Tensor:
        print(f"[CONFIG] [Client {self.client_no}] Optimizing delta displacements using {len(self.svs)} svs...")

        patience = 20
        not_improving = 0

        delta = self.create_delta(device, svs_to_opt)
        best_delta = delta.clone()
        best_loss = np.inf

        optimizer = optim.Adam([delta], lr=0.001)
        num_iterations = 20000

        K = self.kernel_fun(self.svs, self.svs, **self.kernel)

        for i in range(num_iterations):
            optimizer.zero_grad()
            loss_tuple = self.objective_function(delta, K, svs_to_opt)
            loss = loss_tuple[0] + loss_tuple[1]

            loss.backward()
            optimizer.step()

            if loss >= best_loss:
                not_improving += 1
                if not_improving > patience:
                    print(f"   [Client {self.client_no}] NOT IMPROVING: stopped at iter {i+1} with loss {loss:.2f}")
                    break
            else:
                not_improving = 0
                best_loss = loss
                best_delta = delta.clone()

            if (
                loss < self.client_eps
                or (torch.norm(delta.grad) / delta.shape[0]) < 1e-6
            ):
                print(f"   [Client {self.client_no}] Converged at iter {i+1} with loss {loss:.2f}")
                break

            if (i + 1) % 100 == 0:
                print(f"   [Client {self.client_no}] Iter {i+1}: loss {loss:.4f}")

        return best_delta


class FedSVM_MultipleDeltas:
    """
    Mixin class that defines the way multiple deltas are created,
    the objective function to optimize them, and how they are collected.
    """

    def create_delta(self, device, svs_to_opt):
        """
        Create a matrix of delta vectors by randomly sampling from a normal distribution,
        one for each support vector, rescaled by a factor of 0.001.
        The delta0 vector contains the desired norm of each delta vector.
        """
        delta = np.random.randn(len(svs_to_opt), self.svs.shape[1]) * 0.001
        delta = torch.tensor(
            delta, requires_grad=True, dtype=torch.float32, device=device
        )
        self.delta0 = (0.1 + torch.rand([len(svs_to_opt)]) * 0.2) * torch.ones(
            len(svs_to_opt)
        )

        return delta

    def objective_function(self, delta, K, svs_to_opt):
        """
        The objective function to optimize the delta vectors is the sum of two terms:
            1. The first one is the dual constraint that guarantees that the support vectors
            are shifted parallel to the decision boundary;
            2. The second one is the constraint that guarantees that the norm of each delta
            vector is equal to delta0.
        """
        K_d = self.kernel_fun(
            self.svs[svs_to_opt] + delta, self.svs, **self.kernel
        )
        K = K[svs_to_opt]
        aK = self.alphas @ (K_d - K).T
        res = aK.dot(aK)

        return (
            res,
            ((torch.norm(delta, dim=1) - self.delta0.to(self.device)) ** 2).sum()
        )

    def collect_delta_and_shas(self):
        """
        Return the perturbed support vectors each one with its corresponding delta vector
        and their sha as identifier.
        """
        return [(self.svs_shas[index], sv) for index, sv in enumerate(self.svs)]


# ==================== Kernel Functions Mixin ====================

class FedSVMClientKernel:
    """
    Mixin class that provides kernel functions (RBF, polynomial, linear).
    """

    def rbf_kernel(self, X, Y, **params):
        """Radial Basis Function kernel"""
        return torch.exp(-params["gamma"] * torch.cdist(X, Y) ** 2)

    def poly_kernel(self, X, Y, **params):
        """Polynomial kernel"""
        return (X @ Y.T + 1.0).pow_(params["degree"])

    def linear_kernel(self, X, Y, **_):
        """Linear kernel"""
        return X @ Y.T

    def select_kernel_fun(self):
        """Select kernel function based on configuration"""
        print(f"[INIT] [Client {self.client_no}] Using kernel: {self.kernel}")
        if self.kernel["kernel"] == "rbf":
            return self.rbf_kernel
        elif self.kernel["kernel"] == "poly":
            return self.poly_kernel
        elif self.kernel["kernel"] == "linear" or self.kernel["kernel"] == "rff":
            return self.linear_kernel
        else:
            raise ValueError(f"Kernel not supported: {self.kernel['kernel']}")


# ==================== Base FedSVM Client ====================

class FedSVMClient:
    """
    Base class to define the common behavior of a client in the FedSVM algorithm,
    regardless of the type of delta used and the optimization used to find it.
    """

    def __init__(
        self, client_no, X, y, rff, client_eps, C, kernel, device, only_load=False
    ) -> None:
        if only_load:
            return

        self.client_no = client_no
        self.device = device
        self.X = X
        self.y = y
        self.C = C
        self.svs = None
        self.svs_shas = None
        self.svs_labels = None
        self.alphas = None
        self.delta0 = None
        self.kernel = kernel
        if kernel["kernel"] == "rff":
            self.kernel["kernel"] = "linear"

        # Handle rff transformation if provided
        if rff is not None:
            transformed = rff.transform(self.X)
            self.Z_ = torch.tensor(np.ascontiguousarray(transformed))
        else:
            self.Z_ = torch.tensor(np.ascontiguousarray(self.X))

        self.known_shas = self.init_known_shas(self.Z_, self.y)
        self.sent = set()
        self.svc = self.local_svm(self.Z_, self.y)
        self.client_eps = client_eps
        self.latest_delta = None
        self.kernel_fun = self.linear_kernel
        self.n_round = 0

    def linear_kernel(self, X, Y, **_):
        """Default linear kernel"""
        return torch.mm(X, Y.T)

    def save_data(self):
        """A client can be saved to disk using joblib, only its svc is needed."""
        return self.svc

    def load_data(self, data):
        """If a client has been saved to disk, it can be loaded using joblib."""
        self.svc = data

    def predict(self, X, rff):
        """
        Predict the labels of the examples X, using the random fourier features transformer rff,
        and the svc learned.
        """
        if rff is not None:
            Z = torch.tensor(rff.transform(X))
        else:
            Z = torch.tensor(X)
        return self.svc.predict(Z)

    def local_svm(self, X, y):
        """
        Define the local SVM of the client, using the examples X and the labels y.
        When the rff kernel is used, the X are the random fourier features of the original examples.
        """
        print(f"[CONFIG] [Client {self.client_no}] Fitting SVM on {X.shape[0]} samples, {X.shape[1]} features")

        svc = SVC(C=self.C, **self.kernel)
        svc = svc.fit(X, y)

        self.svs = torch.tensor(
            svc.support_vectors_, dtype=torch.float32, device=self.device
        )

        print(f"   [Client {self.client_no}] # of known_shas: {len(self.known_shas)}")

        shas = list(self.known_shas.keys())
        self.svs_shas = [shas[index] for index in svc.support_]

        self.alphas = torch.tensor(
            svc.dual_coef_[0], dtype=torch.float32, device=self.device
        )
        self.svs_labels = y[svc.support_]

        return svc

    def Z(self):
        """Return the random fourier features of the examples known by the client."""
        return torch.cat(
            [sv.cpu().reshape(1, -1) for sv, _ in self.known_shas.values()], dim=0
        )

    def labels(self):
        """Return the labels of the examples known by the client."""
        return torch.tensor([label for _, label in self.known_shas.values()])

    def init_known_shas(self, X, y):
        """
        Initialize the dictionary of examples known by the client.
        The key is the sha256 hash of the example, the value is a tuple containing
        the example itself and its label. We use the sha256 hash to identify the
        examples so that its dataset does not explode in size.
        """
        known_shas = OrderedDict()
        for i in range(len(X)):
            array = np.ascontiguousarray(X[i].numpy())
            known_shas[sha256(array).hexdigest()] = (X[i], y[i])
        return known_shas

    def update_known_shas(self, svs_clients, y_clients):
        """
        Only the new examples received from the server are added to the known examples.
        New examples are identified by their unique sha256 hash.
        """
        print(
            f"[SEARCH] [Client {self.client_no}] Filtering received examples. "
            f"Received: {len([ex for client in svs_clients for ex in client])}, "
            f"Known: {len(self.known_shas)}"
        )

        for ind_client, client in enumerate(svs_clients):
            for ind_sv, (sv_key, sv) in enumerate(client):
                if sv_key not in self.known_shas:
                    self.known_shas[sv_key] = (sv, y_clients[ind_client][ind_sv])
                    self.sent.add(sv_key)

        print(f"   [Client {self.client_no}] Number of examples after filtering: {len(self.known_shas)}")

    def receive_svs(self, svs_clients, labels_clients):
        """
        The client receives the support vectors of the other clients from the server
        and updates its known examples. After that, it trains its local SVM using the new examples.
        """
        self.update_known_shas(svs_clients, labels_clients)
        self.svc = self.local_svm(self.Z(), self.labels())

    def send_svs(self):
        """
        The client sends its support vectors to the server but before that it computes the delta
        perturbation to apply to the support vectors.
        """
        self.n_round += 1

        svs_and_shas = self.collect_delta_and_shas()

        # Collect into to_send only the support vectors that have not been sent yet
        to_send = []
        to_send_idx = []
        for index, (sv_key, sv) in enumerate(svs_and_shas):
            if sv_key not in self.sent:
                to_send.append((sv_key, sv))
                to_send_idx.append(index)

        # Sample only a fraction of the support vectors to send
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        perc = sigmoid(self.n_round - 3)
        sampled_idx = np.random.choice(
            range(len(to_send_idx)),
            size=int(np.ceil(len(to_send_idx) * perc)),
            replace=False,
        )

        # Filter the support vectors to send based on the sampled indices
        to_send_idx = [to_send_idx[idx] for idx in sampled_idx]
        to_send = [to_send[idx] for idx in sampled_idx]

        print(
            f"[SEND] [Client {self.client_no}] Sampled {int(100 * perc)}% of support vectors: "
            f"{len(to_send_idx)} new deltas needed"
        )

        # Compute the delta perturbation to apply to the support vectors
        self.latest_delta = self.fit(self.device, to_send_idx)

        delta_norm = torch.mean(
            torch.norm(self.latest_delta, dim=len(self.latest_delta.shape) - 1)
        )
        print(f"   [Client {self.client_no}] Avg delta norm: {delta_norm:.2f}")

        # Apply the delta perturbation to the support vectors, store them among the sent ones
        # and send them to the server
        self.sent = self.sent.union(set([sv_key for sv_key, _ in to_send]))
        to_send_delta = []
        for index, (sv_key, sv) in enumerate(to_send):
            indexed_delta = (
                self.latest_delta[index].detach()
                if len(self.latest_delta.shape) > 1
                else self.latest_delta.detach()
            )
            to_send_delta.append((sv_key, sv.cpu() + indexed_delta.cpu()))

        return to_send_delta, self.svs_labels[to_send_idx]


# ==================== FedSVMClientOptMD Implementation ====================

class FedSVMClientOptMD(
    FedSVMClient, FedSVM_Optim, FedSVM_MultipleDeltas, FedSVMClientKernel
):
    """
    FedSVM client that uses optimized multiple deltas with kernel support.

    This is the recommended FedSVM implementation that provides:
    - Multiple delta perturbations (one per support vector)
    - Optimized delta computation for better accuracy
    - Support for RBF and polynomial kernels
    - Privacy preservation through delta perturbations
    """

    def __init__(self, client_no, X, y, rff, client_eps, C, kernel, device):
        super().__init__(client_no, X, y, rff, client_eps, C, kernel, device)
        print(f"[OK] [Client {self.client_no}] Using multiple deltas with optimization")

        # Set kernel function based on configuration
        self.kernel_fun = self.select_kernel_fun()


# ==================== Evaluation Functions ====================

def evaluate_fedsvm(y_true, y_pred):
    """
    Evaluate the predictions y_pred against the true labels y_true.

    Returns:
        Dictionary with accuracy, precision, recall, f1 scores
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
