"""
E2 – Forward Pass from Scratch
Introduction to Deep Learning, THWS

linear.py: implement all functions in this file.
Run the notebook cells to test your implementations using checks.py.
"""

import torch


# ==============================================================================
# Block 1 – Scalar linear function and scalar ReLU
# ==============================================================================

def linear_scalar(x, theta):
    """Scalar linear function: f(x) = theta_1 * x + theta_0.

    Args:
        x:     scalar input (Python float)
        theta: 1-D tensor of shape (2,) — (theta_0, theta_1)

    Returns:
        Scalar output theta_1 * x + theta_0.
    """
    pass


def relu_scalar(x):
    """Scalar ReLU: relu(x) = max(0, x).

    Args:
        x: scalar input (Python float)

    Returns:
        max(0, x)
    """
    pass


# ==============================================================================
# Block 2 – Shallow neural network (scalar input)
# ==============================================================================

def relu_unit(x, theta):
    """Single ReLU unit: sigma(theta_1 * x + theta_0).

    Args:
        x:     scalar input (Python float)
        theta: 1-D tensor of shape (2,) — (theta_0, theta_1)

    Returns:
        Scalar output.
    """
    pass


def shallow(x, theta_hidden, theta_out):
    """Shallow neural network with scalar input and scalar output.

    Computes: theta_out[0] + sum_j theta_out[j] * relu_unit(x, theta_hidden[j-1])

    Args:
        x:            scalar input (Python float)
        theta_hidden: list of k tensors of shape (2,), one per hidden unit
                      — each is (theta_j0, theta_j1)
        theta_out:    1-D tensor of shape (k+1,)
                      — theta_out[0] is bias, theta_out[j] is weight for unit j

    Returns:
        Scalar output.
    """
    pass


# ==============================================================================
# Block 3 – Vector input, scalar output
# ==============================================================================

def linear_vector(x, theta):
    """Linear function with vector input and scalar output.

    Computes: theta_1 · x + theta_0, where theta_1 = theta[1:].

    Args:
        x:     1-D tensor, shape (d,)
        theta: 1-D tensor, shape (d+1,) — (theta_0, theta_1_1, ..., theta_1_d)

    Returns:
        Scalar output (0-d tensor).
    """
    pass


def relu_tensor(x):
    """Element-wise ReLU for a tensor of any shape.

    Args:
        x: tensor

    Returns:
        Tensor of same shape with negative values zeroed out.
    """
    pass


def relu_unit_vector(x, theta):
    """Single ReLU unit with vector input: sigma(theta_1 · x + theta_0).

    Args:
        x:     1-D tensor, shape (d,)
        theta: 1-D tensor, shape (d+1,) — (theta_0, theta_1_1, ..., theta_1_d)

    Returns:
        Scalar output (0-d tensor).
    """
    pass


def shallow_vector(x, theta_hidden, theta_out):
    """Shallow network with vector input and scalar output.

    Computes: theta_out[0] + sum_j theta_out[j] * relu_unit_vector(x, theta_hidden[j-1])

    Args:
        x:            1-D tensor, shape (d,)
        theta_hidden: list of k tensors of shape (d+1,), one per hidden unit
        theta_out:    1-D tensor of shape (k+1,)

    Returns:
        Scalar output (0-d tensor).
    """
    pass


# ==============================================================================
# Block 4 – Batching
# ==============================================================================

def shallow_batch_loop(X, theta_hidden, theta_out):
    """Shallow network forward pass over a batch using a Python loop over samples.

    Args:
        X:            2-D tensor, shape (N, d)
        theta_hidden: list of k tensors of shape (d+1,), one per hidden unit
        theta_out:    1-D tensor of shape (k+1,)

    Returns:
        Output tensor, shape (N,)
    """
    pass


def linear_batch(X, theta):
    """Linear function over a batch of samples with scalar output per sample.

    Vectorised version of linear_vector over N samples.

    Args:
        X:     2-D tensor, shape (N, d)
        theta: 1-D tensor, shape (d+1,) — (theta_0, theta_1_1, ..., theta_1_d)

    Returns:
        Output tensor, shape (N,)
    """
    pass


def shallow_batch(X, theta_hidden, theta_out):
    """Shallow network forward pass over a batch, vectorised over samples.

    Uses linear_batch for each hidden unit — processes all N samples at once
    per unit, but still loops over hidden units.

    Args:
        X:            2-D tensor, shape (N, d)
        theta_hidden: list of k tensors of shape (d+1,), one per hidden unit
        theta_out:    1-D tensor of shape (k+1,)

    Returns:
        Output tensor, shape (N,)
    """
    pass


def linear_layer(X, Theta):
    """Linear layer mapping a batch of inputs to multiple outputs.

    Applies a linear transformation to all N samples simultaneously,
    producing k outputs per sample.

    Args:
        X:     2-D tensor, shape (N, d)
        Theta: 2-D tensor, shape (k, d+1) — each row is one output unit's parameters
               (theta_0, theta_1_1, ..., theta_1_d)

    Returns:
        Output tensor, shape (N, k)
    """
    pass


def shallow_batch_vectorised(X, Theta_hidden, theta_out):
    """Shallow network forward pass, fully vectorised over samples and hidden units.

    Uses linear_layer to compute all hidden activations at once.

    Args:
        X:             2-D tensor, shape (N, d)
        Theta_hidden:  2-D tensor, shape (k, d+1) — each row is one hidden unit's parameters
        theta_out:     1-D tensor of shape (k+1,)

    Returns:
        Output tensor, shape (N,)
    """
    pass


# ==============================================================================
# Block 6 – Two-layer MLP
# ==============================================================================

def mlp_batch(X, Theta_1, Theta_2, theta_out):
    """Two-layer MLP forward pass, fully vectorised.

    Args:
        X:         2-D tensor, shape (N, d)
        Theta_1:   2-D tensor, shape (k1, d+1)  — first hidden layer parameters
        Theta_2:   2-D tensor, shape (k2, k1+1) — second hidden layer parameters
        theta_out: 1-D tensor, shape (k2+1,)    — output layer parameters

    Returns:
        Output tensor, shape (N,)
    """
    pass
