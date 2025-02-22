from sympy import IndexedBase, Sum, symbols


def get_symbolic_loss() -> tuple:
    y_true = IndexedBase("y_true")
    y_pred = IndexedBase("y_pred")
    weight = IndexedBase("w")
    i, n = symbols("i n", integer=True)
    r_square_eq = 1.0 - Sum(weight[i] * (y_true[i] - y_pred[i]) ** 2, (i, 0, n)) / Sum(
        weight[i] * (y_true[i]) ** 2, (i, 0, n)
    )
    grad_eq = r_square_eq.diff(y_pred[i])
    hess_eq = r_square_eq.diff(y_pred[i]).diff(y_pred[i])
    return r_square_eq, grad_eq, hess_eq


def r_square_loss(y_pred, y_true, weight):
    """Modify R^2 Score s.t. it ranges [0, +inf) instead of (-inf, 1]."""
    nominator = y_true - y_pred
    r_square = 1.0 - (weight * nominator * nominator).sum() / (weight * y_true * y_true).sum()
    return -(r_square - 1)
