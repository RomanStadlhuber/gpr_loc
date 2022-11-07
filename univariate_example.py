from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as snl
import numpy as np
import pandas as pd

"""NOTE:
see the following links for additional information
- free GP Book: http://gaussianprocess.org/gpml/chapters/RW.pdf
- online lecture of the introduction to gaussian processes:
  https://www.youtube.com/watch?v=4vGiHC35j9s&t=2150s&ab_channel=NandodeFreitas
- more example code, ipynb notebook and additional notes:
  https://github.com/jwangjie/Gaussian-Processes-Regression-Tutorial
"""


# the squared exponential function:
"""
             || a - b ||^2
            --------------
                  l^2
k(a,b) = s * exp
"""
SIGMA = 0.5
LAMBDA = 1.0


def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The squared exponential kernel function"""
    squared_distance = (
        np.reshape(np.sum(a**2, axis=1), (-1, 1))  # S(a^2)
        + np.sum(b**2, 1)  # + S(b^2)
        - 2 * np.dot(a, b.T)  # - 2 * <a, b>
    )
    return SIGMA * np.exp(-1 / (LAMBDA**2) * squared_distance)


def dataset(n: int, sigma_noise: float) -> Tuple[np.ndarray, np.ndarray]:
    """return the training dataset as `D=(X,y)`"""
    # input value range
    xs = np.linspace(-np.pi, np.pi, n * 4).reshape(
        -1, 1
    )  # [x1, x2, ..., x_n] row vector
    # noisy sinusodial signal
    rng = np.random.default_rng()
    xs = rng.choice(xs, n, replace=False)
    ys = np.sin(xs) + rng.normal(loc=0, scale=sigma_noise**2)  # [y1, y2, ...]^(T)
    # return as dataset tuple D=(X, y)
    return (xs, ys)


def plot(xs: np.ndarray, ys: np.ndarray) -> None:
    """Plot a line representing a function evaluation

    If multiple y values are provided for the same x values, these are captured
    as "variances" in the plot.

    Supply mean, upper and lower bounds in the following form

    ```python
    # xs -> input values
    # mus -> output mean value vector
    # vars -> outupt covariance diagonal elements
    plot(
        xs = np.repeat(xs, 3),
        ys = np.hstack((mus, mus + 2 * vars, mus - 2 * vars))
    )
    ```

    To achieve a plot containing mean value and confidence interval.
    """
    df = pd.DataFrame({"x": xs, "f(x)": ys})
    snl.relplot(data=df, x="x", y="f(x)", kind="line")
    plt.show()


def regression(
    x: np.ndarray, Xs: np.ndarray, K_xx_inv=np.ndarray, ys=np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """perfrom regression on an input and return the resulting distribution

    For n-dimensional input vectors, the function will return a distribution
    mean vector of shape `(n)` and a covariance matrix of shape `(n,n)`
    """
    # shape regression input as a vector
    xr = np.reshape([x], (-1, 1))
    # k(x*, X) - R(100,1)
    k_xrx = kernel(xr, Xs)
    # k(x*, x*) - R(1,1)
    k_xrxr = kernel(xr, xr)
    # regression mean value
    fx_mu = k_xrx @ K_xx_inv @ ys
    # regression variance value
    fx_var = k_xrxr - k_xrx @ K_xx_inv @ k_xrx.T
    # return as (mu, s)
    return (fx_mu.reshape(-1), fx_var)


if __name__ == "__main__":
    n = 50
    s = 0.1
    Xs, ys = dataset(n, s)
    # plot the original dataset
    # plot_data(Xs, ys)
    # construct and invert the noisy data covariance matrix
    # K[X, X] - R(100 x 100)
    K_xx = kernel(Xs, Xs) + s * np.identity(n)
    # [K[X,X] + s I]^(-1)
    # inverting is easier and more stable using the colesky decomposition
    L_k = np.linalg.cholesky(K_xx)
    L_k_inv = np.linalg.inv(L_k)
    # since K = L @ L.T ==> K^-1 = L.T^-1 @ L^-1
    K_xx_inv = L_k_inv.T @ L_k_inv

    xrs = np.linspace(-np.pi, np.pi, n * 4)
    (mus, cov) = regression(xrs, Xs=Xs, K_xx_inv=K_xx_inv, ys=ys)

    vars = 2 * np.diagonal(cov)

    plot(xs=np.repeat(xrs, 3), ys=np.hstack((mus, mus + vars, mus - vars)))
