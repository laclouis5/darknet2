import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression


class BivariateFunction:
    def __init__(self, fn):
        """
        Callable object representing a function of 2 variables:
        func: (x, y) -> (x2, y2)

        Args:
            - func: lamda or callable object
        """
        self.fn = fn

    def __call__(self, x, y):
        return self.fn(x, y)


class Equation:
    """
    2D plane equation.
    """
    def __init__(self, coeffs):
        """
        Args:
            - coeffs [array(float)]: Plane coeeficients for equation coeffs[0] * x + coeffs[1] * y + coeffs[2]
        """
        self.coeffs = coeffs

    def __call__(self, x, y):
        return self.coeffs[0] * x + self.coeffs[1] * y + self.coeffs[2]

    def __repr__(self):
        return f"{self.coeffs[0]} * x + {self.coeffs[1]} * y + {self.coeffs[2]}"


def fit_plane(X, Y, Z, mask=None):
    """
    Fits a plane to observed data corrupted with noise:
        Z = a*X + b*Y + c + e

    Paramters:
        X, Y (array<float> of ndim=2):
            Arrays containing the coordinates of the point
            in the 2-D space.
        Z (array<float> of ndim=2):
            Observed values of the function.
        mask (optional array<bool> of ndim=2):
            Mask where to keep data.

    Returns:
        (array of size 3):
            The coefficients of the fitted plane.
            Z = a*X + b*Y + c + e  ->  [a, b, c]
    """
    if mask is not None:
        X = X[mask == True]
        Y = Y[mask == True]
        Z = Z[mask == True]

        T = np.stack((X, Y), axis=1)
        V = Z
    else:
        T = np.stack((X, Y), axis=2).reshape(-1, 2)
        V = Z.reshape(-1)

    reg = LinearRegression().fit(T, V)
    coeffs = [*reg.coef_, reg.intercept_]

    return Equation(coeffs)


def reg_score(y, f, mask=None):
    """
    Parameters:
        y(N-D array): the observed data
        f(N-D array): the predicted data

    Returns:
        (float): the R2 value
    """
    if mask is not None:
        y = y[mask == True]
        f = f[mask == True]

    y_mean = np.mean(y)
    SStot = np.sum(np.square(y - y_mean))
    SSres = np.sum(np.square(y - f))

    return 1 - SSres / SStot


if __name__ == "__main__":
    width, height = (100, 50)
    a, b, c = 0.002, 0.003, 25

    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    noise = 0.01 * np.random.randn(height, width)
    Z = a*X + b*Y + c + noise

    mask = np.full_like(Z, False)  # Simulate missing data
    mask[::4, ::4] = True

    # Fit the data
    eq = fit_plane(X, Y, Z, mask)

    # Predicted Plane
    H = eq(X, Y)

    print("Real Plane: Z = {}*X + {}*Y + {}".format(a, b, c))
    print("Predicted Plane: Z = {:.4}*X + {:.4}*Y + {:.4}".format(*eq.coeffs))
    print("R²: {:.2%}".format(reg_score(Z, H, mask)))

    # Plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(X, Y, Z, s=1)
    ax.plot_surface(X, Y, H, color="orange")
    plt.show()
