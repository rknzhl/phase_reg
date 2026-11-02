import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist


def prepare_time_series(path, centred=True):
    """Read accelerometer CSV and return magnitude and uniform time grid."""
    path = Path(path)
    if not path.is_file():
        path = Path(__file__).parent / path
    data = pd.read_csv(path, sep=';', decimal=',')
    time = data['time'].to_numpy(dtype=float)
    frequency = len(time) / (time[-1] - time[0])
    if not 490 < frequency < 510:
        raise ValueError(f'Expected 500 Hz, got {frequency:.2f} Hz')
    axes = data[['X_value', 'Y_value', 'Z_value']].to_numpy(dtype=float)
    signal = np.linalg.norm(axes, axis=1)
    if centred:
        signal = signal - signal.mean()
    return signal, np.linspace(0, time[-1] - time[0], len(signal))


def HankelMatrix(signal, L):
    """Return all overlapping windows as rows."""
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if not 1 <= L <= len(signal):
        raise ValueError('Window length must lie between 1 and signal length')
    return np.lib.stride_tricks.sliding_window_view(signal, L).copy()


class NWregression:
    """Nadaraya–Watson regression with optional scalar phase distance."""

    def __init__(self, p=2, h=0.03, kernel='gaussian', metric=None):
        if h <= 0:
            raise ValueError('Bandwidth h must be positive')
        if kernel not in {'gaussian', 'rectangular', 'triangular', 'quadratic', 'quartic'}:
            raise ValueError(f'Unknown kernel: {kernel}')
        self.p = p
        self.h = h
        self.kernel = kernel
        self.metric = metric

    def fit(self, X, Y):
        self.X = np.asarray(X, dtype=float).reshape(len(X), -1)
        self.Y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
        if len(self.X) != len(self.Y) or len(self.X) == 0:
            raise ValueError('X and Y must contain the same nonzero number of rows')
        return self

    def _weights(self, r):
        if self.kernel == 'gaussian':
            return np.exp(-2 * r**2)
        if self.kernel == 'rectangular':
            return (r < 1).astype(float)
        if self.kernel == 'triangular':
            return np.maximum(1 - r, 0)
        if self.kernel == 'quadratic':
            return np.maximum(1 - r**2, 0)
        return np.maximum(1 - r**2, 0)**2

    def predict(self, X):
        X = np.asarray(X, dtype=float).reshape(len(X), -1)
        result = []
        for start in range(0, len(X), 256):
            query = X[start:start + 256]
            if self.metric is None:
                distances = cdist(query, self.X, metric='minkowski', p=self.p)
            elif query.shape[1] == self.X.shape[1] == 1:
                distances = self.metric(query[:, 0, None], self.X[None, :, 0])
            else:
                distances = cdist(query, self.X, metric=self.metric)
            weights = self._weights(distances / self.h)
            total = weights.sum(axis=1, keepdims=True)
            if np.any(total == 0):
                raise ValueError('No training points within bandwidth; increase h')
            result.append(weights @ self.Y / total)
        return np.vstack(result)


def integ(xs,ys):
    Y = []
    X = []
    temp_y = 0.0
    temp_x = 0.0
    for i in range(len(xs)-1):
        temp_x = (xs[i+1]+xs[i])/2
        X.append(temp_x)

        temp_y += (xs[i+1]-xs[i])*(ys[i+1]+ys[i])/2
        Y.append(temp_y)
    return np.array(X),np.array(Y)

def diff(x,y):
    h = x[1]-x[0]
    y0 = (-3*y[0]+4*y[1]-y[2])/(2*h)
    y1 = (y[2:]-y[:-2])/(2*h)
    y2 = (y[-3]-4*y[-2]+3*y[-1])/(2*h)
    return np.concatenate(([y0],y1,[y2]))

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def hankel_matrix(array, l, each = 1):
    return HankelMatrix(np.asarray(array)[::each], l)

def eigenvalues_plot(__x,n,each = 1,num_l = None, MinMax = False):
    HM = hankel_matrix(__x, n,each)

    S = np.linalg.svd(HM,compute_uv=False)
    if MinMax:
        S /= max(S)

    plt.figure(figsize=(11, 6))

    plt.plot(S[:num_l]**.5,'o')
    plt.plot(S[:num_l]**.5,'--')

    #plt.xlim(-1,20)


    plt.xlabel('Компонента', size=23)
    plt.ylabel('√σ', size=23)

    plt.tick_params(axis='both', which='major', labelsize=25,length=8, width=4)
    plt.grid(which='major',
        color = 'gray',
        linewidth = 0.8)
    plt.minorticks_on()
    plt.grid(which='minor',
        color = 'gray',
        linestyle = '--',
        linewidth = 0.5)
    plt.show()

