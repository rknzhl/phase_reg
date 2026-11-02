import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd




class SSA(object):

    __supported_types = (pd.Series, np.ndarray, list)

    def __init__(self, tseries, L, save_mem=True, n_components=None):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.
        
        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list.
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.
        
        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """
        
        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")
        
        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N/2:
            raise ValueError("The window length must be in the interval [2, N/2].")
        
        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1
        
        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i:L+i] for i in range(0, self.K)]).T
        
        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = svd(self.X, full_matrices=False)
        tolerance = self.Sigma[0] * max(self.X.shape) * np.finfo(float).eps
        self.rank = int(np.count_nonzero(self.Sigma > tolerance))
        if n_components is not None and not 1 <= n_components <= min(self.X.shape):
            raise ValueError('Invalid number of SSA components')
        self.d = self.rank if n_components is None else min(n_components, self.rank)
        
        self.TS_comps = np.zeros((self.N, self.d))
        
        weights = np.convolve(np.ones(self.L), np.ones(self.K))
        for i in range(self.d):
            self.TS_comps[:, i] = (
                self.Sigma[i] * np.convolve(self.U[:, i], VT[i]) / weights
            )
        self.X_elem = None
        self.V = None
        if not save_mem:
            self.X_elem = np.array([
                self.Sigma[i] * np.outer(self.U[:, i], VT[i])
                for i in range(self.d)
            ])
            self.V = VT.T
        
        # Calculate the w-correlation matrix.
        self.calc_wcorr()
            
    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        
        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)
            

    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.
        
        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]
        
        ts_vals = self.TS_comps[:,indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)

    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """
             
        # Calculate the weights
        w = np.convolve(np.ones(self.L), np.ones(self.K))
        gram = self.TS_comps.T @ (w[:, None] * self.TS_comps)
        norms = np.sqrt(np.diag(gram))
        denominator = np.outer(norms, norms)
        self.Wcorr = np.divide(
            np.abs(gram), denominator,
            out=np.zeros_like(gram), where=denominator > 0,
        )
        np.fill_diagonal(self.Wcorr, 1)

    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d
        
        if self.Wcorr is None:
            self.calc_wcorr()
        
        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        colorbar = plt.colorbar(ax, fraction=0.045)
        colorbar.set_label("$W_{i,j}$")
        plt.clim(0,1)
        
        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d-1
        else:
            max_rnge = max
        
        plt.xlim(min-0.5, max_rnge+0.5)
        plt.ylim(max_rnge+0.5, min-0.5)
