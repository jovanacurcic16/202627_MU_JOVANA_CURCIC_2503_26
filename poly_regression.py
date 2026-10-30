import numpy as np
import mpmath as mp

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class PolynomialRegression(RegressorMixin, BaseEstimator):

    def __init__(self, degree=2, regularization=0.0):
        self.degree = degree
        self.regularization = regularization

    def _create_matrix(self, X):
        X = np.asarray(X, dtype=float).reshape(-1)
        return np.vander(X, N=self.degree + 1, increasing=True)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if not isinstance(self.degree, (int, np.integer)) or self.degree < 0:
            raise ValueError("degree mora biti nenegativan ceo broj.")

        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization mora biti nenegativan broj.")

        if len(X) != len(y) or len(X) == 0:
            raise ValueError("X i y moraju imati isti broj merenja.")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("Podaci moraju biti konacni brojevi.")

        with mp.workdps(100):
            x_mp = [mp.mpf(str(x)) for x in X]
            y_mp = [mp.mpf(str(v)) for v in y]

            redovi = [
                [x ** j for j in range(self.degree + 1)]
                for x in x_mp
            ]

            lam = mp.mpf(str(self.regularization))

            if lam > 0:
                koren = mp.sqrt(lam)

                for j in range(1, self.degree + 1):
                    red = [mp.mpf("0")] * (self.degree + 1)
                    red[j] = koren
                    redovi.append(red)
                    y_mp.append(mp.mpf("0"))

            Phi = mp.matrix(redovi)
            cilj = mp.matrix(y_mp)

            koeficijenti, ostatak = mp.qr_solve(Phi, cilj)

            self._coef_mp = [mp.mpf(v) for v in koeficijenti]
            self.coef_ = np.array(
                [float(v) for v in self._coef_mp]
            )

        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")

        X = np.asarray(X, dtype=float).reshape(-1)
        predikcije = []

        with mp.workdps(100):
            for x in X:
                x_mp = mp.mpf(str(x))
                vrednost = mp.mpf("0")

                for w in reversed(self._coef_mp):
                    vrednost = vrednost * x_mp + w

                predikcije.append(float(vrednost))

        return np.array(predikcije)

    def score(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)

        if len(y) != len(y_pred):
            raise ValueError("X i y moraju imati isti broj merenja.")

        return float(np.mean((y - y_pred) ** 2))
