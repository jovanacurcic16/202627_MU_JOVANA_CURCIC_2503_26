import numpy as np
import mpmath as mp

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class PolynomialRegression(RegressorMixin, BaseEstimator):

    def __init__(self, degree=2):
        self.degree = degree

    def _create_matrix(self, X):
        X = np.asarray(X, dtype=float).reshape(-1)
        return np.vander(X, N=self.degree + 1, increasing=True)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if not isinstance(self.degree, (int, np.integer)) or self.degree < 0:
            raise ValueError("Stepen polinoma mora biti nenegativan ceo broj.")

        if len(X) != len(y) or len(X) == 0:
            raise ValueError("X i y moraju imati isti broj merenja.")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("Podaci moraju biti konacni brojevi.")

        # Racunanje sa 100 decimalnih cifara preciznosti
        with mp.workdps(100):

            x_mp = [mp.mpf(str(x)) for x in X]
            y_mp = mp.matrix([mp.mpf(str(v)) for v in y])

            # Formiranje regresione matrice
            Phi = mp.matrix([
                [x ** j for j in range(self.degree + 1)]
                for x in x_mp
            ])

            # Resavanje problema najmanjih kvadrata
            # koristeci QR dekompoziciju
            w, residual = mp.qr_solve(Phi, y_mp)

            # Cuvanje preciznih parametara
            self._coef_mp = [mp.mpf(v) for v in w]

            # Parametri za prikaz i analizu
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

                # Hornerov postupak za racunanje polinoma
                vrednost = mp.mpf("0")

                for w in reversed(self._coef_mp):
                    vrednost = vrednost * x_mp + w

                predikcije.append(float(vrednost))

        return np.array(predikcije)

    def score(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)

        return np.mean((y - y_pred) ** 2)
