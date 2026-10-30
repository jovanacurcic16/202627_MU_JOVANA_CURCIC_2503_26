import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from poly_regression import PolynomialRegression


folder = "Rezultati/Regularizacija"
os.makedirs(folder, exist_ok=True)

# Ucitavanje podataka
podaci = pd.read_csv("podaci.csv")
X = podaci["X"].to_numpy()
y = podaci["Y"].to_numpy()

# Razliciti modeli i regularizacije
stepeni = [1, 2, 3, 5, 9]
lambde = [0, 1, 1e6, 1e12]

# Nove tacke za predikciju
X_novo = np.array([600, 900, 1300, 1600])

greske = []
predikcije = []
parametri = []
modeli = {}

for M in stepeni:
    for lam in lambde:

        model = PolynomialRegression(
            degree=M,
            regularization=lam
        )

        model.fit(X, y)
        modeli[(M, lam)] = model

        mse = model.score(X, y)
        err = mse / 2
        y_novo = model.predict(X_novo)

        # Regularizovana funkcija cilja
        kazna = lam * np.sum(model.coef_[1:] ** 2) / (2 * len(y))
        J = err + kazna

        print("\n" + "=" * 45)
        print(f"M = {M}, lambda = {lam:g}")
        print(f"MSE = {mse:.8f}")
        print(f"Err(w) = {err:.8f}")
        print(f"J(w) = {J:.8f}")

        greske.append({
            "M": M,
            "lambda": lam,
            "Err(w)": err,
            "MSE": mse,
            "J(w)": J
        })

        for j, w in enumerate(model.coef_):
            parametri.append({
                "M": M,
                "lambda": lam,
                "Parametar": f"w{j}",
                "Vrednost": w
            })

        for x, pred in zip(X_novo, y_novo):
            predikcije.append({
                "M": M,
                "lambda": lam,
                "Padavine": x,
                "Predvidjena biomasa": pred
            })

# Cuvanje rezultata
pd.DataFrame(greske).to_csv(
    f"{folder}/greske.csv", index=False
)

pd.DataFrame(predikcije).to_csv(
    f"{folder}/predikcije.csv", index=False
)

pd.DataFrame(parametri).to_csv(
    f"{folder}/parametri.csv", index=False
)

# Crtanje grafikona
for M in [2, 5, 9]:
    plt.figure(figsize=(10, 6))

    plt.scatter(
        X, y, color="black",
        label="Merenja", zorder=5
    )

    x_grafik = np.linspace(520, 1600, 300)

    for lam in lambde:
        y_grafik = modeli[(M, lam)].predict(x_grafik)

        plt.plot(
            x_grafik, y_grafik,
            label=f"lambda = {lam:g}"
        )

    plt.title(f"Ridge regresija - M = {M}")
    plt.xlabel("Padavine (mm)")
    plt.ylabel("Biomasa (t/ha)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{folder}/ridge_M{M}.png",
        dpi=300
    )
    plt.close()

print("\nPOREDJENJE MODELA:")
print(pd.DataFrame(greske).to_string(index=False))

print("\nSvi rezultati su sacuvani!")
