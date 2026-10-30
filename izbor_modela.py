import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from poly_regression import PolynomialRegression

# Ucitavanje podataka
podaci = pd.read_csv("podaci.csv")

X = podaci["X"].to_numpy()
y = podaci["Y"].to_numpy()

# Stepeni polinoma koje ispitujemo
stepeni = [1, 2, 3, 5, 9]

# Nove vrednosti padavina
X_novo = np.array([600, 900, 1300, 1600])

# Tabela rezultata
rezultati = []
predikcije = pd.DataFrame({"Padavine": X_novo})

# Grafikoni
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()

for indeks, M in enumerate(stepeni):

    # Kreiranje i obucavanje modela
    model = PolynomialRegression(degree=M)
    model.fit(X, y)

    # Predikcije na skupu za obuku
    y_pred = model.predict(X)

    # Racunanje gresaka
    mse = model.score(X, y)
    err = mse / 2

    # Predikcije za nove vrednosti
    y_novo = model.predict(X_novo)

    # Prikaz rezultata
    print("\n" + "=" * 45)
    print(f"POLINOM STEPENA M = {M}")
    print("=" * 45)

    print("\nParametri modela:")
    for j, w in enumerate(model.coef_):
        print(f"w{j} = {w:.12g}")

    print(f"\nErr(w) = {err:.8f}")
    print(f"MSE = {mse:.8f}")

    print("\nPredikcije za nove tacke:")
    for x, pred in zip(X_novo, y_novo):
        print(f"X = {x} mm, Y = {pred:.6f} t/ha")

    # Cuvanje rezultata
    rezultati.append({
        "Stepen M": M,
        "Err(w)": err,
        "MSE": mse
    })

    predikcije[f"M={M}"] = y_novo

    # Crtanje regresione krive
    x_grafik = np.linspace(500, 1650, 300)
    y_grafik = model.predict(x_grafik)

    ax = axes[indeks]

    ax.scatter(X, y, label="Merenja")
    ax.plot(x_grafik, y_grafik, label=f"M={M}")

    ax.set_title(f"Polinom stepena {M}")
    ax.set_xlabel("Padavine (mm)")
    ax.set_ylabel("Biomasa (t/ha)")
    ax.grid(True)
    ax.legend()

# Uklanjanje praznog grafika
fig.delaxes(axes[-1])

plt.tight_layout()

# Cuvanje grafikona
plt.savefig(
    "Rezultati/poredjenje_modela.png",
    dpi=300
)

plt.show()

# Cuvanje tabela
df_rezultati = pd.DataFrame(rezultati)

df_rezultati.to_csv(
    "Rezultati/greske_modela.csv",
    index=False
)

predikcije.to_csv(
    "Rezultati/predikcije.csv",
    index=False
)

print("\n" + "=" * 45)
print("POREDJENJE MODELA")
print("=" * 45)
print(df_rezultati.to_string(index=False))

print("\nRezultati su sacuvani u folderu Rezultati.")

