import pandas as pd

from poly_regression import PolynomialRegression


# 1. Ucitavanje podataka
podaci = pd.read_csv("podaci.csv")

# 2. Razdvajanje ulazne i izlazne promenljive
X = podaci["X"].to_numpy()
y = podaci["Y"].to_numpy()

# 3. Kreiranje modela drugog stepena
model = PolynomialRegression(degree=2)

# 4. Obucavanje modela
model.fit(X, y)

# 5. Predvidjanje biomase
y_pred = model.predict(X)

# 6. Racunanje srednje kvadratne greske
mse = model.score(X, y)

# 7. Prikazivanje rezultata
print("POLINOMIJALNA REGRESIJA")
print("-----------------------------")

print("Stepen polinoma:", model.degree)

print("\nParametri modela:")
for i, w in enumerate(model.coef_):
    print(f"w{i} = {w:.10f}")

print("\nPoredjenje stvarnih i predvidjenih vrednosti:")

rezultati = pd.DataFrame({
    "Padavine": X,
    "Stvarna biomasa": y,
    "Predvidjena biomasa": y_pred
})

print(rezultati.to_string(index=False))

print("\nSrednja kvadratna greska (MSE):")
print(f"{mse:.6f}")

print("\nFunkcija greske Err(w):")
print(f"{mse / 2:.6f}")
