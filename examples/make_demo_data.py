"""Dati simulati (nomi colonna grezzi in italiano). Effetto vero del trattamento: +2.0 su 'Outcome'."""
from pathlib import Path
import numpy as np, polars as pl

out = Path(__file__).parent / "data"; out.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

def cohort(n, shift, name, treated):
    eta = rng.normal(55 + shift, 10, n).round()
    male = (rng.random(n) < 0.55 + 0.05 * (shift > 0)).astype(int)
    bmi = rng.normal(26 + 0.1 * shift, 4, n)
    diab = (rng.random(n) < 0.15 + 0.01 * shift).astype(int)
    alt = np.exp(rng.normal(3.6 + 0.01 * shift, 0.5, n))
    outcome = 2.0 * treated + 0.08 * eta + 0.15 * bmi + 1.0 * male + rng.normal(0, 2, n)
    event = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 0.7 * treated + 0.02 * (eta - 55))))).astype(int)
    df = pl.DataFrame({"ID": [f"{name}{i}" for i in range(n)], "Eta": eta, "Sesso": male, "BMI": bmi.round(1),
                       "Diabete": diab, "ALT": alt.round(1), "Outcome": outcome.round(2), "Evento": event,
                       "Note": [f"nota {i}" for i in range(n)]})   # colonna NON in config: viene ripristinata in output
    return df.with_columns(pl.when(pl.int_range(0, n) % 23 == 0).then(None).otherwise(pl.col("BMI")).alias("BMI"))

treated, controls = cohort(250, 6, "T", 1), cohort(900, 0, "C", 0)
treated.write_csv(out / "treated.csv"); controls.write_csv(out / "controls.csv")
pl.concat([treated.with_columns(pl.lit(1).alias("Gruppo")), controls.with_columns(pl.lit(0).alias("Gruppo"))]).write_csv(out / "all.csv")
print("dati scritti in", out)
