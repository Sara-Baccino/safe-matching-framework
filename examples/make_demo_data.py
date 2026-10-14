"""Genera dati simulati (nomi colonna grezzi in italiano, come nei JSON SAFE)."""
from pathlib import Path
import numpy as np, polars as pl

out = Path(__file__).parent / "data"; out.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

def cohort(n, shift, name):
    eta = rng.normal(55 + shift, 10, n).round()
    male = (rng.random(n) < 0.55 + 0.05 * (shift > 0)).astype(int)
    bmi = rng.normal(26 + 0.1 * shift, 4, n)
    diab = (rng.random(n) < 0.15 + 0.01 * shift).astype(int)
    alt = np.exp(rng.normal(3.6 + 0.01 * shift, 0.5, n))
    df = pl.DataFrame({"ID": [f"{name}{i}" for i in range(n)], "Eta": eta, "Male": male, "BMI": bmi.round(1),
                       "Diabete": diab, "ALT": alt.round(1)})
    return df.with_columns(pl.when(pl.int_range(0, n) % 23 == 0).then(None).otherwise(pl.col("BMI")).alias("BMI"))

treated, controls = cohort(250, 6, "T"), cohort(900, 0, "C")
treated.write_csv(out / "treated.csv"); controls.write_csv(out / "controls.csv")
pl.concat([treated.with_columns(pl.lit(1).alias("Gruppo")), controls.with_columns(pl.lit(0).alias("Gruppo"))]).write_csv(out / "all.csv")
print("dati scritti in", out)
