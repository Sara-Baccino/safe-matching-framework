import numpy as np
import polars as pl
import pytest

KINDS = {"Age": "numerical", "BMI": "numerical", "Male": "categorical", "Stage": "categorical"}


@pytest.fixture(scope="session")
def population():
    rng = np.random.default_rng(3)
    n = 700
    age, bmi = rng.normal(55, 10, n), rng.normal(26, 4, n)
    male, stage = rng.integers(0, 2, n), rng.integers(0, 3, n)
    g = (rng.random(n) < 1 / (1 + np.exp(-(-4 + 0.06 * age + 0.4 * male)))).astype(np.int8)
    return pl.DataFrame({"__uid__": np.arange(n), "__group__": g, "Age": age, "BMI": bmi, "Male": male, "Stage": stage})


@pytest.fixture(scope="session")
def kinds():
    return dict(KINDS)
