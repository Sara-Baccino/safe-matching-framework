"""Esegue l'intera pipeline di matching:  python main.py

1. legge dataset + config dati con safe-data-processing (lettura, mappatura, tipi, imputazione)
2. legge/costruisce la config di matching
3. esegue il matching
4. salva il dataset matchato (colonne ORIGINALI + metadati match_*), le metriche e il report PDF

Tutti i parametri si impostano qui sotto (o nei file JSON di configurazione). Nessuna CLI.
"""

import logging
from pathlib import Path

from safe_matching import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# ------------------------------------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------------------------------------
BASE_PATH = Path("C:/Users/sbaccino/Desktop/IVI RMA")

# 1 dataset (con trattati e controlli)  ->  [f"{BASE_PATH}/all.csv"]   (serve GROUP_COLUMN)
# 2 dataset (il primo = trattati)       ->  [f"{BASE_PATH}/treated.csv", f"{BASE_PATH}/controls.csv"]
DATASET_PATHS = [BASE_PATH /  "PGTy_imp.xlsx", BASE_PATH / "PGTn_imp.xlsx"]

# Un JSON di config dati (stessa struttura SAFE) per dataset; con 2 dataset ne basta uno, riusato per entrambi.
DATA_CONFIG_PATHS = [BASE_PATH / "config/PGTn_config_gp.json"]

# Solo con 1 dataset: colonna (nome ORIGINALE nel file) che identifica il gruppo, e valore dei trattati
# (TREATED_VALUE puo' restare None se la colonna e' 0/1).
GROUP_COLUMN: str | None = None
TREATED_VALUE = None

# Solo con 2 dataset: quale dei due e' il gruppo trattato (0 = primo, 1 = secondo).
TREATED_DATASET_INDEX = 0

# Config del matching: percorso di un JSON oppure un dict Python (stessa struttura, vedi README).
MATCHING_CONFIG = "configs/matching_hungarian.json"
# MATCHING_CONFIG = {
#     "variables": {"matching": ["Age", "BMI", "ALT"], "stratification": ["Male"], "evaluation": ["Diabetes"]},
#     "constraints": {"calipers": [{"variable": "Age", "value": 5}]},
#     "metrics": {"name": "mahalanobis"},
#     "method": {"name": "hungarian", "params": {}},
#     "validation": {"smd_threshold": 0.1},
# }

# ------------------------------------------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
OUTPUT_NAME = "matching"                  # prefisso dei file: matching_dataset.csv, matching_report.pdf, ...
OUTPUT_FORMATS = ("csv", "xlsx")          # csv, xlsx, parquet
# "postprocessed": colonne attive con nomi/valori ORIGINALI (Postprocessor di safe-data-processing; i valori
#                  imputati restano) + colonne originali non in config, prese dal file;
# "raw":           righe del file originale cosi' come sono (missing inclusi).
OUTPUT_VALUES = "postprocessed"
RESTORE_INACTIVE_COLUMNS = True
GENERATE_PDF = True

# Lettura: opzioni per file (Excel: {"sheet_name": "Foglio1"}; CSV: {"separator": ";"} - altrimenti rilevato).
READ_OPTIONS: dict | None = None
# Valori non presenti in 'mappings': "error" (default, elenca i valori), "missing" (diventano mancanti) oppure "union"
# (le nuove etichette delle variabili NOMINALI sono aggiunte ai mappings; le ordinali restano un errore).
# ON_CAST_ERROR (valori non convertibili nel 'type'): "error" oppure "missing".
ON_UNMAPPED = "error"
ON_CAST_ERROR = "error"


if __name__ == "__main__":
    result = run_pipeline(
        datasets=DATASET_PATHS,
        data_configs=DATA_CONFIG_PATHS,
        matching_config=MATCHING_CONFIG,
        group_column=GROUP_COLUMN,
        treated_value=TREATED_VALUE,
        treated_dataset_index=TREATED_DATASET_INDEX,
        output_dir=OUTPUT_DIR,
        output_name=OUTPUT_NAME,
        output_values=OUTPUT_VALUES,
        output_formats=OUTPUT_FORMATS,
        restore_inactive_columns=RESTORE_INACTIVE_COLUMNS,
        generate_pdf=GENERATE_PDF,
        read_options=READ_OPTIONS,
        on_unmapped=ON_UNMAPPED,
        on_cast_error=ON_CAST_ERROR,
    )

    m, b, s = result.outcome.metrics, result.balance.summary, result.success
    print(f"\nMatching '{result.outcome.method}': {m['n_query_matched']}/{m['n_query_total']} unita' abbinate "
          f"({100 * m['match_rate']:.1f}%)")
    if b.get("n_variables"):
        print(f"|SMD| medio {b['mean_abs_smd_before']:.3f} -> {b['mean_abs_smd_after']:.3f} | "
              f"bilanciate {b['pct_balanced_before']:.0f}% -> {b['pct_balanced_after']:.0f}%")
    print(f"Riuscita: {'SI' if s.passed else 'NO'} ({s.n_passed}/{s.n_checks} criteri)")
    for note in result.outcome.notes:
        print(f"  nota: {note}")
    print(f"File in {result.output_dir.resolve()}:")
    for name in result.files:
        print(f"  - {name}")
