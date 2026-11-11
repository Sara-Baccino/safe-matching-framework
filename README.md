# safe-matching-framework

Libreria SAFE per il **matching di popolazioni**. Si esegue solo con `python main.py` (nessuna CLI): tutti i
parametri sono variabili in `main.py` o campi dei JSON di configurazione.

Dipende da **safe-data-processing** (gia' nel venv), che gestisce lettura dei dati, mappatura, casting, imputazione e
conversione in uscita (mapping inverso e nomi originali). Matching e diagnostiche sono implementati qui; architettura
derivata da `synapse-matching` (Sara-Baccino/Synapse).

## Installazione con uv

Serve Python >= 3.12 (come `safe_validation`) e accesso ai repository GitHub `train-ai` (credenziali git o SSH configurate).

```bash
uv sync --extra dev                     # crea .venv, installa safe-data-processing da git (branch dev) e le dipendenze
uv run pytest                           # test (i test end-to-end partono se safe_data_processing e' importabile)
uv run python main.py                   # esegue la pipeline

uv sync --extra dev --extra validation  # opzionale: installa anche safe_validation (tira dentro torch, xgboost, lifelines...)
```

Note:
* `[tool.uv.sources]` vale solo per il progetto radice: le sorgenti git dichiarate in `safe_validation` **non** sono ereditate, per
  questo `safe-data-processing` e' dichiarato anche in questo `pyproject.toml`. Se cambi branch/tag, modificalo li'.
* Per lavorare su una copia locale di una libreria SAFE: `uv add --editable ../safe-data-processing-framework`
  (oppure `uv pip install -e ../safe-data-processing-framework` nel venv).
* Le versioni di polars, pydantic, scikit-learn, scipy e pyarrow sono allineate a `safe_validation` per evitare conflitti di risoluzione.
* `uv sync` genera `uv.lock`: versionalo per avere ambienti riproducibili.

## Uso

1. Modifica le variabili in `main.py` (percorsi dataset, config dati, config matching, colonna gruppo, output).
2. `python main.py`

| Variabile | Significato |
|---|---|
| `DATASET_PATHS` | 1 file (trattati + controlli) oppure 2 file (il primo = trattati, con almeno una colonna attiva in comune dopo `new_name`) |
| `DATA_CONFIG_PATHS` | JSON di config dati (struttura safe-data-processing), uno per dataset; con 2 dataset ne basta uno |
| `GROUP_COLUMN`, `TREATED_VALUE` | solo con 1 dataset: colonna del file originale che identifica il gruppo e valore dei trattati (None se 0/1). Non serve che sia attiva in config |
| `TREATED_DATASET_INDEX` | solo con 2 dataset: quale dei due e' il gruppo trattato |
| `MATCHING_CONFIG` | percorso JSON **oppure** dict Python |
| `OUTPUT_DIR`, `OUTPUT_NAME`, `OUTPUT_FORMATS` | dove e come salvare (`csv`, `xlsx`, `parquet`) |
| `OUTPUT_VALUES` | `postprocessed` (default) o `raw` (vedi sotto) |

## Output

* `<nome>_dataset.*` - righe abbinate con **colonne originali** + metadati:
  `match_role` (`query` = unita' di partenza, `match` = controparte in coppia, `selected` = unita' selezionata da OT),
  `match_pair_id`, `match_weight`, `match_distance`, `match_stratum`, `match_group` (1 trattato / 0 controllo),
  `match_source`, `source_row` (posizione nel file originale), `match_id`, `match_uid`, `match_ps` (se propensity score).
  * `postprocessed`: colonne attive con nome/valori originali tramite `Postprocessor` (mapping inverso; **i valori imputati
    restano imputati**), piu' le colonne originali non in config (ripristinate dal file).
  * `raw`: le righe del file originale cosi' come sono, valori mancanti inclusi.
* `<nome>_unmatched.*` - unita' query senza abbinamento.
* `<nome>_report.pdf` - **valutazione della riuscita**: criteri pass/fail, riepilogo, configurazione, copertura per strato,
  bilanciamento (SMD, VR, KS, chi2, JS; Love plot, overlap PS, distanze) e le analisi multivariate:
  * **PCA** - proiezione prima/dopo sugli stessi assi (PCA stimata sulla popolazione eleggibile), varianza spiegata, |SMD| dei punteggi
    per componente, loadings;
  * **Distribuzioni** - griglie prima/dopo (numeriche: istogrammi pesati; categoriche: proporzioni) per le variabili piu' sbilanciate;
  * **Correlazioni** - matrici per trattati e controlli prima/dopo e loro differenza, coppie con maggiore scarto, correlazione di ogni
    variabile con l'appartenenza al gruppo.
* `<nome>_metrics.json`, `<nome>_balance_table.csv`, `<nome>_pca_variance.csv`, `<nome>_pca_loadings.csv`,
  `<nome>_correlation_pairs.csv`, `<nome>_correlation_with_group.csv`, `<nome>_strata.csv`, `<nome>_run_config.json`, `figures/`.

## Se `uv run pytest` fallisce con `KeyError: 'string'`

Il `type` delle colonne nella data config deve essere uno di quelli accettati dalla tua versione di safe-data-processing (nei config reali:
`int`, `float`, `str`; mappature in `categorical_mappings`, `missing_data_management.missing_values`). `configs/demo_data_config.json` e'
scritto in quel formato. L'errore ora e' riportato con un messaggio che elenca i tipi usati nel file.

## Config matching

```json
{
  "population":  {"direction": "treated_to_control"},
  "variables":   {"matching": ["Age","BMI"], "stratification": ["Male"], "evaluation": ["Diabetes"]},
  "constraints": {"calipers": [{"variable": "Age", "value": 5, "scale": "absolute"}],
                  "global_caliper": {"value": 0.3, "scale": "absolute"}},
  "metrics":     {"name": "mahalanobis", "params": {}},
  "method":      {"name": "hungarian", "params": {}},
  "validation":  {"smd_threshold": 0.1,
                  "success_criteria": {"min_match_rate": 0.8, "max_mean_abs_smd": 0.1, "max_abs_smd": 0.25,
                                       "min_pct_balanced": 90, "require_improvement": true},
                  "balance_metrics": ["smd","variance_ratio","ks","chi_square","jensen_shannon"],
                  "analysis": {"pca": true, "pca_variance_target": 0.8, "distributions": true,
                               "correlations": true, "correlation_method": "pearson"},
                  "max_distribution_plots": 12, "plots": true, "hooks": []},
  "options":     {"drop_rows_with_missing": true, "random_state": 42}
}
```

Le analisi multivariate usano matching + stratificazione + `evaluation`: aggiungi in `variables.evaluation` tutte le variabili
che vuoi vedere in PCA, distribuzioni e correlazioni. `success_criteria` accetta anche `max_pca_smd` e `max_corr_diff`
(spenti di default).

Le variabili in `variables`/`constraints` usano i **`new_name`** della data config. Ogni soglia di `success_criteria`
puo' essere `null` per disattivare il criterio.

Riferimento completo di tutti i parametri: **`docs/config_reference.md`**; template con tutti i default: **`configs/matching_full_template.json`**
(+ JSON Schema `configs/matching_config.schema.json`).

**Metodi** (`method.name`): `propensity_score`, `greedy_nn`, `hungarian`, `sorted`, `partial_ot`.

| Metodo | Parametri | Output |
|---|---|---|
| `propensity_score` | `model` (`logistic`/`random_forest`/`xgboost`/`gradient_boosting`), `model_params`, `cross_fit`, `caliper_sd` (0.2 su logit), `trim`, `algorithm` (+`algorithm_params`) | coppie |
| `greedy_nn` | `ratio_k`, `replacement`, `order`, `prefilter_k` | coppie 1:K |
| `hungarian` | `prefilter_k` | coppie 1:1 a costo globale minimo |
| `sorted` | `sort_variable` | coppie 1:1 ottime su UNA variabile (DP) |
| `partial_ot` | `target_size` o `target_ratio`, `nb_dummies`, `prefilter_k` | **selezione** pesata del pool |

**Distanze**: `euclidean`, `manhattan`, `mahalanobis`, `hybrid`, `gower`. **Vincoli**: stratificazione (match esatto),
calipers per variabile, caliper globale sulla distanza.

## Stima degli effetti (ATT / ATC / ATE) e overlap del PS

Con `estimation.enabled` la pipeline stima l'effetto del trattamento sugli outcome indicati (continui o binari) con errori standard e
intervalli da **bootstrap** (default: ricampionamento delle unita' con riesecuzione di PS e matching a ogni replica), e li confronta con
la differenza non aggiustata. Il report ha la sezione "Treatment effect estimation" (tabella, forest plot, distribuzioni bootstrap) e
`<nome>_effects.csv`. Gli outcome non entrano nel matching e, se la data config li imputa, i valori imputati valgono come osservati:
per gli outcome usa `strategy: maintain`.

Il report mostra anche il **supporto comune del propensity score** (istogrammi specchiati prima/dopo, AUC, quota di unita' fuori supporto,
coefficiente di overlap), calcolato con qualunque metodo (con metodi non-PS il PS e' stimato solo per la diagnostica).

## Validazione esterna (altre librerie SAFE)

`validation.hooks`: callable `"pacchetto.modulo:funzione"` che riceve `before`, `after`, `group_col`, `variables`, `kinds`;
i risultati entrano nel PDF. Esempio: `safe_matching/integrations/example_hook.py`.

## Limiti noti di safe-data-processing (verificati leggendo il codice)

* Le strategie `drop` e `replace` di `Preprocessor.preprocess_dataset` non funzionano: `drop` cerca la colonna `row_index`
  ma la crea come `row_idx` e filtra `data` invece del dataset preprocessato; `replace` modifica `data`, mai restituito.
  Usa `impute` o `maintain`. Il matching richiede comunque che il preprocessing non elimini righe.
* Le colonne attive con `strategy: impute` devono essere numeriche dopo mapping/cast (la knn imputation le usa tutte): una
  colonna `id` o testuale deve avere `active: false`.
* Il percorso del modulo `Postprocessor` non e' noto: la ricerca automatica prova `safe_data_processing.postprocessing[.postprocessing]`
  e `safe_data_processing.preprocessing.postprocessing`; altrimenti imposta `POSTPROCESSOR` in `main.py`.
