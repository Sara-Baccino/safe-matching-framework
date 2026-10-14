# safe-matching-framework

Libreria SAFE per il **matching di popolazioni**: carica 1 o 2 dataset, applica le config dati SAFE, esegue il
matching configurato da JSON e produce un **report PDF** con le diagnostiche di bilanciamento.

Architettura e parte dell'implementazione derivano da `synapse-matching` (Sara-Baccino/Synapse); la libreria e'
**autonoma** (nessuna dipendenza da Synapse) e ha un punto di aggancio per altre librerie SAFE (hook, vedi sotto).

## Uso rapido

```bash
uv sync            # oppure: pip install -e .
python examples/make_demo_data.py

# 1 dataset con colonna gruppo
python main.py -d examples/data/all.csv -c configs/demo_data_config.json \
               -m configs/matching_hungarian.json --group-col Gruppo -o outputs/run1

# 2 dataset (il primo = trattati, il secondo = controlli)
python main.py -d treated.csv -d controls.csv -c data_t.json -c data_c.json -m matching.json
```

Da codice: `from safe_matching import run_pipeline`.

## Input

| Cosa | Note |
|---|---|
| Dataset | `.csv .tsv .parquet .json .xlsx`. 1 file con trattati+controlli, oppure 2 file con almeno una colonna attiva in comune (dopo `new_name`). |
| Config dati (JSON) | Stessa struttura SAFE: chiave = nome originale, `new_name`, `active`, `id`, `categorical/numerical`, `mappings`, `type`, `missing_data_management`... Uno per dataset (con 2 dataset ne basta uno, riusato). I flag `gene/cytogenetic/clinical` diventano `semantic_roles`. |
| `--group-col` | Solo con 1 dataset: colonna che identifica il gruppo. `--treated-value` se non e' 0/1. |
| Config matching (JSON) | vedi sotto |

## Config matching

```json
{
  "population":  {"direction": "treated_to_control"},
  "variables":   {"matching": ["Age","BMI"], "stratification": ["Male"], "evaluation": ["Diabetes"]},
  "constraints": {"calipers": [{"variable": "Age", "value": 5, "scale": "absolute"}],
                  "global_caliper": {"value": 0.3, "scale": "absolute"}},
  "metrics":     {"name": "mahalanobis", "params": {}},
  "method":      {"name": "hungarian", "params": {}},
  "validation":  {"smd_threshold": 0.1, "balance_metrics": ["smd","ks","chi_square"], "plots": true, "hooks": []},
  "options":     {"drop_rows_with_missing": true, "random_state": 42}
}
```

**Metodi** (`method.name`): `propensity_score`, `greedy_nn`, `hungarian`, `sorted`, `partial_ot`
(alias: `ps`, `nn`, `optimal`, `ot`, ...). Altri si aggiungono con `@register_algorithm`.

| Metodo | Parametri principali | Output |
|---|---|---|
| `propensity_score` | `model` (logistic/random_forest/gradient_boosting), `poly_degree`, `caliper_sd` (default 0.2 su logit), `trim`, `algorithm` (greedy_nn/hungarian/sorted/partial_ot) + `algorithm_params` | coppie |
| `greedy_nn` | `ratio_k`, `replacement`, `order` (given/random/hardest_first), `prefilter_k` | coppie 1:K |
| `hungarian` | `prefilter_k` | coppie 1:1 a costo globale minimo |
| `sorted` | `sort_variable` | coppie 1:1 ottime su UNA variabile (DP) |
| `partial_ot` | `target_size` o `target_ratio`, `nb_dummies`, `prefilter_k` | **selezione** di unita' del pool pesate (non coppie) |

**Distanze** (`metrics.name`): `euclidean`, `manhattan` (variabili numeriche standardizzate, `params.scale`, `params.weights`),
`mahalanobis` (`params.covariance`: total/within), `hybrid` (L1 normalizzata sulle numeriche + Hamming sulle categoriche,
`weight_numerical`/`weight_categorical`), `gower`. Con `propensity_score` e `sorted` il matching e' 1-D e la metrica e' ignorata.

**Vincoli**: `stratification` = match esatto (il matching avviene dentro ogni strato); `calipers` = finestre hard per variabile
(`absolute` o `standard_deviation`; per categoriche = uguaglianza); `global_caliper` = distanza massima di una coppia.

## Output (`--output-dir`)

`matching_report.pdf`, `matched_dataset.csv/.parquet` (colonne `pair_id`, `weight`, `distance`, `match_role`, `stratum`),
`unmatched_query.csv`, `strata.csv`, `balance_table.csv`, `metrics.json`, `run_config.json`, `figures/*.png`.

## Integrazione con altre librerie SAFE

`validation.hooks` accetta callable `"pacchetto.modulo:funzione"` che ricevono `before`, `after`, `group_col`, `variables`,
`kinds` (+ `params`) e restituiscono dict o DataFrame: i risultati finiscono nel PDF. Un errore nella hook non ferma la pipeline.
Esempio: `safe_matching/integrations/example_hook.py`.

## Struttura

```
safe_matching/
  schemas/        ColumnInfo, DataConfig, MatchingConfig, parametri dei metodi
  preprocessing/  loader, preprocess (mappings, tipi, imputazione), assembly (1-2 dataset)
  matching/       engine, distances/, algorithms/, constraints/, representation/ (PS)
  evaluation/     balance (SMD, VR, KS, chi2, JS, Wasserstein), plots, hooks
  reporting/      pdf_report
  pipeline.py  cli.py
main.py
```

## Differenze rispetto a Synapse (da conoscere)
- Il tuo JSON SAFE non passa la validazione di `synapse-core`: la chiave deve coincidere con `new_name` (qui e' il nome originale),
  `ID` ha `id` e `categorical` insieme (rifiutato li'), e `Previous Event` compare due volte come `new_name`.
- In Synapse `matching_algorithm` accetta `optimal_transport_sinkhorn` ma il modulo registra `optimal_transport_selection`:
  la selezione OT non parte. Qui il nome e' unico (`partial_ot`) con alias per entrambi.
- Il SMD post-matching per OT in Synapse e' nullo (solo unita' pool nel dataset); qui il dataset include i query e il SMD e' pesato.
- Aggiunti: `manhattan`, `sorted`, matching su PS con algoritmo a scelta, calipers SD, `target_ratio`, 2 dataset.
