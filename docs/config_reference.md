# Riferimento del config di matching

Un file JSON con queste sezioni. **`configs/matching_full_template.json`** contiene TUTTI i parametri con i default (le chiavi che
iniziano con `_` e `$schema` sono commenti e vengono ignorate); `configs/matching_config.schema.json` e' lo JSON Schema per
l'autocompletamento nell'editor. Entrambi si rigenerano con `PYTHONPATH=. python examples/make_configs.py`.
Le variabili si indicano con il **`new_name`** della data config.

## `population`
| Campo | Default | Note |
|---|---|---|
| `direction` | `treated_to_control` | `control_to_treated` inverte i ruoli (query = controlli) |

## `variables`
| Campo | Note |
|---|---|
| `matching` | variabili che definiscono la distanza / il PS (obbligatorie) |
| `stratification` | match esatto: il matching avviene dentro ogni strato |
| `evaluation` | non usate per il matching, ma incluse in bilanciamento, PCA, distribuzioni, correlazioni |

## `constraints`
| Campo | Default | Note |
|---|---|---|
| `calipers[]` | `[]` | `{variable, value, scale: absolute \| standard_deviation}`; per categoriche = uguaglianza |
| `global_caliper` | `null` | `{value, scale}`: distanza massima di una coppia; con `standard_deviation` = value x SD della feature (1-D) o delle distanze |

## `metrics` (distanza; ignorata da `propensity_score` e `sorted`, che sono 1-D)
| `name` | `params` |
|---|---|
| `euclidean`, `manhattan` | `scale: standard \| minmax \| none`, `weights: {variabile: peso}` |
| `mahalanobis` | `covariance: total \| within`, `regularization` |
| `hybrid` | `weight_numerical` (0.6), `weight_categorical` (0.4), `weights` |
| `gower` | `weights` (pesi dei blocchi proporzionali al numero di variabili) |

## `method` = `{name, params}`
| `name` | Parametri (default) |
|---|---|
| `greedy_nn` | `ratio_k` (1), `replacement` (false), `order` (`given`/`random`/`hardest_first`), `prefilter_k` |
| `hungarian` | `prefilter_k` |
| `sorted` | `sort_variable` (obbligatoria se i matching variables sono piu' di uno) |
| `partial_ot` | `target_size` **o** `target_ratio` (1.0), `nb_dummies` (10), `max_iterations`, `prefilter_k` |
| `propensity_score` | vedi sotto |

### `propensity_score`
| Parametro | Default | Note |
|---|---|---|
| `model` | `logistic` | `logistic`, `random_forest`, `xgboost`, `gradient_boosting` |
| `model_params` | `{}` | iperparametri del modello, sovrascrivono i default (es. `{"n_estimators": 500, "max_depth": 3}`) |
| `poly_degree`, `C`, `max_iter` | 1, 0.5, 5000 | solo `logistic` |
| `cross_fit` | `null` (auto) | PS fuori-campione K-fold; automatico per i modelli ad alberi/boosting (evita PS ~0/1 per overfitting) |
| `cross_fit_folds` | 5 | |
| `include_stratification` | true | le variabili di stratificazione entrano nel modello PS |
| `match_on` | `logit` | `logit` o `ps` |
| `caliper_sd` | 0.2 | caliper in SD del logit(PS) (Austin); `null` = nessuno |
| `trim` / `trim_quantile` | `none` / 0.02 | trimming del supporto comune: `minmax` o `quantile` |
| `algorithm` / `algorithm_params` | `greedy_nn` / `{}` | abbinamento sul PS: `greedy_nn`, `hungarian`, `sorted`, `partial_ot` |

Default dei modelli: random forest 500 alberi, `min_samples_leaf` 10; XGBoost 300 alberi, `max_depth` 3, `learning_rate` 0.05,
`subsample`/`colsample_bytree` 0.8, `min_child_weight` 5.

## `validation`
| Campo | Default | Note |
|---|---|---|
| `smd_threshold` | 0.1 | soglia |SMD| di "bilanciato" |
| `balance_metrics` | smd, variance_ratio, ks, chi_square, jensen_shannon | anche `wasserstein` |
| `success_criteria` | `min_match_rate` 0.8, `max_mean_abs_smd` 0.1, `max_abs_smd` 0.25, `min_pct_balanced` 90, `require_improvement` true, `max_pca_smd` null, `max_corr_diff` null | `null` disattiva il criterio |
| `analysis` | `pca`, `pca_variance_target` 0.8, `pca_loadings_components` 3, `distributions`, `correlations`, `correlation_method` (`pearson`/`spearman`), `max_correlation_variables` 25, `max_scatter_points` 3000, `ps_support`, `ps_support_bins` 30 | `ps_support` calcola il supporto comune del PS anche con metodi non-PS (PS logistico solo diagnostico) |
| `max_distribution_plots` | 12 | variabili (le piu' sbilanciate) nelle griglie di distribuzioni |
| `plots` | true | `false` = solo tabelle |
| `hooks[]` | `[]` | `{name, callable: "modulo:funzione", params, enabled}` validazione esterna |

## `estimation` (ATT / ATC / ATE con bootstrap)
| Campo | Default | Note |
|---|---|---|
| `enabled` | false | |
| `estimands` | `["ATT"]` | `ATT`, `ATC`, `ATE` |
| `outcomes[]` | `[]` | `{variable, type: continuous \| binary, label}`; binary (0/1) aggiunge il risk ratio. L'outcome non puo' essere variabile di matching/stratificazione |
| `include_naive` | true | differenza non aggiustata come riferimento |
| `bootstrap.n_boot` | 200 | |
| `bootstrap.ci_level` | 0.95 | |
| `bootstrap.resample` | `full_pipeline` | `full_pipeline`: ricampiona le unita' e RIESEGUE PS + matching a ogni replica; `matched_pairs`: ricampiona le coppie (veloce, solo metodi a coppie) |
| `bootstrap.random_state` | `null` | `null` = `options.random_state` |
| `bootstrap.n_jobs` | 1 | -1 = tutti i core |

**Copertura**: ATC/ATE richiedono che il matching copra (quasi) tutto il gruppo di riferimento. Con metodi senza reinserimento e
gruppi sbilanciati la copertura e' bassa (hungarian da' lo stesso abbinamento nelle due direzioni): usa `greedy_nn` con
`replacement: true`. La tabella degli effetti riporta la copertura. Il bootstrap con matching nearest-neighbour non e' formalmente
valido (Abadie & Imbens, 2008): leggi SE e IC come indicativi.

## `options`
| Campo | Default | Note |
|---|---|---|
| `drop_rows_with_missing` | true | esclude (e riporta) le righe con mancanti nelle variabili usate |
| `random_state` | 42 | |
