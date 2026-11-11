# Changelog

## 0.1.0
- Prima versione: pipeline 1-2 dataset, config dati SAFE, config matching JSON, metodi propensity_score / greedy_nn /
  hungarian / sorted / partial_ot, distanze euclidean / manhattan / mahalanobis / hybrid (+ gower), calipers,
  stratificazione, diagnostiche di bilanciamento, report PDF.

## 0.2.0
- Lettura, mappatura, imputazione e conversione in uscita delegate a safe-data-processing.
- Rimossa la CLI: si esegue solo `python main.py` (variabili nel main o JSON).
- Output con colonne originali + metadati `match_*`; criteri di riuscita (pass/fail) nel report PDF.

## 0.3.0
- Report ampliato: PCA (prima/dopo, varianza, SMD per componente, loadings), griglie di distribuzioni, correlazioni
  (trattati/controlli, differenza, associazione col gruppo); nuove tabelle CSV e criteri di riuscita opzionali.
- pyproject per uv: Python >=3.12, `[tool.uv.sources]` per safe-data-processing e safe_validation (extra `validation`).

## 0.4.0
- Propensity score con XGBoost e random forest (iperparametri via `model_params`, cross-fitting automatico).
- Stima ATT/ATC/ATE con bootstrap (SE e IC), differenza non aggiustata, copertura; sezione dedicata nel report.
- Diagnostica di supporto comune/overlap del PS (grafico specchiato, AUC, % fuori supporto) con qualunque metodo.
- Config completo (`matching_full_template.json`), JSON Schema e `docs/config_reference.md`; chiavi `_commento` ammesse.
- Config demo nel formato reale di safe-data-processing (`type` int/float/str, `categorical_mappings`); coercizione numerica dopo il preprocessing.
- Note e avvisi del report in inglese.
