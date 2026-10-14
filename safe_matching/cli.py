"""Interfaccia a riga di comando (usata da main.py e dallo script ``safe-matching``)."""

from __future__ import annotations

import argparse
import logging
import sys

from safe_matching.exceptions import SafeMatchingError
from safe_matching.pipeline import run_pipeline
from safe_matching.version import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="safe-matching",
        description="Matching di popolazioni: dataset -> config -> matching -> report PDF.",
        epilog=("Esempi:\n"
                "  1 dataset : python main.py -d all.csv -c data.json -m matching.json --group-col Treated\n"
                "  2 dataset : python main.py -d treated.csv -d controls.csv -c data_t.json -c data_c.json -m matching.json"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-d", "--dataset", action="append", required=True, help="File dati (1 o 2 volte). Con 2, il primo e' il gruppo trattato.")
    p.add_argument("-c", "--data-config", action="append", required=True, help="JSON di configurazione dati (uno per dataset; con 2 dataset ne basta uno).")
    p.add_argument("-m", "--matching-config", required=True, help="JSON di configurazione del matching.")
    p.add_argument("-g", "--group-col", help="Colonna del gruppo (obbligatoria con 1 solo dataset).")
    p.add_argument("--treated-value", help="Valore della colonna gruppo che identifica i trattati (default: 1 se il gruppo e' 0/1).")
    p.add_argument("--treated-dataset", type=int, default=0, choices=[0, 1], help="Con 2 dataset: indice del dataset trattato (default 0).")
    p.add_argument("-o", "--output-dir", default="outputs")
    p.add_argument("--lenient-columns", action="store_true", help="Ignora (con avviso) le colonne attive della config assenti dal dataset.")
    p.add_argument("--no-pdf", action="store_true", help="Non generare il report PDF.")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        res = run_pipeline(
            args.dataset, args.data_config, args.matching_config, group_col=args.group_col,
            treated_value=args.treated_value, treated_dataset=args.treated_dataset, output_dir=args.output_dir,
            lenient_columns=args.lenient_columns, generate_pdf=not args.no_pdf,
        )
    except SafeMatchingError as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 2
    m, s = res.outcome.metrics, res.balance.summary
    print(f"Matching '{res.outcome.method}' completato: {m['n_query_matched']}/{m['n_query_total']} query abbinate "
          f"({100 * m['match_rate']:.1f}%).")
    if s.get("n_variables"):
        print(f"|SMD| medio {s['mean_abs_smd_before']:.3f} -> {s['mean_abs_smd_after']:.3f}; "
              f"bilanciate {s['pct_balanced_before']:.0f}% -> {s['pct_balanced_after']:.0f}%.")
    for n in res.outcome.notes:
        print(f"  nota: {n}")
    print(f"Output in: {res.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
