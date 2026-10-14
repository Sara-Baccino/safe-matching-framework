"""Entry point: esegue l'intera pipeline di matching.

    python main.py -d dataset.csv -c data_config.json -m matching_config.json --group-col Group

Vedi ``python main.py --help``. Per l'uso da codice: ``safe_matching.run_pipeline``.
"""

from safe_matching.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
