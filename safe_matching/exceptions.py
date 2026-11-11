"""Gerarchia di eccezioni di safe-matching-framework."""

from __future__ import annotations

__all__ = [
    "SafeMatchingError",
    "ConfigError",
    "DatasetLoadError",
    "PreprocessingError",
    "MatchingError",
    "UnsupportedOptionError",
    "ReportError",
    "MissingDependencyError",
]


class SafeMatchingError(Exception):
    """Radice di tutti gli errori della libreria."""


class ConfigError(SafeMatchingError):
    """Configurazione (dati o matching) mancante, non parsabile o incoerente."""


class DatasetLoadError(SafeMatchingError):
    """Il file dati non puo' essere letto."""


class PreprocessingError(SafeMatchingError):
    """Errore durante l'applicazione della DataConfig ai dati."""


class MatchingError(SafeMatchingError):
    """Errore durante il matching (dati insufficienti, vincoli infattibili, ...)."""


class UnsupportedOptionError(ConfigError):
    """Opzione valida come nome ma non (ancora) implementata."""


class ReportError(SafeMatchingError):
    """Errore nella generazione del report PDF."""


class MissingDependencyError(SafeMatchingError):
    """Una libreria SAFE richiesta (es. safe-data-processing) non e' installata nel venv."""
