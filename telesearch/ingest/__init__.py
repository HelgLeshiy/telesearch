"""Ingestion of Telegram chat exports."""

from .telegram_parser import parse_export

__all__ = ["parse_export"]
