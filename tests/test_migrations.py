"""Alembic migration smoke test: a fresh DB upgrades to head with all tables."""

import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

REPO = Path(__file__).resolve().parents[1]


def test_alembic_upgrade_head(tmp_path, monkeypatch):
    db = tmp_path / "m.db"
    monkeypatch.setenv("TELESEARCH_DATABASE_URL", f"sqlite:///{db}")

    import telesearch.server.config as scfg

    scfg.get_server_settings.cache_clear()

    cfg = Config(str(REPO / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO / "alembic"))
    command.upgrade(cfg, "head")

    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    con.close()
    expected = {
        "users", "workspaces", "memberships", "sources", "jobs", "shares",
        "saved_searches", "graph_snapshots", "audit_log", "alembic_version",
    }
    assert expected <= tables
    scfg.get_server_settings.cache_clear()
