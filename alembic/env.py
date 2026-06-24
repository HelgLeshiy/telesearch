"""Alembic migration environment for the telesearch service DB.

The URL is read from ``TELESEARCH_*`` settings (same source as the app), so
migrations target whatever database the service is configured to use. Batch mode
is enabled for SQLite-compatible ALTERs.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from telesearch.server import models  # noqa: F401  (register tables on metadata)
from telesearch.server.config import get_server_settings
from telesearch.server.db import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    return get_server_settings().resolved_database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_url(), future=True)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
