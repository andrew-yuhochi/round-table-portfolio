# apply_schema.py — Applies the canonical SQLite ledger schema to a database file.
#
# Usage:
#   python -m round_table_portfolio.storage.apply_schema [--db-path state/ledger.db]
#
# Idempotent: safe to re-run against an existing DB. Tables are created with
# CREATE TABLE IF NOT EXISTS; seed rows use INSERT OR IGNORE.
#
# IMPORTANT: sets PRAGMA foreign_keys = ON on every connection — SQLite does NOT
# enforce foreign keys by default and silently passes bad FK values without it.

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Location of the DDL file relative to this module.
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Default DB path relative to the project root (two levels up from src/).
_DEFAULT_DB_PATH = Path(__file__).parents[3] / "state" / "ledger.db"


def apply_schema(db_path: Path | None = None) -> Path:
    """Apply schema.sql to a SQLite database, creating it if absent.

    Parameters
    ----------
    db_path:
        Filesystem path for the SQLite file.  Defaults to ``state/ledger.db``
        relative to the project root.  The parent directory is created if it
        does not exist.

    Returns
    -------
    Path
        The resolved path of the database file that was written.

    Raises
    ------
    FileNotFoundError
        If ``schema.sql`` cannot be located next to this module.
    sqlite3.DatabaseError
        On any SQLite-level error during schema application.
    """
    resolved_db = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
    resolved_db = resolved_db.resolve()

    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"schema.sql not found at expected location: {_SCHEMA_PATH}"
        )

    # Ensure parent directory exists.
    resolved_db.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Applying schema to %s", resolved_db)

    sql = _SCHEMA_PATH.read_text(encoding="utf-8")

    conn = sqlite3.connect(str(resolved_db))
    try:
        # Mandatory: SQLite does NOT enforce foreign keys by default.
        conn.execute("PRAGMA foreign_keys = ON")

        # Verify the pragma took effect before running DDL.
        fk_status = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        if fk_status != 1:
            raise RuntimeError(
                "PRAGMA foreign_keys = ON was set but PRAGMA foreign_keys "
                f"returned {fk_status!r} — foreign-key enforcement is not active."
            )

        conn.executescript(sql)
        conn.commit()

        logger.info("Schema applied successfully (foreign_keys=%s)", fk_status)
    except Exception:
        logger.exception("Schema application failed for %s", resolved_db)
        conn.close()
        raise
    else:
        conn.close()

    return resolved_db


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a sqlite3 connection with foreign-key enforcement enabled.

    Callers are responsible for closing the connection (or using it as a
    context manager).

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Defaults to ``state/ledger.db``.
    """
    resolved_db = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
    resolved_db = resolved_db.resolve()

    conn = sqlite3.connect(str(resolved_db))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the round-table-portfolio SQLite schema."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path for the SQLite file (default: state/ledger.db).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    path = apply_schema(db_path=args.db_path)
    print(f"Schema applied to: {path}")
