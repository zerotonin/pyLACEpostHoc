# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — run_scripts.run_spike2Scripts                   ║
# ║  « build the c-start database »                                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Runs the multi-trace folder importer over the c-start raw data  ║
# ║  to populate the c-start fish database.                          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Populate the c-start fish database from the raw recording folders."""
from __future__ import annotations

import config
from fish_data_base.fishDataBase import FishDataBase


def main() -> None:
    """Import every rei and sufge1 c-start recording into the database."""
    database_path = config.get_path("database_path")
    data_root = config.get_path("data_root")
    db = FishDataBase(database_path, database_path / "fishDataBase_cstart.csv")

    for tag in ("rei", "sufge1"):
        folder = data_root / "cstart_experiments" / tag
        # Experiment tags: CCur counter-current, Ta tapped, Unt untapped, cst c-start.
        db.run_multi_trace_folder(folder, tag, "cst", "08-2019", start_at=0, gui_correction=False)


if __name__ == "__main__":
    main()
