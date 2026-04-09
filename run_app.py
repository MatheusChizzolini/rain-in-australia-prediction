from __future__ import annotations

import os
import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    app_path = project_dir / "app.py"

    os.chdir(project_dir)
    sys.argv = ["streamlit", "run", str(app_path)]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
