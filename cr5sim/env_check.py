"""Fail early with a clear message when an app is started without the project's virtualenv."""

import importlib.util
import sys
from pathlib import Path

REQUIRED = ("jax", "jaxlie", "jaxls", "pyroki", "viser", "yourdfpy", "mujoco", "tyro")


def require_venv() -> None:
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if missing:
        venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
        script = Path(sys.argv[0]).name
        sys.exit(
            f"Missing packages: {', '.join(missing)}.\n"
            f"This interpreter ({sys.executable}) is not the project's environment. Run:\n"
            f"    {venv_python} {script}"
        )
