"""Fail early with a clear message or auto-switch when started without required packages."""

import importlib.util
import os
import sys
from pathlib import Path

REQUIRED = ("jax", "jaxlie", "jaxls", "pyroki", "viser", "yourdfpy", "mujoco", "tyro")


def require_venv() -> None:
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if missing:
        venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
        if (venv_python.exists() or os.path.islink(str(venv_python))) and sys.executable != str(venv_python):
            # Auto re-exec with the virtualenv Python
            try:
                os.execv(str(venv_python), [str(venv_python)] + sys.argv)
            except OSError:
                pass
        script = Path(sys.argv[0]).name
        sys.exit(
            f"Missing packages: {', '.join(missing)}.\n"
            f"This interpreter ({sys.executable}) does not have the required dependencies.\n\n"
            f"Run using the project's virtual environment:\n"
            f"    .venv/bin/python {script}\n"
            f"Or activate it first:\n"
            f"    source .venv/bin/activate && python {script}\n"
        )
