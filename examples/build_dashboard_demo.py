"""Rebuild ``docs/dashboard-demo.html``, the live demo linked from the README.

It is the ``dashboard.html`` that ``biosigkit demo`` writes (8 simulated subjects, 192 trials), so
the numbers on the page are the ones in the README. Run it after changing the dashboard::

    python examples/build_dashboard_demo.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from biosigkit.demo import run_demo


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "docs" / "dashboard-demo.html"
    with TemporaryDirectory() as tmp:
        run_demo(tmp, log=None)
        shutil.copyfile(Path(tmp) / "dashboard.html", target)
    print(f"wrote {target} ({target.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
