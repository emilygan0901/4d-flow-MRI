"""Record software versions into pipeline_config.yaml.

Run inside 3D Slicer's Python environment (from the repository root):
    Slicer --no-main-window --python-script validation/record_versions.py -- config/pipeline_config.yaml
or paste into the Slicer Python console (set CONFIG path below).
"""
import sys, platform, datetime
from importlib import metadata

from pathlib import Path
_DEFAULT = Path(__file__).resolve().parents[1] / "config" / "pipeline_config.yaml" if "__file__" in globals() \
    else Path("config") / "pipeline_config.yaml"
CONFIG = sys.argv[-1] if sys.argv and sys.argv[-1].endswith(".yaml") else str(_DEFAULT)

def pkg(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"

info = {
    "python": platform.python_version(),
    "os": platform.platform(),
}
try:
    import slicer
    info["slicer_version"] = slicer.app.applicationVersion
    info["slicer_revision"] = slicer.app.repositoryRevision
except ImportError:
    info["slicer_version"] = "not run inside Slicer"
    info["slicer_revision"] = "n/a"

info["slicerheart_4dflow"] = pkg("slicerheart_4dflow")

text = open(CONFIG, encoding="utf-8").read()
for key, val in info.items():
    # replace "  key: <anything>" on its own line
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}:") and line.startswith("  "):
            lines[i] = f'  {key}: "{val}"'
            break
    text = "\n".join(lines) + "\n"
open(CONFIG, "w", encoding="utf-8").write(text)

stamp = datetime.date.today().isoformat()
print(f"[{stamp}] versions written to {CONFIG}:")
for k, v in info.items():
    print(f"  {k}: {v}")
if info["slicerheart_4dflow"] not in ("0.5.2",):
    print("WARNING: slicerheart_4dflow is not 0.5.2 - phantom results were obtained with 0.5.2")
