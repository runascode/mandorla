"""Cross-experiment bridge to Exp 01's `src/` modules.

Exp 02 reuses Exp 01's FAISS index, box store, projection, encoder, and
retrieval primitives bit-for-bit. We need to *import* those primitives
from `experiments/exp1-vesica-rag/src/` without colliding with Exp 02's
own `src/` package.

The trick: register Exp 01's `src/` directory as a *separately named*
package (`exp01_src`) before any of its modules are imported. Their
relative imports (`from .regions import BoxExtent`) then resolve back
into `exp01_src.*` rather than Exp 02's `src.*`, avoiding the package
collision.

Callers should `import _exp01_bridge as bridge` and then use
`bridge.exp01_data`, `bridge.exp01_index_io`, etc.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

EXP01_SRC = Path(__file__).resolve().parents[2] / "exp1-vesica-rag" / "src"
if not EXP01_SRC.exists():
    raise RuntimeError(f"exp01 src/ not found at {EXP01_SRC}")

_PKG = "exp01_src"
if _PKG not in sys.modules:
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(EXP01_SRC)]
    pkg.__package__ = _PKG
    sys.modules[_PKG] = pkg


def load(modname: str):
    """Load `exp01_src.<modname>` and return it."""
    full = f"{_PKG}.{modname}"
    if full in sys.modules:
        return sys.modules[full]
    return importlib.import_module(full)


# Pre-load the modules Exp 02 actually uses, in dependency order so
# relative-import resolution can't surprise a caller.
exp01_regions = load("regions")
exp01_projection = load("projection")
exp01_box = load("box")
exp01_retrieve = load("retrieve")
exp01_data = load("data")
exp01_index_io = load("index_io")
