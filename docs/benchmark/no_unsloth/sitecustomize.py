"""sitecustomize: block `import unsloth` for the whole process.

Placed on PYTHONPATH for the soup_hf_pure arm only.

Why this exists: Soup's `commands/train.py:1517` calls `is_unsloth_available()`
even when `backend: transformers`, and that function does `import unsloth`.
Unsloth patches the whole HF/trl stack at import time. So a Soup
"transformers" run on a machine where unsloth is installed is silently
unsloth-backed, and cannot serve as a stock-HF baseline.

sitecustomize is imported by CPython during interpreter startup, so the block
is in place before any user code -- including Soup's availability probe --
executes. Asserting at the end that unsloth really is unimportable.
"""
import sys


class _BlockUnsloth:
    def find_spec(self, name, path=None, target=None):
        if name == "unsloth" or name.startswith("unsloth."):
            raise ImportError(
                "unsloth blocked by bench harness (sitecustomize): this arm "
                "must run without unsloth's import-time patching"
            )
        return None


sys.meta_path.insert(0, _BlockUnsloth())

if "unsloth" in sys.modules:  # pragma: no cover
    raise SystemExit("HARNESS BROKEN: unsloth already imported before sitecustomize")