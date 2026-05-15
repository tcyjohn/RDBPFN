from .base import *
from .numeric import *
from .category import *
from .wrapper import *
from .composite import *
from .datetime import *
from .key_mapping import *
from .fill_timestamp import *
from .dummy_table import *
from .canonicalize import *
from .filter_column import *

_LAZY_LOADED = False

def __getattr__(name: str):
    global _LAZY_LOADED
    _TEXT_NAMES = {"DPRTextEmbeddingTransformConfig", "DPRTextEmbeddingTransform",
                   "GloveTextEmbeddingTransformConfig", "GloveTextEmbeddingTransform",
                   "_run_one_device", "_run_one_proc"}
    if name in _TEXT_NAMES:
        if not _LAZY_LOADED:
            import importlib
            for mod_name in (".text_dpr", ".text_glove"):
                mod = importlib.import_module(mod_name, __package__)
                globals().update({k: v for k, v in mod.__dict__.items() if not k.startswith("_") or k in _TEXT_NAMES})
            _LAZY_LOADED = True
        return globals().get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
