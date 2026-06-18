from .tools.logger import root_logger

try:
    from importlib.metadata import version
    __version__ = version("qumet")
except Exception:
    __version__ = "0.1.0"
