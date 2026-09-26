"""The workspace's one AI suite: AIService plus the shared provider/model menu.

Consumers import it from the sibling ai-suite checkout when present, else from the copy
vendored into their own repository (kept identical by ai-suite's sync.py). API keys load
from the .env next to this package; environment values always win.
"""

from .env import exit_on_ctrl_c, load_local_env

load_local_env()

from .providers import choose_ai, provider_config_path  # noqa: E402
from .service import AIService, IncompleteGenerationError, ProviderLimitReached  # noqa: E402

__all__ = ["AIService", "IncompleteGenerationError", "ProviderLimitReached", "choose_ai",
           "provider_config_path", "exit_on_ctrl_c", "load_local_env"]
