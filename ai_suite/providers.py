"""Provider and model menu shared by every AIService consumer.

provider_config_path() names the config file for a provider; choose_ai() is the
interactive provider/model picker (book writer, mathforge, music writer, ...)."""

from __future__ import annotations

import colorsys
import contextlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

from .service import (EFFORT_PROVIDERS, ensure_openai_oauth_proxy, load_opencode_go_sync,
                      background_process_options)


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

PROVIDER_CONFIG_MAP = {
    "google": str(PACKAGE_ROOT / "config" / "ai_config_google.json"),
    "openai": str(PACKAGE_ROOT / "config" / "ai_config_openai.json"),
    "openai-oauth": str(PACKAGE_ROOT / "config" / "ai_config_openai_oauth.json"),
    "groq": str(PACKAGE_ROOT / "config" / "ai_config_groq.json"),
    "minimax": str(PACKAGE_ROOT / "config" / "ai_config_minimax.json"),
    "openrouter": str(PACKAGE_ROOT / "config" / "ai_config_openrouter.json"),
    "opencode-go": str(PACKAGE_ROOT / "config" / "ai_config_opencode_go.json"),
    "opencode-zen": str(PACKAGE_ROOT / "config" / "ai_config_opencode_zen.json"),
    "claude": str(PACKAGE_ROOT / "config" / "ai_config_claude.json"),
    "commandcode": str(PACKAGE_ROOT / "config" / "ai_config_commandcode.json"),
    "hyper": str(PACKAGE_ROOT / "config" / "ai_config_hyper.json"),
    "grok": str(PACKAGE_ROOT / "config" / "ai_config_grok.json"),
    "nvidia": str(PACKAGE_ROOT / "config" / "ai_config_nvidia.json"),
    "gpt4free": str(PACKAGE_ROOT / "config" / "ai_config_gpt4free.json"),
    "cerebras": str(PACKAGE_ROOT / "config" / "ai_config_cerebras.json"),
    "mistral": str(PACKAGE_ROOT / "config" / "ai_config_mistral.json"),
    "cloudflare": str(PACKAGE_ROOT / "config" / "ai_config_cloudflare.json"),
    "sambanova": str(PACKAGE_ROOT / "config" / "ai_config_sambanova.json"),
    "chutes": str(PACKAGE_ROOT / "config" / "ai_config_chutes.json"),
    "pollinations": str(PACKAGE_ROOT / "config" / "ai_config_pollinations.json"),
    "ollama": str(PACKAGE_ROOT / "config" / "ai_config_ollama.json"),
    "lmstudio": str(PACKAGE_ROOT / "config" / "ai_config_lmstudio.json"),
}
# Providers whose model list lives in their config file and is picked at runtime.
CATALOGUE_PROVIDERS = ("opencode-go", "opencode-zen", "claude", "commandcode", "hyper", "grok", "nvidia", "gpt4free",
                       "cerebras", "mistral", "cloudflare", "sambanova", "chutes", "pollinations",
                       "ollama", "lmstudio")
# Local servers: the menu offers whatever they have loaded, whatever models.dev knows.
LOCAL_PROVIDERS = ("ollama", "lmstudio")
# Cap sent to a local model models.dev can't size.
LOCAL_MAX_OUTPUT = 32768
# Chat runs through the provider's own CLI, which lists only chat models.
CLI_PROVIDERS = ("commandcode", "opencode-zen")
# Where choose_ai remembers the last picks when a caller names no state file.
PROVIDER_STATE_FILE = REPO_ROOT / "provider_state.json"
# Offline fallback for the OpenAI API menu; online it lists what /v1/models serves.
OPENAI_MODEL_OPTIONS = ("gpt-6-sol", "gpt-6-luna", "gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-luna")

# models.dev publishes each model's context/output limits and list price per 1M
# tokens; opencode keeps a copy of it on disk. Sources are searched in order.
MODELS_DEV_URL = "https://models.dev/api.json"
MODELS_DEV_CACHE = Path.home() / ".cache" / "opencode" / "models.json"
MODELS_DEV_SOURCES = {
    "google": ("google",),
    "openai": ("openai",),
    "openai-oauth": ("openai",),
    "groq": ("groq",),
    "minimax": ("minimax",),
    "openrouter": ("openrouter",),
    "opencode-go": ("opencode-go",),
    "opencode-zen": ("opencode",),
    "claude": ("anthropic",),
    # Command Code resells other labs' models; show the maker's price when
    # models.dev has it, else OpenRouter's.
    "commandcode": ("anthropic", "openai", "google", "openrouter"),
    "hyper": ("hyper",),
    "grok": ("xai",),
    "nvidia": ("nvidia",),
    # gpt4free names models its own way ("qwen-3-235b"), so _gpt4free_facts matches
    # them to any models.dev entry by name shape; these makers' own listings win ties.
    "gpt4free": ("openai", "anthropic", "google", "deepseek", "xai", "mistral", "alibaba",
                 "moonshotai", "zai", "minimax", "meta", "cohere", "perplexity", "nvidia",
                 "stepfun", "thinkingmachines", "openrouter"),
    "cerebras": ("cerebras",),
    "mistral": ("mistral",),
    "cloudflare": ("cloudflare-workers-ai",),
    "sambanova": ("sambanova", "openrouter"),
    "chutes": ("chutes", "openrouter"),
    "pollinations": ("openrouter",),
    "ollama": ("openrouter",),
    "lmstudio": ("openrouter",),
}
# Artificial Analysis Intelligence Index per model, refetched on load once the disk
# copy is AA_REFRESH_DAYS old (default 1).
AA_REFRESH_DAYS = float(os.environ.get("AA_REFRESH_DAYS") or 1)
AA_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
AA_CACHE = Path.home() / ".cache" / "ai-book-creator" / "artificial_analysis.json"
# `cmdc --list-models` takes 5-25 s, so its listing is reused for a day.
CMDC_MODELS_CACHE = Path.home() / ".cache" / "ai-book-creator" / "cmdc_models.txt"
# Zen chat runs through the OpenCode CLI, whose own listing drops ids the gateway's
# /models still names but no longer serves (deepseek-v4-flash-free).
OPENCODE_MODELS_CACHE = Path.home() / ".cache" / "ai-book-creator" / "opencode_models.txt"
# Slug words that only name a reasoning setting or release channel.
AA_VARIANT_WORDS = frozenset(
    "thinking reasoning nonreasoning non adaptive preview exp low medium high xhigh max minimal".split()
)
# Hosted names AA lists under the open-weights model they serve.
AA_ALIASES = {
    "qwen3.5-plus": "qwen3-5-397b-a17b",
    "qwen3.8-flash": "qwen3-8-flash-next",
    # OpenCode's stealth model; GLM-4.6 by community identification, never confirmed.
    "big-pickle": "glm-4-6",
    # Command Code's "high-speed GLM-5.3 Flash".
    "glm-5.3-flashx": "glm-5-3-flash",
}
# Serving tiers of the same weights (kimi-k2.7-code-highspeed, hy3-paid); tried only
# when the full name misses, since grok-4-fast is its own model.
AA_SERVING_WORDS = frozenset("fast highspeed ultraspeed paid".split())
# A model without a score triggers a background refetch at most this often.
AA_RETRY_SECONDS = 3600
# Paid through a subscription, so the price shown is only the API list rate.
SUBSCRIPTION_PROVIDERS = ("claude", "commandcode", "opencode-go", "openai-oauth")


@lru_cache(maxsize=None)
def _models_dev() -> dict:
    """models.dev catalogue: opencode's copy if under a day old, else live, else stale."""
    try:
        if time.time() - MODELS_DEV_CACHE.stat().st_mtime < 86400:
            return json.loads(MODELS_DEV_CACHE.read_text(encoding="utf-8"))
    except Exception:
        pass
    try:
        req = Request(MODELS_DEV_URL, headers={"User-Agent": "ai-book-creator"})
        with urlopen(req, timeout=10) as resp:
            return json.load(resp)
    except Exception:
        pass
    try:
        return json.loads(MODELS_DEV_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


@lru_cache(maxsize=None)
def _artificial_analysis() -> list:
    """Artificial Analysis model list: disk copy if under AA_REFRESH_DAYS old, else live, else stale.

    Needs a free key in ARTIFICIAL_ANALYSIS_API_KEY; without one, returns [].
    """
    try:
        if time.time() - AA_CACHE.stat().st_mtime < AA_REFRESH_DAYS * 86400:
            return json.loads(AA_CACHE.read_text(encoding="utf-8"))
    except Exception:
        pass
    data = _fetch_artificial_analysis()
    if data:
        return data
    try:
        return json.loads(AA_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _fetch_artificial_analysis() -> list:
    """Live Artificial Analysis list, saved to AA_CACHE; [] without a key or on failure."""
    key = os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY", "").strip()
    if not key:
        return []
    try:
        req = Request(AA_URL, headers={"x-api-key": key, "User-Agent": "ai-book-creator"})
        with urlopen(req, timeout=10) as resp:
            data = json.load(resp).get("data") or []
        if data:
            AA_CACHE.parent.mkdir(parents=True, exist_ok=True)
            AA_CACHE.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception:
        return []


def _refresh_missing_intelligence(mids) -> bool:
    """Refetch Artificial Analysis in a detached process when a model has no score yet.

    The menu never waits: new scores show on the next load. Runs at most once per
    AA_RETRY_SECONDS (the cache mtime marks the last fetch). True when started.
    """
    if not os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY", "").strip():
        return False
    if all(_intelligence(mid) is not None for mid in mids):
        return False
    try:
        if time.time() - AA_CACHE.stat().st_mtime < AA_RETRY_SECONDS:
            return False
    except OSError:
        pass
    subprocess.Popen(
        [sys.executable, "-c", "from ai_suite.providers import _fetch_artificial_analysis as f; f()"],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        **background_process_options(),
    )
    return True


def _name_key(name: str) -> tuple[tuple[str, ...], frozenset[str]]:
    """'claude-sonnet-4-5' and 'claude-4-5-sonnet' -> (('4', '5'), {'claude', 'sonnet'}).

    Numbers keep their order (gpt-5.4 is not gpt-4.5); words don't.
    """
    # "qwen3-5" and "qwen-3.5" are one model: split a glued family name off its version.
    name = re.sub(r"([a-z]{2,})(\d)", r"\1-\2", name.lower())
    parts = [p for p in re.split(r"[^a-z0-9]+", name) if p and p not in ("free", "contributor")]
    return (tuple(p for p in parts if p.isdigit()), frozenset(p for p in parts if not p.isdigit()))


def _intelligence(mid: str) -> float | None:
    """Artificial Analysis Intelligence Index for a model id, or None if unmatched."""
    tail = mid.lower().rsplit("/", 1)[-1]
    numbers, words = _name_key(AA_ALIASES.get(tail, tail))
    words -= {"preview", "exp"}  # not "max": gpt-5.1-codex-max is its own model
    for key in dict.fromkeys((words, words - AA_SERVING_WORDS)):
        best = _intelligence_match(numbers, key)
        if best is not None:
            return best
    return None


def _intelligence_match(numbers: tuple, words: frozenset) -> float | None:
    best = None
    for entry in _artificial_analysis():
        score = (entry.get("evaluations") or {}).get("artificial_analysis_intelligence_index")
        if not isinstance(score, (int, float)):
            continue
        e_numbers, e_words = _name_key(str(entry.get("slug") or ""))
        # Trailing multi-digit numbers are snapshot dates: mimo-v2-5-0424, grok-build-0-1-06-16.
        dated = e_numbers[len(numbers):]
        if e_numbers[:len(numbers)] != numbers or any(len(n) < 2 for n in dated):
            continue
        if not words <= e_words:
            continue
        # AA slugs may add the maker and parameter count: nvidia-nemotron-3-ultra-550b-a55b.
        maker = str((entry.get("model_creator") or {}).get("slug") or "").lower()
        extra = {w for w in e_words - words if w != maker and not re.fullmatch(r"a?\d+b", w)}
        # ponytail: AA lists one entry per reasoning setting; best one wins.
        if extra <= AA_VARIANT_WORDS:
            best = max(best or 0.0, float(score))
    return best


def _model_facts(provider: str, mid: str) -> dict:
    """models.dev entry ({"limit": ..., "cost": ...}) plus AA "intelligence", or {}."""
    info = _models_dev_facts(provider, mid)
    score = _intelligence(mid)
    return {**info, "intelligence": score} if score is not None else info


@lru_cache(maxsize=None)
def _models_dev_index() -> dict:
    """Every models.dev entry by _name_key numbers: [(words, source rank, info)]."""
    rank = {source: i for i, source in enumerate(MODELS_DEV_SOURCES["gpt4free"])}
    index: dict = {}
    for source, data in _models_dev().items():
        for key, info in ((data or {}).get("models") or {}).items():
            numbers, words = _name_key(key.rsplit("/", 1)[-1])
            index.setdefault(numbers, []).append((words, rank.get(source, len(rank)), info))
    return index


def _gpt4free_facts(mid: str) -> dict:
    """models.dev facts for a g4f name, priced free: g4f bills nothing.

    Matches like _intelligence: same version numbers, the g4f words all present, and
    extras only variant words, "instruct"/"it" or parameter counts. Fewest extras win,
    then the maker's own listing.
    """
    numbers, words = _name_key(mid.rsplit("/", 1)[-1])
    best = None
    for e_words, rank, info in _models_dev_index().get(numbers, ()):
        extra = e_words - words
        if not words <= e_words or not all(
                w in AA_VARIANT_WORDS or w in ("instruct", "it") or re.fullmatch(r"a?\d+[be]", w)
                for w in extra):
            continue
        key = (len(extra), rank)
        if best is None or key < best[0]:
            best = (key, info)
    if best is None:
        return {}
    return {**best[1], "cost": {"input": 0, "output": 0}}


def _models_dev_facts(provider: str, mid: str) -> dict:
    if provider == "gpt4free":
        return _gpt4free_facts(mid)
    mid = mid.lower()
    tail = mid.rsplit("/", 1)[-1]
    for source in MODELS_DEV_SOURCES.get(provider, ()):
        models = (_models_dev().get(source) or {}).get("models") or {}
        # Claude Code aliases ("opus") follow the newest model of that family.
        family = [m for m in models.values() if m.get("family") == f"claude-{mid}"]
        if provider == "claude" and family:
            return max(family, key=lambda m: m.get("release_date", ""))
        by_id = {key.lower(): info for key, info in models.items()}
        by_tail = {key.lower().rsplit("/", 1)[-1]: info for key, info in models.items()}
        if mid in by_id or tail in by_tail:
            return by_id.get(mid) or by_tail[tail]
    return {}


def _tokens(n: int) -> str:
    """1048576 -> '1M', 131072 -> '131K'."""
    return f"{n / 1e6:.3g}M" if n >= 999_500 else f"{round(n / 1e3)}K"


SORT_KEYS = {"price": "price", "ctx": "context", "AA": "intelligence", "score": "aggregate score"}


def _scales(infos: list[dict]) -> list[dict]:
    """Each model's ctx, price and AA scaled 0..1 across the menu (1 = best), plus
    "score": their mean, with a metric the model lacks counted as 0."""
    def raw(info: dict) -> dict:
        ctx = (info.get("limit") or {}).get("context")
        cost = info.get("cost") or {}
        # Logs: context and price span orders of magnitude. Output price is what
        # a book's worth of prose costs.
        return {"ctx": math.log(ctx) if ctx else None,
                "price": -math.log1p(cost["output"]) if "output" in cost else None,
                "AA": info.get("intelligence")}

    raws = [raw(i) for i in infos]
    scaled: list[dict] = [{} for _ in infos]
    present = [k for k in ("ctx", "price", "AA") if any(r[k] is not None for r in raws)]
    for key in present:
        values = [r[key] for r in raws if r[key] is not None]
        lo, hi = min(values), max(values)
        for r, s in zip(raws, scaled):
            if r[key] is not None:
                s[key] = (r[key] - lo) / (hi - lo) if hi > lo else 1.0
    for s in scaled:
        s["score"] = sum(s.values()) / len(present) if present else 0.0
    # Rescale the score too, so its colors also run from the menu's worst to best.
    lo, hi = min((s["score"] for s in scaled), default=0), max((s["score"] for s in scaled), default=0)
    for s in scaled:
        s["score_t"] = (s["score"] - lo) / (hi - lo) if hi > lo else 1.0
    return scaled


def _gradient(text: str, t: float | None) -> str:
    """t 0..1 colors red through orange and yellow to bright green (24-bit).

    The red end stays bright enough to read on a dark console (low vision).
    """
    if t is None:
        return text
    r, g, b = colorsys.hsv_to_rgb(t / 3, 1, 0.8 + 0.2 * t)
    return _color(text, f"38;2;{round(r * 255)};{round(g * 255)};{round(b * 255)}")


def _facts_label(info: dict, scale: dict | None = None) -> str:
    """'ctx 1M | $4/$20 | AA 45 -> 72' — price is $ per 1M input/output tokens.

    scale (from _scales) colors each figure; the arrow and aggregate score need it.
    """
    scale = scale or {}
    context = (info.get("limit") or {}).get("context")
    parts = [_gradient(f"ctx {_tokens(context)}", scale.get("ctx")) if context else "ctx ?"]
    cost = info.get("cost") or {}
    if "input" in cost and "output" in cost:
        free = not (cost["input"] or cost["output"])
        price = "free" if free else f"${cost['input']:.3g}/${cost['output']:.3g}"
        parts.append(_gradient(price, scale.get("price")))
    else:
        parts.append("$?")
    if _artificial_analysis():
        score = info.get("intelligence")
        parts.append(_gradient(f"AA {score:.0f}", scale.get("AA")) if score is not None else "AA ?")
    label = " | ".join(parts)
    if "score" in scale:
        label += " -> " + _gradient(f"{scale['score'] * 100:.0f}", scale["score_t"])
    return label


def _cost_key(info: dict) -> tuple:
    """Menu sort: free first, then output price, then input; unpriced last."""
    cost = info.get("cost") or {}
    if "input" not in cost or "output" not in cost:
        return (1, 0, 0)
    return (0, cost["output"], cost["input"])


def _enable_ansi() -> None:
    """Enable console escapes without spawning cmd.exe for every label."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetStdHandle.restype = wintypes.HANDLE
        kernel.GetConsoleMode.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.SetConsoleMode.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        mode = wintypes.DWORD()
        handle = kernel.GetStdHandle(-11)
        if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel.SetConsoleMode(handle, mode.value | 0x0004)


def _color(text: str, code: str) -> str:
    """ANSI-colored text on a terminal; plain when piped or NO_COLOR is set."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    _enable_ansi()
    return f"\033[{code}m{text}\033[0m"


def _cli_listing(args: list[str], cache: Path) -> str:
    """A CLI's model listing, reused for a day."""
    try:
        if time.time() - cache.stat().st_mtime < 86400:
            return cache.read_text(encoding="utf-8")
    except OSError:
        pass
    try:
        listing = subprocess.run(
            [shutil.which(args[0]) or args[0], *args[1:]],
            capture_output=True, text=True, encoding="utf-8", timeout=30, check=True,
            **background_process_options(),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        if cache.exists():  # a stale listing beats an empty menu
            return cache.read_text(encoding="utf-8")
        raise
    if listing.strip():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(listing, encoding="utf-8")
    return listing


def _live_model_ids(provider: str) -> set[str] | None:
    """Model ids the provider serves right now, or None when it can't be asked."""
    try:
        if provider == "commandcode":
            listing = _cli_listing(["cmdc", "--list-models"], CMDC_MODELS_CACHE)
            # Model rows start at column 0 with an id; headings ("Available models", "Open
            # Source") have no dash or slash in their first word.
            return {word.lower() for word in (line.split()[0] for line in listing.splitlines()
                    if line.strip() and not line[0].isspace()) if "-" in word or "/" in word} or None
        if provider == "opencode-zen":
            listing = _cli_listing(["opencode", "models", "opencode"], OPENCODE_MODELS_CACHE)
            return {line.strip().split("/", 1)[1].lower() for line in listing.splitlines()
                    if line.strip().startswith("opencode/")} or None
        if provider == "claude":
            # The CLI can't list models; models.dev names each family's newest release
            # (on a same-day tie the shorter id: the bare one, not its dated snapshot).
            newest: dict = {}
            for mid, info in ((_models_dev().get("anthropic") or {}).get("models") or {}).items():
                family = info.get("family")
                key = (info.get("release_date", ""), -len(mid))
                if family and (family not in newest or key > newest[family][0]):
                    newest[family] = (key, mid.lower())
            return {mid for _, mid in newest.values()} or None
        with open(PROVIDER_CONFIG_MAP[provider], "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("base_url"):
            return None
        # opencode's CDN answers urllib's default User-Agent with a 403.
        req = Request(data["base_url"].rstrip("/") + "/models",
                      headers={"User-Agent": "ai-book-creator"})
        key = os.environ.get(data.get("api_key_env", ""), "")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        with urlopen(req, timeout=5) as resp:
            # gpt4free also lists each backend (provider: true) and image models.
            return {str(m["id"]).lower() for m in json.load(resp)["data"]
                    if not m.get("provider") and not m.get("image")}
    except Exception:
        return None


def _load_catalogue(provider: str) -> dict:
    # Local config first (carries friendly display names like "GLM-5.2").
    out: dict = {}
    try:
        with open(PROVIDER_CONFIG_MAP[provider], "r", encoding="utf-8") as f:
            data = json.load(f)
        for mid, info in (data.get("models") or {}).items():
            out[str(mid).lower()] = [str(info.get("name", mid)), int(info.get("max_output", 4096))]
    except Exception:
        pass
    # Then overlay the output limits from opencode's userspace config for the
    # curated opencode-go models; the live listing below decides what exists.
    if provider == "opencode-go":
        for mid, info in load_opencode_go_sync().get("models", {}).items():
            if mid in out:
                out[mid][1] = int(info["max_output"])
    # Drop curated ids the provider has retired, so the menu never offers a
    # dead model. Offline or unreachable: keep the curated list as is.
    live = _live_model_ids(provider)
    if live and provider == "claude":
        # Only a newest-per-family list, so it adds but never retires; the CLI takes no cap.
        for mid in sorted(live):
            # Skip it when listed bare or as a dated snapshot (claude-haiku-4-5-20251001),
            # but claude-opus-5-5 does not stand in for claude-opus-5.
            if not any(re.fullmatch(re.escape(mid) + r"(-\d{8})?", have) for have in out):
                out[mid] = [mid, 0]
    elif live and (provider in LOCAL_PROVIDERS or any(mid in live for mid in out)):
        out = {mid: entry for mid, entry in out.items() if mid in live}
        # And offer what the provider added since the list was curated, when
        # models.dev can say how much it writes (skips image/embedding ids). The
        # CLI providers list only chat models and take no cap, so all of theirs join.
        for mid in sorted(live - set(out)):
            max_out = int((_model_facts(provider, mid).get("limit") or {}).get("output") or 0)
            if provider in LOCAL_PROVIDERS:
                max_out = max_out or LOCAL_MAX_OUTPUT
            if max_out or provider in CLI_PROVIDERS:
                out[mid] = [mid, 0 if provider == "commandcode" else max_out]
    if not out:
        fallback = (
            {"nemotron-3-ultra-free": ["Nemotron 3 Ultra Free", 128000]}
            if provider == "opencode-zen"
            else {"glm-5.3": ["GLM-5.3", 131072]}
        )
        out = fallback
    return out


_CATALOGUE_CACHE: dict[str, dict] = {}


def _provider_models(provider: str) -> dict:
    """{model id: [display name, max output tokens]} for a catalogue provider."""
    if provider not in _CATALOGUE_CACHE:
        _CATALOGUE_CACHE[provider] = _load_catalogue(provider)
    return _CATALOGUE_CACHE[provider]


def _model_state_key(provider: str) -> str:
    return f"{provider.replace('-', '_')}_model"

def _load_provider_state(state_file: Path | None = None) -> dict:
    state_file = state_file or PROVIDER_STATE_FILE
    try:
        if state_file.exists():
            with state_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _load_last_provider(default_provider: str = "google", state_file: Path | None = None) -> str:
    data = _load_provider_state(state_file)
    provider = str(data.get("provider", "")).lower()
    if provider in PROVIDER_CONFIG_MAP:
        return provider
    return default_provider


def _default_openai_model() -> str:
    try:
        base_path = Path(PROVIDER_CONFIG_MAP["openai"])
        local_path = base_path.with_name(base_path.stem + ".local.json")
        path_to_load = local_path if local_path.exists() else base_path
        
        with open(path_to_load, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            candidate = str(data.get("writing_model") or data.get("review_model") or "").lower()
            if candidate in OPENAI_MODEL_OPTIONS:
                return candidate
    except Exception:
        pass
    return "gpt-6-luna"


def _load_last_openai_model(
    default_model: str | None = None,
    options: tuple[str, ...] = OPENAI_MODEL_OPTIONS,
    state_key: str = "openai_model",
    state_file: Path | None = None,
) -> str:
    data = _load_provider_state(state_file)
    model = str(data.get(state_key, "")).lower()
    if model in options:
        return model
    fallback = default_model or _default_openai_model()
    return fallback if fallback in options else options[0]


def _save_last_provider(provider: str, openai_model: str | None = None,
                        catalogue_model: str | None = None,
                        state_file: Path | None = None, extra: dict | None = None) -> None:
    state_file = state_file or PROVIDER_STATE_FILE
    data = _load_provider_state(state_file)
    provider = provider.lower()
    data["provider"] = provider
    if openai_model is not None:
        data["openai_oauth_model" if provider == "openai-oauth" else "openai_model"] = openai_model.lower()
    elif provider == "openai" and str(data.get("openai_model", "")).lower() not in OPENAI_MODEL_OPTIONS:
        data["openai_model"] = _default_openai_model()
    if provider in CATALOGUE_PROVIDERS:
        key = _model_state_key(provider)
        models = _provider_models(provider)
        if catalogue_model is not None:
            data[key] = catalogue_model.lower()
        elif str(data.get(key, "")).lower() not in models:
            data[key] = next(iter(models))
    data.update(extra or {})

    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _prompt_provider(default_provider: str) -> str:
    valid_providers = list(PROVIDER_CONFIG_MAP.keys())
    width = max(map(len, valid_providers))
    default_tag = "  " + _color("default", "2")
    picked = _arrow_menu("Provider", [
        (p, f"{p:<{width}}  {PROVIDER_LABELS.get(p, '')}{default_tag if p == default_provider else ''}")
        for p in valid_providers], default_provider)
    if picked is not None:
        return picked
    prompt = f"Choose provider ({', '.join(valid_providers)}) [default: {default_provider}]: "

    while True:
        try:
            choice = input(prompt).strip().lower()
        except EOFError:
            return default_provider
        if not choice:
            return default_provider
        if choice in PROVIDER_CONFIG_MAP:
            return choice
        print(f"Invalid provider. Please choose one of: {', '.join(valid_providers)}")


def _normalize_openai_model(choice: str) -> str:
    normalized = choice.strip().lower()
    if normalized.isdigit() and 1 <= int(normalized) <= len(OPENAI_MODEL_OPTIONS):
        return OPENAI_MODEL_OPTIONS[int(normalized) - 1]
    return normalized if normalized in OPENAI_MODEL_OPTIONS else ""


def _load_openai_api_models() -> dict[str, tuple[int | None, int | None]] | None:
    """Text models the OpenAI API serves now; None without a key or offline."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    try:
        req = Request("https://api.openai.com/v1/models",
                      headers={"Authorization": f"Bearer {key}", "User-Agent": "ai-book-creator"})
        with urlopen(req, timeout=10) as resp:
            ids = sorted(str(m["id"]) for m in json.load(resp)["data"])
    except Exception:
        return None
    # Dated snapshots repeat their alias; audio, realtime, embedding and the rest are
    # not chat-completion text models, and -pro ids answer only the Responses API.
    # models.dev's output limit screens out anything else it does not know.
    skip = re.compile(r"-\d{4}-\d{2}-\d{2}$|audio|realtime|search|transcribe|tts|image|embedding|codex|chat-latest|deep-research|-pro$")
    models = {mid: (None, None) for mid in ids if not skip.search(mid)
              and (_model_facts("openai", mid).get("limit") or {}).get("output")}
    return models or None


def _load_openai_oauth_models() -> dict[str, tuple[int | None, int | None]]:
    ensure_openai_oauth_proxy()
    with urlopen("http://127.0.0.1:10531/v1/models", timeout=10) as response:
        payload = json.load(response)
    models: dict[str, tuple[int | None, int | None]] = {}
    for item in payload.get("data", []) if isinstance(payload, dict) else []:
        model_id = str(item.get("id", "")).strip() if isinstance(item, dict) else ""
        if not model_id or "image" in model_id.lower():
            continue
        context = item.get("context_window")
        output = item.get("max_output_tokens")
        models[model_id] = (
            int(context) if isinstance(context, int) and context > 0 else None,
            int(output) if isinstance(output, int) and output > 0 else None,
        )
    if not models:
        raise RuntimeError("OpenAI OAuth returned no text models from /v1/models.")
    return models


def _prompt_openai_model(
    default_model: str,
    model_info: dict[str, tuple[int | None, int | None]] | None = None,
    role: str = "",
    oauth: bool = True,
) -> str:
    options = tuple(model_info) if model_info is not None else OPENAI_MODEL_OPTIONS
    rows = []
    for model in options:
        context, output = model_info.get(model, (None, None)) if model_info is not None else (None, None)
        info = _model_facts("openai", model)
        # The OAuth endpoint's own figures win over models.dev when it reports them.
        reported = {k: v for k, v in (("context", context), ("output", output)) if v}
        rows.append((model, {**info, "limit": {**(info.get("limit") or {}), **reported}}))
    oauth = oauth and model_info is not None
    title = ("OpenAI via ChatGPT sign-in (live)" if oauth
             else "OpenAI (live)" if model_info is not None else "OpenAI")
    paid_by = "; your ChatGPT subscription pays" if oauth else ""
    return _pick_model(title, rows, default_model, role, paid_by,
                       None if model_info is not None else _normalize_openai_model)


MENU_ROWS = 20  # rows on screen at once; the rest scroll


class _BelowLabel:
    """sys.stdout while a loading label shows: anything printed meanwhile (a CLI
    notice, a retry warning) wipes the label first and redraws it after a finished
    line, so the label always sits on the last row and never gets stranded."""

    def __init__(self, inner, text: str):
        self.inner, self.text, self.shown = inner, text, False

    def show(self) -> None:
        self.inner.write(self.text)
        self.inner.flush()
        self.shown = True

    def hide(self) -> None:
        if self.shown:
            self.inner.write("\r\033[K")
            self.inner.flush()
            self.shown = False

    def write(self, data: str) -> int:
        self.hide()
        n = self.inner.write(data)
        if data.endswith("\n"):
            self.show()
        return n

    def __getattr__(self, name):
        return getattr(self.inner, name)


@contextlib.contextmanager
def _loading(what: str):
    """"Loading <what>..." on a terminal while the block runs, in place on the last
    row, erased when it ends. Plain passthrough when piped."""
    if not sys.stdout.isatty():
        yield
        return
    _enable_ansi()
    label = _BelowLabel(sys.stdout, _color(f"Loading {what}...", "2"))
    label.show()
    sys.stdout = label
    try:
        yield
    finally:
        sys.stdout = label.inner
        label.hide()


def _arrow_menu(title: str, rows: list[tuple[str, str]], default: str, footer: str = "",
                sorts: list[tuple[str, list[str]]] | None = None, name: str = "",
                multi: bool = False) -> str | None:
    """Arrow-key menu on a Windows console, MENU_ROWS rows at a time.

    Up/Down (or W/S)/PgUp/PgDn/Home/End move, Space selects or deselects the row under the
    cursor, Enter confirms the selection (the cursor row when nothing is selected),
    Esc keeps the default, Tab cycles `sorts` [(name, ids in order)]. Opens with the
    default selected. On confirm the menu collapses to one "<name>: <pick>" line.
    multi=True: Space toggles rows, numbered in the order picked; default and result are
    comma-joined ids in that order, and Enter with nothing picked returns "".
    Returns None off a console so callers fall back to typing.
    """
    if os.name != "nt" or not sys.stdin.isatty() or not sys.stdout.isatty():
        # ponytail: Windows console only; termios raw mode if that's ever needed.
        return None
    import msvcrt
    text = dict(rows)
    sorts = sorts or [("", [rid for rid, _ in rows])]
    sort = 0
    picks = [d for d in (default.split(",") if multi else [default]) if d in text]
    cur = sorts[0][1].index(picks[0]) if picks else 0
    top = drawn = 0
    _enable_ansi()
    keys = "↑↓/W S move · Space select · Enter confirm · Esc default"
    if len(sorts) > 1:
        keys += " · Tab sort"

    def redraw(lines: list[str]) -> None:
        nonlocal drawn
        # Back to the title line, clear everything below it, draw again.
        print((f"\033[{drawn}A" if drawn else "") + "\r\033[J" + "\n".join(lines), flush=True)
        drawn = len(lines)

    def done(picked: str) -> str:
        shown = picked.replace(",", ", ") if picked or not multi else "none"
        redraw([f"{_color('✓', '32')} {name or title.rstrip(':')}: {_color(shown, '1')}"])
        return picked

    while True:
        order = sorts[sort][1]
        top = min(max(top, cur - MENU_ROWS + 1), cur)
        shown = order[top:top + MENU_ROWS]
        sorted_by = f"  sorted by {sorts[sort][0]}" if len(sorts) > 1 else ""
        lines = [_color(title, "1") + _color(sorted_by, "2")]
        for i, rid in enumerate(shown):
            mark = (str(picks.index(rid) + 1) if multi else "x") if rid in picks else " "
            row = f"[{mark}] {text[rid]}"
            lines.append(_color(f"  > {row}", "1;36") if top + i == cur else f"    {row}")
        scroll = f"{top + 1}-{top + len(shown)} of {len(order)}"
        lines.append(_color(f"    {scroll} · {keys}", "2"))
        lines += [_color(line, "2") for line in footer.splitlines()]
        redraw(lines)
        ch = msvcrt.getwch()
        if ch in "\x00\xe0":  # arrow/function key: the second half says which
            step = {"H": -1, "P": 1, "I": -MENU_ROWS, "Q": MENU_ROWS,
                    "G": -len(order), "O": len(order)}.get(msvcrt.getwch(), 0)
            cur = min(max(cur + step, 0), len(order) - 1)
        elif ch in "wWsS":
            cur = min(max(cur + (-1 if ch in "wW" else 1), 0), len(order) - 1)
        elif ch == " ":
            if order[cur] in picks:
                picks.remove(order[cur])
            else:
                picks = [*picks, order[cur]] if multi else [order[cur]]
        elif ch in "\r\n":
            return done(",".join(picks) if multi else (picks or [order[cur]])[0])
        elif ch == "\t" and len(sorts) > 1:
            sort = (sort + 1) % len(sorts)
            cur = sorts[sort][1].index(order[cur])
        elif ch in "\x1b\x1a":
            return done(default)
        elif ch == "\x03":
            raise KeyboardInterrupt


def _pick_model(title: str, rows: list[tuple[str, dict]], default_model: str, role: str,
                paid_by: str, normalize=None) -> str:
    """Arrow-key model menu (`_arrow_menu`), cheapest first; Tab re-sorts by the next
    of SORT_KEYS, best first. Off a console: numbered list, number or id typed."""
    ids = [mid for mid, _ in sorted(rows, key=lambda row: _cost_key(row[1]))]
    infos = dict(rows)
    scales = dict(zip(infos, _scales(list(infos.values()))))
    width = max(map(len, ids))
    default_tag = "  " + _color("default", "2")
    label = {mid: f"{mid:<{width}}  {_facts_label(infos[mid], scales[mid])}"
                  f"{default_tag if mid == default_model else ''}" for mid in ids}
    # Stable over the price order, so ties stay cheapest first; unknowns go last.
    sorts = [(name, sorted(ids, key=lambda mid: -scales[mid].get(key, -1)))
             for key, name in SORT_KEYS.items()]
    legend = f"    $ = API list price per 1M in/out tokens{paid_by}; -> = aggregate score 0-100."
    name = f"{role.capitalize()} model" if role else "Model"
    heading = f"{name} · {title}"
    picked = _arrow_menu(heading, [(mid, label[mid]) for mid in ids], default_model, legend, sorts,
                         name=name)
    if picked is not None:
        return picked

    order = sorts[0][1]
    print(heading)
    for i, mid in enumerate(order, 1):
        print(f"  {i:2d}. {label[mid]}")
    print(legend)
    prompt = f"Choose {role + ' ' if role else ''}model number or id [default: {default_model}]: "
    while True:
        try:
            choice = input(prompt).strip()
        except EOFError:
            return default_model
        if not choice:
            return default_model
        if choice.isdigit() and 1 <= int(choice) <= len(order):
            return order[int(choice) - 1]
        for model in (choice.lower(), normalize(choice) if normalize else None):
            if model in infos:
                return model
        print(f"Choose a number from 1 to {len(order)} or a listed model id.")


def _default_catalogue_model(provider: str) -> str:
    models = _provider_models(provider)
    try:
        with open(PROVIDER_CONFIG_MAP[provider], "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            candidate = str(data.get("writing_model") or "").lower()
            if candidate in models:
                return candidate
    except Exception:
        pass
    return next(iter(models))


def _load_last_catalogue_model(provider: str, default_model: str | None = None,
                               state_key: str | None = None, state_file: Path | None = None) -> str:
    models = _provider_models(provider)
    data = _load_provider_state(state_file)
    model = str(data.get(state_key or _model_state_key(provider), "")).lower()
    if model in models:
        return model
    if default_model and default_model.lower() in models:
        return default_model.lower()
    return _default_catalogue_model(provider)


PROVIDER_LABELS = {
    "google": "Google Gemini API",
    "openai": "OpenAI API",
    "openai-oauth": "OpenAI via ChatGPT sign-in (your subscription)",
    "groq": "Groq",
    "minimax": "MiniMax",
    "openrouter": "OpenRouter",
    "nvidia": "NVIDIA NIM",
    "opencode-go": "OpenCode Go",
    "opencode-zen": "OpenCode Zen (through the OpenCode CLI)",
    "claude": "Claude Code (your subscription, no API key)",
    "commandcode": "Command Code (your subscription, no API key)",
    "hyper": "hyper.charm.land",
    "grok": "xAI Grok",
    "gpt4free": "gpt4free (local `g4f api` server, free)",
    "cerebras": "Cerebras (free tier)",
    "mistral": "Mistral La Plateforme (free Experiment tier)",
    "cloudflare": "Cloudflare Workers AI (free daily allowance)",
    "sambanova": "SambaNova Cloud (free tier)",
    "chutes": "Chutes (pay as you go, no free tier)",
    "pollinations": "Pollinations (daily free pollen grant)",
    "ollama": "Ollama (local, no key)",
    "lmstudio": "LM Studio (local, no key)",
}


def _prompt_catalogue_model(provider: str, default_model: str, role: str = "") -> str:
    models = _provider_models(provider)
    label = PROVIDER_LABELS.get(provider, provider)
    rows = [(mid, _model_facts(provider, mid)) for mid in models]
    paid_by = "; your subscription pays" if provider in SUBSCRIPTION_PROVIDERS else ""
    return _pick_model(label, rows, default_model, role, paid_by)


def _effort_levels(provider: str, mid: str) -> list[str]:
    """Reasoning efforts models.dev lists for the model; [] when it lists none or the
    provider's transport cannot send one."""
    if provider == "gpt4free":
        # ponytail: its facts are a fuzzy name match and whether the g4f server forwards
        # an effort is unchecked; offer levels once a live request confirms it.
        return []
    with open(provider_config_path(provider), "r", encoding="utf-8") as f:
        if str(json.load(f).get("provider", "")).lower() not in EFFORT_PROVIDERS:
            return []
    for option in _models_dev_facts(provider, mid).get("reasoning_options") or ():
        if option.get("type") == "effort":
            return [str(level) for level in option.get("values") or ()]
    return []


def _prompt_effort(levels: list[str], default: str, role: str = "") -> str:
    """Arrow-key effort menu: the model's own levels, or "" for the provider's default."""
    name = f"{role.capitalize()} effort" if role else "Effort"
    rows = [("default", "default  the provider's own")] + [(level, level) for level in levels]
    # ponytail: asked on a console only. Off one the remembered pick stands, because a
    # typed prompt here would eat the next scripted answer of a piped run.
    picked = _arrow_menu(name, rows, default or "default", name=name)
    return default if picked is None else "" if picked == "default" else picked




def provider_options() -> dict[str, dict]:
    """Ordered provider catalogue for consumers that own their routing UI.

    Config files only: consumers call this at import, so it never asks a provider what
    it serves. choose_ai's model menu is where the live listing happens."""
    options = {}
    for provider in PROVIDER_CONFIG_MAP:
        with open(provider_config_path(provider), "r", encoding="utf-8") as f:
            data = json.load(f)
        options[provider] = {
            "provider": provider,
            "label": PROVIDER_LABELS.get(provider, provider),
            "models": tuple(str(mid).lower() for mid in data.get("models") or ())
                      if provider in CATALOGUE_PROVIDERS else (),
            "writing_model": str(data.get("writing_model") or "").lower(),
            "review_model": str(data.get("review_model") or data.get("writing_model") or "").lower(),
            "needs_api_key": bool(data.get("api_key_env")),
        }
    return options


def provider_config_path(provider: str) -> str:
    """Config file AIService should load for a provider: the user's .local.json copy when present.

    Consumers that choose the provider themselves (book-watch's report picker, lamplight's
    wizard, the calibre plugin) call this instead of the interactive choose_ai menu.
    """
    provider = "openai-oauth" if provider == "codex" else provider
    base = Path(PROVIDER_CONFIG_MAP[provider])
    local = base.with_name(base.stem + ".local.json")
    return str(local if local.exists() else base)


def choose_ai(
    provider: str | None = None,
    mode: str = "review",
    state_file: Path | None = None,
    roles: tuple[str, ...] = ("writing",),
    defaults: tuple[str, ...] = (),
    default_provider: str = "google",
    effort: bool = True,
) -> tuple[str, str, list[str]]:
    """Provider and model menu shared by every script that uses AIService.

    Book writer, mathforge and music writer all call this one function, so a
    provider or model added here, or newly served by a provider, shows up in
    each of them. Asks for the provider unless one is given, then one model per
    role (the first role writes, the last reviews). mode="auto" asks nothing and
    reuses the picks remembered in state_file (default: PROVIDER_STATE_FILE).
    Each model that lists reasoning efforts then gets an effort menu of its own
    levels; effort=False skips it for a caller with its own effort option.
    Exports AI_CONFIG_PATH, the role models, efforts and completion caps; returns
    (provider, config path, models).
    """
    if provider is None:
        last = _load_last_provider(default_provider, state_file)
        provider = last if mode == "auto" else _prompt_provider(last)
    provider = "openai-oauth" if provider == "codex" else provider
    config_path = provider_config_path(provider)
    if config_path.endswith(".local.json"):
        print(f"Loaded local configuration: {Path(config_path).name}")
    os.environ["AI_CONFIG_PATH"] = config_path

    models: list[str] = []
    state_keys: list[str] = []
    label = lambda role: role if len(roles) > 1 else ""

    # One effort per role, asked right after that role's model from the levels the model
    # lists; AIService reads the first as the writing effort and the last as the review effort.
    saved = _load_provider_state(state_file)
    efforts: list[str] = []
    answered: dict[str, str] = {}  # only an answered menu is remembered

    def pick_effort(role: str, mid: str) -> None:
        key = f"{provider.replace('-', '_')}_effort" + (f"_{role}" if efforts else "")
        # Like the models: a role with no remembered effort starts on the first role's.
        last = str(saved[key] if key in saved else efforts[0] if efforts else "")
        picked = ""
        # Auto mode with nothing remembered never loads models.dev.
        if effort and mid and (mode != "auto" or last):
            levels = _effort_levels(provider, mid)
            # models.dev unreachable says nothing about the model: the remembered effort stands.
            picked = last if last in levels or not _models_dev() else ""
            if levels and mode != "auto":
                picked = answered[key] = _prompt_effort(levels, picked, label(role))
        efforts.append(picked)

    if mode != "auto" and (provider in CATALOGUE_PROVIDERS or provider in ("openai", "openai-oauth")):
        # The live catalogue, models.dev and Artificial Analysis are all fetched (and
        # cached) before the first menu can draw; say so instead of sitting silent.
        with _loading(f"{PROVIDER_LABELS.get(provider, provider)} models"):
            if provider in CATALOGUE_PROVIDERS:
                _refresh_missing_intelligence(_provider_models(provider))
            _models_dev()
            _artificial_analysis()
    if provider in ("openai", "openai-oauth"):
        with _loading("OpenAI models"):
            model_info = (_load_openai_oauth_models() if provider == "openai-oauth"
                          else _load_openai_api_models())
        options = tuple(model_info) if model_info is not None else OPENAI_MODEL_OPTIONS
        base_key = "openai_oauth_model" if provider == "openai-oauth" else "openai_model"
        for i, role in enumerate(roles):
            key = base_key if i == 0 else f"{base_key}_{role}"
            # A role with no pick of its own yet starts on the first role's.
            fallback = defaults[i] if i < len(defaults) else models[0] if i else (
                "gpt-6-sol" if provider == "openai-oauth" else None)
            default_model = _load_last_openai_model(fallback, options, key, state_file)
            models.append(default_model if mode == "auto"
                          else _prompt_openai_model(default_model, model_info, label(role),
                                                     provider == "openai-oauth"))
            state_keys.append(key)
            pick_effort(role, models[-1])
        os.environ["AI_WRITING_MODEL"] = models[0]
        os.environ["AI_REVIEW_MODEL"] = models[-1]
        os.environ["AI_OPENAI_MODEL"] = models[0]
    elif provider in CATALOGUE_PROVIDERS:
        for i, role in enumerate(roles):
            key = _model_state_key(provider) + ("" if i == 0 else f"_{role}")
            fallback = defaults[i] if i < len(defaults) else models[0] if i else None
            default_model = _load_last_catalogue_model(provider, fallback, key, state_file)
            models.append(default_model if mode == "auto"
                          else _prompt_catalogue_model(provider, default_model, label(role)))
            state_keys.append(key)
            pick_effort(role, models[-1])
        os.environ["AI_WRITING_MODEL"] = models[0]
        os.environ["AI_REVIEW_MODEL"] = models[-1]
        # The Claude Code CLI has no completion-token argument, so its catalogue
        # carries max_output 0 and the caps stay unset.
        write_out = _provider_models(provider)[models[0]][1]
        review_out = _provider_models(provider)[models[-1]][1]
        for key, max_out in (
            ("AI_WRITING_COMPLETION_TOKENS", write_out),
            ("AI_REVIEW_COMPLETION_TOKENS", review_out),
            ("AI_PLANNING_COMPLETION_TOKENS", write_out),
            ("AI_DEFAULT_COMPLETION_TOKENS", write_out),
        ):
            if max_out:
                os.environ[key] = str(max_out)
            else:
                os.environ.pop(key, None)
    else:
        # No model menu for these providers; still show what the config runs on.
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        writing = data.get("writing_model") or ""
        review = data.get("review_model") or writing
        for mid in dict.fromkeys(filter(None, (writing, review))):
            print(f"Model {mid}: {_facts_label(_model_facts(provider, mid))} ($ per 1M in/out)")
        models = [writing] + [review] * (len(roles) - 1)
        for role, mid in zip(roles, models):
            pick_effort(role, mid)

    for name, picked in (("AI_WRITING_EFFORT", efforts[0]), ("AI_REVIEW_EFFORT", efforts[-1])):
        if picked:
            os.environ[name] = picked
        else:
            os.environ.pop(name, None)

    first = models[0] if state_keys else None
    _save_last_provider(
        provider,
        first if provider in ("openai", "openai-oauth") else None,
        first if provider in CATALOGUE_PROVIDERS else None,
        state_file,
        {**dict(zip(state_keys[1:], models[1:])), **answered},
    )
    return provider, config_path, models
