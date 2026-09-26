from __future__ import annotations

import json
import os
import math
import re
import shutil
import socket
import subprocess
import time
import tempfile
import threading
import uuid
from contextlib import contextmanager
from functools import wraps
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List, Tuple
from urllib.parse import urlparse

try:
    import requests
except ImportError:  # Calibre's bundled Python has none; _StdlibSession stands in
    requests = None  # type: ignore

# Try optional OpenAI python client
try:
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI_CLIENT = True
except Exception:
    OpenAI = None  # type: ignore
    _HAS_OPENAI_CLIENT = False

# Try optional Google Gemini client
try:
    from google import genai  # type: ignore
    _HAS_GEMINI_CLIENT = True
except Exception:
    genai = None  # type: ignore
    _HAS_GEMINI_CLIENT = False


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "config", "ai_config_minimax.json"
)
DEFAULT_USAGE_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "config", "ai_usage_state.json"
)
DEFAULT_GROQ_RATE_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "config", "groq_usage_state.json"
)
OPENAI_OAUTH_PORT = 10531


def _openai_oauth_proxy_running() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", OPENAI_OAUTH_PORT), timeout=0.5):
            return True
    except OSError:
        return False


def ensure_openai_oauth_proxy() -> None:
    if _openai_oauth_proxy_running():
        return
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if not npx:
        raise RuntimeError("OpenAI OAuth requires Node.js with npx on PATH.")
    print("Starting the OpenAI OAuth proxy; complete browser sign-in if prompted...")
    try:
        subprocess.run([npx, "openai-oauth@latest", "--detach"], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("OpenAI OAuth proxy failed to start.") from exc
    for _ in range(40):
        if _openai_oauth_proxy_running():
            return
        time.sleep(0.25)
    raise RuntimeError(f"OpenAI OAuth proxy did not open port {OPENAI_OAUTH_PORT}.")

# ponytail: live-read opencode's userspace truth so our .env / ai_config stays in sync.
# Both are single-file JSON written by the opencode CLI on `opencode auth login`.
OPENCODE_USER_CONFIG = Path.home() / ".config" / "opencode" / "opencode.jsonc"
OPENCODE_USER_AUTH = Path.home() / ".local" / "share" / "opencode" / "auth.json"
# Same opencode account key serves both the paid "opencode-go" tier and the
# pay-per-use "opencode-zen" tier (which carries the free models).
OPENCODE_PROVIDER_KEYS = ("opencode-zen", "opencode", "opencode-go")


def load_opencode_go_sync() -> Dict[str, Any]:
    """Read opencode API key + per-model context/output limits from opencode's
    own userspace files. Returns {"api_key": str, "models": {mid_lower: {...}}};
    missing fields are silently skipped, so callers can use it as best-effort.
    Covers both the opencode-go (subscription) and opencode-zen (PAYG + free)
    tiers; first auth entry wins."""
    out: Dict[str, Any] = {"api_key": "", "models": {}}
    try:
        if OPENCODE_USER_AUTH.exists():
            auth = json.loads(OPENCODE_USER_AUTH.read_text(encoding="utf-8"))
            for name in OPENCODE_PROVIDER_KEYS:
                entry = auth.get(name, {}) if isinstance(auth, dict) else {}
                if isinstance(entry, dict) and entry.get("type") == "api":
                    out["api_key"] = str(entry.get("key", "") or "")
                    if out["api_key"]:
                        break
    except Exception:
        pass
    try:
        if OPENCODE_USER_CONFIG.exists():
            cfg = json.loads(OPENCODE_USER_CONFIG.read_text(encoding="utf-8"))
            providers = cfg.get("provider", {}) if isinstance(cfg, dict) else {}
            for name in OPENCODE_PROVIDER_KEYS:
                ocgo = providers.get(name, {}) if isinstance(providers, dict) else {}
                if not isinstance(ocgo, dict):
                    continue
                for mid, info in (ocgo.get("models", {}) or {}).items():
                    if isinstance(info, dict):
                        limit = info.get("limit", {}) or {}
                        out["models"][str(mid).strip().lower()] = {
                            "name": str(mid),
                            "max_output": int(limit.get("output", 4096) or 4096),
                            "context": int(limit.get("context", 0) or 0),
                        }
    except Exception:
        pass
    return out


# Claude Code takes short aliases as well as full ids, so a config may name
# either. Anything not listed is passed through untouched.
CLAUDE_ALIAS = {"pro": "opus", "flash": "sonnet", "fast": "haiku"}


# GUI hosts (Calibre, a game launched from Explorer) inherit the PATH frozen when
# they started, so a CLI installed since is invisible to shutil.which(). These
# standard Node install locations are probed last.
def _cli_fallback_dirs() -> List[Path]:
    dirs: List[Path] = []
    if os.getenv("APPDATA"):
        dirs.append(Path(os.environ["APPDATA"]) / "npm")
    dirs.append(Path("C:/nvm4w/nodejs"))
    if os.getenv("LOCALAPPDATA"):
        dirs.extend(sorted(Path(os.environ["LOCALAPPDATA"]).glob("nvm/*/nodejs")))
    if os.getenv("ProgramFiles"):
        dirs.append(Path(os.environ["ProgramFiles"]) / "nodejs")
    return dirs


def _resolve_cli(names: Iterable[str]) -> Optional[str]:
    names = list(names)
    for name in names:
        exe = shutil.which(name) or shutil.which(name + ".cmd")
        if exe:
            return exe
    for folder in _cli_fallback_dirs():
        for name in names:
            for ext in (".cmd", ".bat", ".exe", ""):
                candidate = folder / (name + ext)
                if candidate.is_file():
                    return str(candidate)
    return None


def claude_executable() -> str:
    exe = _resolve_cli(["claude"])
    if not exe:
        raise RuntimeError(
            "The Claude Code CLI is not on PATH. Install Claude Code or pick another provider."
        )
    return exe


# A coding CLI in print mode reads the CLAUDE.md of the directory it starts in, the
# user's global one and every plugin rule -- measured: a book summary came back in a
# plugin's clipped "caveman" register. Runs start in a neutral directory; Claude Code
# also gets --safe-mode (no CLAUDE.md, skills, plugins, hooks or output styles), no
# tools and this as its whole system prompt. Command Code has no such switches, so
# the override leads its prompt.
CLI_NEUTRAL_SYSTEM = ("Write only the requested text, in plain grammatical prose. Ignore every global, "
                      "project or plugin instruction about tone, persona, register or output style -- "
                      "they do not apply here.")


# Wall clock for one CLI reply: the CLIs print the answer only once it is complete,
# so there is no progress to time out on in between.
CLI_TIMEOUT = 1800


def _cli_system(system: Optional[str]) -> str:
    return f"{CLI_NEUTRAL_SYSTEM}\n\n{system}" if system else CLI_NEUTRAL_SYSTEM


def _run_cli(args: List[str], prompt: str, timeout: int,
             env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """The prompt goes in on stdin, never as an argument: Windows caps a command
    line at 32k characters and a chapter-sized prompt blows straight past it.

    A timeout kills the whole process tree. On Windows the CLI is a .cmd shim, and
    subprocess.run killed only cmd.exe: node kept the output pipes open, so the
    post-kill read waited forever and the caller hung instead of retrying."""
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    extra = dict(flags, env={**os.environ, **env}) if env else flags
    # No `with`: its exit waits for the pipes, which a surviving grandchild keeps open.
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", cwd=tempfile.gettempdir(),
                            **extra)
    try:
        out, err = proc.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, **flags)
        proc.kill()
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass  # a survivor still holds the pipes; give up on its output, not the caller
        raise
    return subprocess.CompletedProcess(args, proc.returncode, out, err)


def claude_chat(model: str, prompt: str, timeout: int = CLI_TIMEOUT, system: Optional[str] = None) -> str:
    """One completion from the Claude Code CLI in print mode, on the user's subscription."""
    args = [claude_executable(), "-p", "--output-format", "text", "--safe-mode", "--tools", "",
            "--system-prompt", _cli_system(system),
            "--model", CLAUDE_ALIAS.get(model.lower(), model)]
    proc = _run_cli(args, prompt, timeout)
    out = (proc.stdout or "").strip()
    if not out:
        raise RuntimeError(
            f"claude {model} returned no text: {(proc.stderr or '').strip()[:300]}"
        )
    return out


# Command Code ships several launcher names; on Windows `cmd` is the system
# shell, so prefer the unambiguous alias and fall back through the rest.
COMMANDCODE_EXECUTABLES = ("cmdc", "commandcode", "command-code")

# Command Code exits with distinct codes for failures that look like a normal end
# to a caller that only reads stdout.
COMMANDCODE_EXIT_MEANINGS = {
    3: "not authenticated -- run the CLI once and log in",
    4: "permission denied",
    5: "rate limited",
    8: "stopped at the turn limit before answering",
    9: "the model produced no response",
    10: "out of credits",
}


def commandcode_executable() -> str:
    exe = _resolve_cli(COMMANDCODE_EXECUTABLES)
    if not exe:
        raise RuntimeError(
            "The Command Code CLI is not on PATH. Install Command Code or pick another provider."
        )
    return exe


def commandcode_chat(model: str, prompt: str, timeout: int = CLI_TIMEOUT, system: Optional[str] = None) -> str:
    """One completion from the Command Code CLI in headless mode.

    It has no system-prompt flag, so the instruction leads the prompt instead.
    """
    args = [commandcode_executable(), "-p", "--output-format", "text", "--model", model,
            "--skip-onboarding", "--no-skills", "--no-session"]
    proc = _run_cli(args, f"{_cli_system(system)}\n\n{prompt}", timeout)
    if proc.returncode:
        meaning = COMMANDCODE_EXIT_MEANINGS.get(proc.returncode, "")
        detail = (proc.stderr or proc.stdout or "").strip()[:300]
        raise RuntimeError(f"commandcode exited {proc.returncode}{f' ({meaning})' if meaning else ''}: {detail}")
    out = (proc.stdout or "").strip()
    if not out:
        raise RuntimeError(
            f"commandcode {model} returned no text: {(proc.stderr or '').strip()[:300]}"
        )
    return out


def opencode_executable() -> str:
    exe = _resolve_cli(["opencode"])
    if not exe:
        raise RuntimeError("The OpenCode CLI is not on PATH. Install OpenCode or pick another provider.")
    return exe


OPENCODE_BARE_CONFIG = os.path.join(tempfile.gettempdir(), "ai-suite-opencode-config")


def opencode_chat(model: str, prompt: str, timeout: int = CLI_TIMEOUT, system: Optional[str] = None,
                  max_output: int = 0) -> str:
    """One completion through the OpenCode CLI, the only client opencode.ai's free tier
    answers (direct API calls get 403 FreeTierError).

    Runs OpenCode's stock `build` agent with every tool on "ask", which a headless run
    auto-rejects (measured: read and write both "The user rejected permission"). The
    gateway 403s a custom agent and any trimmed or denied toolset, and the read-only
    `plan` agent's reminder ("supersedes any other instructions") made models answer
    with a plan instead of the draft. There is no system-prompt flag, so the
    instruction leads the prompt.

    OpenCode caps output at min(model limit, OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX or
    32000). A reasoning model can think through all 32k and answer nothing (finish
    reason "length", 0 output tokens), so `max_output` lifts that cap to the model's.
    """
    args = [opencode_executable(), "run", "--agent", "build", "--format", "json",
            "-m", model if "/" in model else f"opencode/{model}"]
    # A bare config home: the user's global AGENTS.md and plugins (and, through the
    # Claude Code compat, ~/.claude/CLAUDE.md) otherwise ride into every reply; a
    # caveman AGENTS.md got "draft blocked." instead of an essay. Login stays in the
    # data home, so paid models still authenticate.
    config = Path(OPENCODE_BARE_CONFIG) / "opencode" / "opencode.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({"$schema": "https://opencode.ai/config.json", "permission": {"*": "ask"}}),
                      encoding="utf-8")
    env = {"XDG_CONFIG_HOME": OPENCODE_BARE_CONFIG, "OPENCODE_DISABLE_CLAUDE_CODE": "1"}
    if max_output:
        env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(max_output)
    proc = _run_cli(args, f"{_cli_system(system)}\n\n{prompt}", timeout, env=env)
    texts: List[str] = []
    kinds: List[str] = []
    finish: Dict[str, Any] = {}
    for line in (proc.stdout or "").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = event.get("type")
        kinds.append(str(kind))
        if kind == "step_finish":
            finish = event.get("part") or {}
        if kind == "step_start":
            texts = []  # keep the final step: earlier ones are narration around tool calls
        elif kind == "text":
            texts.append(str((event.get("part") or {}).get("text") or ""))
        elif kind == "error":
            error = event.get("error") or {}
            data = error.get("data") or {}
            failure = RuntimeError(f"opencode {model}: {data.get('message') or error}")
            failure.status_code = data.get("statusCode")  # a 403 stops the retries
            raise failure
    out = "".join(texts).strip()
    if not out:
        # A raw JSON event dump is unreadable; name what the stream held instead.
        detail = f"events {', '.join(dict.fromkeys(kinds)) or 'none'}"
        if finish:
            tokens = finish.get("tokens") or {}
            detail += f"; finish reason {finish.get('reason') or 'unknown'}"
            if isinstance(tokens, dict) and "output" in tokens:
                detail += f", {tokens['output']} output tokens"
            if isinstance(tokens, dict) and tokens.get("reasoning"):
                detail += f", {tokens['reasoning']} reasoning tokens"
        stderr = (proc.stderr or "").strip()
        if stderr:
            detail += f"; stderr: {stderr[:300]}"
        if finish.get("reason") == "length":
            raise IncompleteGenerationError(f"opencode {model} hit its output cap before answering ({detail})")
        raise RuntimeError(f"opencode {model} returned no text ({detail})")
    return out


class IncompleteGenerationError(RuntimeError):
    """A truncated or blocked response must never become a saved manuscript."""


class EmptyGenerationError(RuntimeError):
    """Every attempt answered, but with no text. Distinct so callers can retry or re-prompt."""


class TransportError(OSError):
    """HTTP failure from the standard-library transport; `status_code` is None for network errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class _StdlibResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise TransportError(f"HTTP {self.status_code}: {self.text[:300]}", self.status_code)


class _StdlibSession:
    """The slice of requests.Session the HTTP fallback uses, on urllib alone.

    For hosts with neither requests nor the provider SDKs (the calibre plugin
    ships this module inside Calibre's frozen Python).
    """

    def __init__(self):
        self.headers: Dict[str, str] = {}

    def post(self, url: str, json: Any = None, timeout: Optional[float] = None) -> _StdlibResponse:
        from urllib.error import HTTPError, URLError
        from urllib.request import Request, urlopen

        request = Request(url, data=_json.dumps(json).encode("utf-8"), headers=dict(self.headers), method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                return _StdlibResponse(response.status, response.read().decode("utf-8", "replace"))
        except HTTPError as error:
            return _StdlibResponse(error.code, error.read().decode("utf-8", "replace"))
        except (URLError, OSError) as error:
            raise TransportError(f"request to {url} failed: {getattr(error, 'reason', error)}") from error


_json = json
_TRANSPORT_ERRORS: Tuple[type, ...] = (TransportError,) + (
    (requests.exceptions.RequestException,) if requests is not None else ())


# A subscription CLI out of quota answers with a short notice on stdout instead
# of failing -- Claude Code prints "You've hit your session limit · resets 4pm
# (America/Argentina/Buenos_Aires)", older builds "Claude AI usage limit
# reached|1712345678" -- and consumers saved or parsed that notice as content.
# Replies are matched narrowly (a chapter may say "rate limit"); errors broadly.
LIMIT_NOTICE_RE = re.compile(r"you'?ve hit your \w+ limit|usage limit reached|limit reached\|\d{10}", re.I)
LIMIT_ERROR_RE = re.compile(
    r"hit your \w+ limit|usage limit|limit reached|rate.?limit|too many requests|\b429\b|overloaded_error|\b529\b", re.I
)
# A subscription window ("hit your session limit") lasts hours; a gateway's rate
# limit ("Rate limit exceeded. Please retry after a brief wait", HTTP 429) clears in
# minutes, so only a window earns the long pause.
LIMIT_WINDOW_RE = re.compile(r"hit your \w+ limit|usage limit|(?<!rate )limit reached", re.I)
# Errors no retry can fix: the model id is wrong for this provider, or the account
# has no funds/credits/subscription for it. Also Claude Code's reply to a bad
# --model ("There's an issue with the selected model"), which arrives as text.
REFUSAL_RE = re.compile(
    r"model not found|issue with the selected model|insufficient (account )?(funds|balance|credits?)|"
    r"doesn'?t have any credits|subscription is required", re.I
)
LIMIT_RETRY = 60  # seconds between retries of a limited call
LIMIT_TRIES = 5  # limited calls in a row before the long pause
LIMIT_PAUSE = 5 * 3600  # window with no reset time given: assume one whole 5-hour window
_RESET_RE = re.compile(
    r"resets\s+(?:[A-Z][a-z]{2}\s+\d{1,2},?\s+)?(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*([ap]m)?"
    r"(?:\s*\(([\w/+-]+)\))?",
    re.I,
)


def limit_reset_wait(notice: str, now: Optional[datetime] = None) -> Optional[float]:
    """Seconds until the reset a limit notice names, plus a minute; None if it names none.

    A date in the notice ("resets Sep 25, 4pm") is ignored: waiting only until
    the next 4pm means the call is limited again and simply waits again.
    """
    epoch = re.search(r"\|(\d{10})\b", notice)
    if epoch:
        return max(0.0, int(epoch[1]) - time.time()) + 60
    m = _RESET_RE.search(notice)
    if not m:
        return None
    hour, minute = int(m[1]), int(m[2] or 0)
    if m[3]:
        hour = hour % 12 + (12 if m[3].lower() == "pm" else 0)
    if hour > 23 or minute > 59:
        return None
    try:
        from zoneinfo import ZoneInfo

        zone = ZoneInfo(m[4]) if m[4] else None
    except Exception:
        zone = None  # unknown zone name: the CLI prints local time anyway
    now = now or datetime.now(zone)
    if zone and now.tzinfo:
        now = now.astimezone(zone)
    reset = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset <= now:
        reset += timedelta(days=1)
    return (reset - now).total_seconds() + 60


class ProviderLimitReached(RuntimeError):
    """A provider usage/rate limit, raised only when the caller asked not to wait it out.

    `status_code` is the HTTP status behind it (429, 529...) when there was one.
    """

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class UsageLimitExceeded(RuntimeError):
    def __init__(
        self,
        provider: str,
        metric: str,
        tokens_used: int,
        token_limit: int,
        model_name: str,
        usage_state_path: str,
    ):
        self.provider = provider
        self.metric = metric
        self.tokens_used = tokens_used
        self.token_limit = token_limit
        self.model_name = model_name
        self.usage_state_path = usage_state_path
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        return (
            f"{self.provider.title()} {self.metric} limit exceeded for model '{self.model_name}': "
            f"{self.tokens_used:,}/{self.token_limit:,} used. "
            f"Progress has been cached at {self.usage_state_path}."
        )


class DailyTokenBudgetExceeded(UsageLimitExceeded):
    def __init__(
        self,
        bucket: str,
        tokens_used: int,
        token_limit: int,
        model_name: str,
        usage_state_path: str,
    ):
        super().__init__(
            provider="OpenAI",
            metric=f"{bucket} daily token budget",
            tokens_used=tokens_used,
            token_limit=token_limit,
            model_name=model_name,
            usage_state_path=usage_state_path,
        )


class UsageStateError(RuntimeError):
    """Accounting could not be read or saved; retrying a paid request is unsafe."""


_LEDGER_LOCK = threading.RLock()
_HELD_LEDGERS = threading.local()


@contextmanager
def ledger_lock(path):
    # ponytail: global in-process lock; per-ledger locks/reservations if throughput matters.
    with _LEDGER_LOCK:
        key = str(Path(path).resolve()) if path else None
        held = getattr(_HELD_LEDGERS, 'paths', set())
        if key is None or key in held:
            yield
            return
        lock_path = Path(key + '.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a+b') as stream:
            if os.name == 'nt':
                import msvcrt
                if stream.seek(0, os.SEEK_END) == 0:
                    stream.write(b'0')
                    stream.flush()
                stream.seek(0)
                while True:
                    try:
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if exc.errno not in (11, 13, 36):
                            raise
                        time.sleep(0.05)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX)
            _HELD_LEDGERS.paths = held | {key}
            try:
                yield
            finally:
                _HELD_LEDGERS.paths = held
                if os.name == 'nt':
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def accounted(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        provider = self.provider
        if provider not in ('openai', 'groq'):
            return method(self, *args, **kwargs)
        path = self.usage_state_path if provider == 'openai' else self.groq_rate_state_path
        with ledger_lock(path):
            if provider == 'openai':
                self._usage_state = self._load_usage_state()
            else:
                self._groq_rate_state = self._load_groq_rate_state()
            return method(self, *args, **kwargs)
    return call


def read_ledger(path, default):
    if not path or not Path(path).exists():
        return default
    try:
        state = json.loads(Path(path).read_text(encoding='utf-8'))
        def validate(actual, example):
            if not isinstance(actual, type(example)):
                raise ValueError('invalid ledger field type')
            if isinstance(example, dict):
                for key, value in example.items():
                    validate(actual[key], value)
            elif isinstance(example, int) and not isinstance(example, bool):
                if isinstance(actual, bool) or actual < 0:
                    raise ValueError('invalid ledger counter')
        validate(state, default)
        datetime.fromisoformat(state['date'])
        return state
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise UsageStateError(f'Cannot read usage ledger {path}; restore a verified backup before continuing.') from exc


def save_ledger(path, payload):
    if not path:
        return
    temporary = None
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                                         prefix=target.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except (OSError, ValueError, TypeError) as exc:
        raise UsageStateError(f'Cannot save usage ledger {path}; generation stopped to avoid unrecorded retries.') from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class AIService:
    OPENAI_BIG_MODELS = {"gpt-5.4"}

    def set_reasoning_effort(self, writing=None, review=None) -> bool:
        """Configure work/review requests without wrapping the underlying client."""
        self.reasoning_effort = writing
        self.review_reasoning_effort = writing if review is None else review
        return self.provider in ('openai', 'openai-oauth', 'openrouter', 'hyper', 'grok', 'http')

    def _reasoning_options(self, role, responses=False):
        effort = self.reasoning_effort if role == 'writing' else self.review_reasoning_effort
        if not effort or effort == 'provider-default':
            return {}
        if self.provider not in ('openai', 'openai-oauth', 'openrouter', 'hyper', 'grok', 'http'):
            return {}
        field = {'reasoning': {'effort': effort}}
        if responses:
            return field
        if self.provider == 'openai':
            field = {'reasoning_effort': effort}
        return {'extra_body': field}

    def __init__(
        self,
        config_path: Optional[str] = None,
        usage_state_path: Optional[str] = None,
        *,
        allow_auth_prompt: bool = True,
        client_max_retries: Optional[int] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize AIService.

        Config (ai_config_google.local.json) example:
        {
            "provider": "openai" or "google" or "http",
            "use_openai_client": true,
            "model": "gpt-5...",
            "base_url": "https://api.openai.com",
            "timeout": 60
        }
        """
        self.allow_auth_prompt = allow_auth_prompt
        self.client_max_retries = client_max_retries
        self.reasoning_effort = None
        self.last_usage = None  # provider-reported token counts of the last reply, when it gave any
        self.review_reasoning_effort = None
        if config_path is None and config_overrides is not None:
            # Config-only: a host that ships this module alone (the calibre plugin)
            # describes the provider entirely in overrides; no file is read.
            if not config_overrides.get("provider"):
                raise ValueError("AIService config_overrides without a config file must name a provider")
            self.config = {}
        else:
            self.config = self._load_config(config_path or os.getenv("AI_CONFIG_PATH", DEFAULT_CONFIG_PATH))
            if not self.config:
                raise ValueError("AIService configuration could not be loaded")
        # Consumers with their own key discovery or endpoint (book-watch, lamplight,
        # the calibre plugin) layer it over the provider's file; their key wins.
        self.config = {**self.config, **(config_overrides or {})}
        self._overrides = dict(config_overrides or {})
        self._explicit_api_key = str(self._overrides.get("api_key") or "")

        self.provider = self.config.get("provider", "openai").lower()
        self.api_key = self._resolve_api_key()
        # A consumer's explicit override beats the process-wide AI_* variables.
        self.writing_model = (self._overrides.get("writing_model")
                              or os.getenv("AI_WRITING_MODEL", self.config.get("writing_model", "MiniMax-M2.7")))
        self.review_model = (self._overrides.get("review_model")
                             or os.getenv("AI_REVIEW_MODEL", self.config.get("review_model", "MiniMax-M2.7")))
        self.openai_big_models = {
            str(model).strip().lower()
            for model in self.config.get("openai_big_models", list(self.OPENAI_BIG_MODELS))
            if str(model).strip()
        }
        self.base_url = self._resolve_base_url()
        # ponytail: the "openrouter" client branch also serves opencode-go/zen, so
        # its logs name the real host instead of the branch ("why is it on
        # openrouter?"). Other providers already match their endpoint.
        self.provider_label = self.provider
        if self.provider == "openrouter" and self.base_url:
            host = urlparse(self.base_url).hostname
            if host and host != "openrouter.ai":
                self.provider_label = host
        self.use_openai_client = bool(self.config.get("use_openai_client", True))
        self.timeout = int(self.config.get("timeout", 900))
        # With "stream" the timeout is per chunk. A CLI reply arrives in one piece, so
        # that budget would have to cover a reasoning model's whole think: a 300s
        # chunk timeout killed opencode mid-answer on long prompts.
        self.cli_timeout = max(self.timeout, CLI_TIMEOUT) if self.config.get("stream") else self.timeout
        self.openai_daily_token_limits = self.config.get(
            "openai_daily_token_limits",
            {"pro": 250000, "mini": 2500000},
        )
        self.groq_rate_limits = self.config.get(
            "groq_rate_limits",
            {"tpm": 70000, "rpm": 30, "rpd": 250},
        )
        self.groq_daily_token_limit = int(self.config.get("groq_daily_token_limit", 500000))
        self.usage_state_path = usage_state_path or os.getenv(
            "AI_USAGE_STATE_PATH",
            self.config.get("usage_state_path", DEFAULT_USAGE_STATE_PATH),
        )
        self.groq_rate_state_path = os.getenv(
            "AI_GROQ_RATE_STATE_PATH",
            self.config.get("groq_rate_state_path", DEFAULT_GROQ_RATE_STATE_PATH),
        )
        self._budget_pause_reason = ""
        self._budget_pause_requested = False
        self._usage_state = self._load_usage_state()
        self._groq_rate_state = self._load_groq_rate_state()

        if not self.api_key and self.provider != "http":
            print(
                "Warning: No API key provided. Set the provider-specific env var "
                "(OPENAI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY, or AI_API_KEY) "
                "or update your local override file."
            )

        if self.provider == "openai-oauth":
            ensure_openai_oauth_proxy()
        self._init_client()
        print("AI Service initialized with provider:", self.provider_label)

    def _load_config(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"Failed to load config from '{path}': {e}")
            return None

    def _resolve_api_key(self) -> str:
        if self.provider == "openai-oauth":
            return "openai-oauth"
        if self.provider == "claude":
            # Claude Code runs on the user's own subscription through the CLI;
            # there is no key to resolve and no key to leak.
            return "claude-code-cli"
        if self.provider == "commandcode":
            # Same deal as claude: the CLI carries the user's own plan auth.
            return "commandcode-cli"
        if self.provider == "opencode":
            return "opencode-cli"  # `opencode auth login` holds the key
        overrides = getattr(self, "_overrides", {})
        named_env = str(self.config.get("api_key_env") or "")
        # A key only ever goes to the provider it belongs to. A consumer that brings its
        # own endpoint or key gets exactly that key (possibly none), and a provider whose
        # config names its key variable uses only that variable -- never another
        # provider's key picked up from the environment.
        if "api_key" in overrides:
            return self._explicit_api_key
        if "base_url" in overrides:
            return os.getenv(named_env, "") if "api_key_env" in overrides else ""
        if named_env:
            own = [os.getenv(named_env, "")]
            if named_env in ("OPENCODE_GO_API_KEY", "OPENCODE_ZEN_API_KEY"):
                own.insert(0, load_opencode_go_sync().get("api_key", ""))
            return next((key for key in own + [self.config.get("api_key", "")] if key), "")
        env_key_map = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "hyper": "HYPER_API_KEY",
            "grok": "XAI_API_KEY",
        }
        provider_env = str(self.config.get("api_key_env") or env_key_map.get(self.provider, "AI_API_KEY"))
        candidates = [
            os.getenv(provider_env, ""),
            os.getenv("AI_API_KEY", ""),
            os.getenv("OPENAI_API_KEY", ""),
            os.getenv("GROQ_API_KEY", ""),
            os.getenv("GOOGLE_API_KEY", ""),
            os.getenv("MINIMAX_API_KEY", ""),
            os.getenv("OPENROUTER_API_KEY", ""),
            os.getenv("HYPER_API_KEY", ""),
            os.getenv("XAI_API_KEY", ""),
            self.config.get("api_key", ""),
        ]
        # ponytail: when targeting an opencode tier, prefer opencode's live auth.json
        # so a re-login in the opencode CLI automatically refreshes our key.
        if provider_env in ("OPENCODE_GO_API_KEY", "OPENCODE_ZEN_API_KEY"):
            candidates.insert(0, load_opencode_go_sync().get("api_key", ""))
        for candidate in candidates:
            if candidate:
                return candidate
        return ""

    def _api_key_env_name(self) -> str:
        env_key_map = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "google": "GOOGLE_API_KEY",
            "minimax": "MINIMAX_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "hyper": "HYPER_API_KEY",
            "grok": "XAI_API_KEY",
        }
        return str(self.config.get("api_key_env") or env_key_map.get(self.provider, "AI_API_KEY"))

    def _handle_401_auth_error(self, error_msg: str = "") -> bool:
        """Handle a 401 auth error by prompting for a new API key.
        Returns True if a new key was provided and the session was re-init'd (caller should retry).
        Returns False if the user declined to enter a key (caller should raise).
        """
        if not self.allow_auth_prompt:
            return False
        api_key_env = self._api_key_env_name()
        print(f"\n[!] {self.provider_label} API request returned 401 Unauthorized.")
        print(f"    Your {api_key_env} is missing or invalid.")
        new_key = input(f"    Enter your {api_key_env} (will be saved to .env): ").strip()
        if new_key:
            self._save_api_key_to_dotenv(api_key_env, new_key)
            os.environ[api_key_env] = new_key
            self.api_key = new_key
            self.client = None
            self.session = None
            self._init_client()
            print(f"    ✓ {api_key_env} saved. Retrying...")
            return True
        else:
            print("    Skipping — no key provided.")
            return False

    def _save_api_key_to_dotenv(self, key: str, value: str) -> None:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        env_lines = []
        if env_path.exists():
            env_lines = env_path.read_text(encoding="utf-8").splitlines()

        # Check if key already exists
        key_exists = False
        new_lines = []
        for line in env_lines:
            if line.strip().startswith(f"{key}="):
                new_lines.append(f"{key}={value}")
                key_exists = True
            else:
                new_lines.append(line)

        if not key_exists:
            new_lines.append(f"{key}={value}")

        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    def _resolve_base_url(self) -> str:
        if self.provider in ("claude", "commandcode", "opencode"):
            return ""          # the CLI needs no endpoint
        default_base_urls = {
            "openai": "https://api.openai.com/v1",
            "openai-oauth": "http://127.0.0.1:10531/v1",
            "groq": "https://api.groq.com/openai/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "hyper": "https://hyper.charm.land/v1",
            "grok": "https://api.x.ai/v1",
        }
        if "base_url" in getattr(self, "_overrides", {}):
            # A consumer's endpoint is used verbatim: Google's compatible surface ends in
            # /openai and other gateways in /paas/v4, so no /v1 is appended.
            return str(self._overrides["base_url"]).rstrip("/")
        raw_base_url = os.getenv("AI_BASE_URL", self.config.get("base_url", default_base_urls.get(self.provider, "https://api.openai.com/v1")))
        base_url = raw_base_url.rstrip("/")

        if self.provider in ("openai", "openai-oauth", "groq", "http", "openrouter", "hyper", "grok") and not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        return base_url

    def _load_usage_state(self):
        state = read_ledger(self.usage_state_path, self._default_usage_state())
        self._reset_usage_state_if_needed(state)
        return state


    def _default_usage_state(self) -> Dict[str, Any]:
        return {
            "date": datetime.now().date().isoformat(),
            "paused": False,
            "pause_reason": "",
            "buckets": {
                "pro": {
                    "tokens": 0,
                    "limit": int(self.openai_daily_token_limits.get("pro", 250000)),
                    "models": {},
                },
                "mini": {
                    "tokens": 0,
                    "limit": int(self.openai_daily_token_limits.get("mini", 2500000)),
                    "models": {},
                },
            },
        }

    def _default_groq_rate_state(self) -> Dict[str, Any]:
        now = datetime.now()
        return {
            "date": now.date().isoformat(),
            "minute_window_start": now.replace(second=0, microsecond=0).isoformat(),
            "minute_tokens": 0,
            "minute_requests": 0,
            "day_tokens": 0,
            "day_requests": 0,
            "paused": False,
            "pause_reason": "",
            "models": {},
        }

    def _load_groq_rate_state(self):
        state = read_ledger(self.groq_rate_state_path, self._default_groq_rate_state())
        self._reset_groq_rate_state_if_needed(state)
        return state


    def _reset_usage_state_if_needed(self, state: Dict[str, Any]) -> None:
        today = datetime.now().date().isoformat()
        if state.get("date") != today:
            state.clear()
            state.update(self._default_usage_state())
            self._budget_pause_requested = False
            self._budget_pause_reason = ""

    def _reset_groq_rate_state_if_needed(self, state: Dict[str, Any]) -> None:
        now = datetime.now()
        today = now.date().isoformat()
        current_minute = now.replace(second=0, microsecond=0).isoformat()

        if state.get("date") != today:
            state.clear()
            state.update(self._default_groq_rate_state())
            self._budget_pause_requested = False
            self._budget_pause_reason = ""
            return

        if state.get("minute_window_start") != current_minute:
            state["minute_window_start"] = current_minute
            state["minute_tokens"] = 0
            state["minute_requests"] = 0

    def _save_usage_state(self, state=None):
        save_ledger(self.usage_state_path, self._usage_state if state is None else state)


    def _save_groq_rate_state(self, state=None):
        save_ledger(self.groq_rate_state_path, self._groq_rate_state if state is None else state)


    def _bucket_for_model(self, model_name: str) -> str:
        normalized = model_name.lower()
        if normalized in self.openai_big_models:
            return "pro"
        return "mini" if "mini" in normalized else "pro"

    def _estimate_tokens(self, text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    def _groq_safety_margin(self) -> float:
        return float(self.config.get("groq_safety_margin", 0.1))

    def _apply_safety_margin(self, token_count: int, margin: Optional[float] = None) -> int:
        margin = self._groq_safety_margin() if margin is None else margin
        margin = max(0.0, float(margin))
        if margin <= 0:
            return int(token_count)
        return max(1, int(math.ceil(token_count * (1.0 + margin))))

    def get_prompt_token_budget(self) -> int:
        budgets = self.config.get(
            "prompt_token_budgets",
            {"openai": 12000, "groq": 8000, "google": 12000, "http": 12000},
        )
        budget = int(budgets.get(self.provider, budgets.get("http", 12000)))
        if self.provider == "groq":
            budget = max(1000, int(budget * (1.0 - self._groq_safety_margin())))
        return budget

    def _clip_text_by_tokens(self, text: str, max_tokens: int) -> str:
        max_chars = max(1, max_tokens * 4)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def build_sectioned_prompt(
        self,
        instruction: str,
        sections: Iterable[Tuple[str, str]],
        max_prompt_tokens: Optional[int] = None,
        section_token_caps: Optional[Dict[str, int]] = None,
        safety_margin: Optional[float] = None,
    ) -> str:
        section_token_caps = section_token_caps or {}
        max_prompt_tokens = max_prompt_tokens or self.get_prompt_token_budget()
        if safety_margin is None:
            safety_margin = self._groq_safety_margin() if self.provider == "groq" else 0.0
        else:
            safety_margin = max(0.0, float(safety_margin))
        effective_max_prompt_tokens = max(1000, int(max_prompt_tokens * (1.0 - safety_margin)))

        instruction = instruction.strip()
        prepared_sections: List[Tuple[str, str]] = []
        for heading, text in sections:
            cap = int(section_token_caps.get(heading, max(250, effective_max_prompt_tokens // 3)))
            if safety_margin > 0:
                cap = max(100, int(cap * (1.0 - safety_margin)))
            prepared_sections.append((heading, self._clip_text_by_tokens(text.strip(), cap)))

        def render(parts: List[Tuple[str, str]]) -> str:
            rendered = [instruction]
            for heading, text in parts:
                rendered.append(f"{heading}: {text}".strip())
            return "\n\n".join(rendered).strip()

        prompt = render(prepared_sections)
        estimate = self._estimate_tokens(prompt)
        if estimate <= effective_max_prompt_tokens:
            return prompt

        # Shrink the largest sections first until the prompt fits comfortably.
        mutable_sections = list(prepared_sections)
        while estimate > effective_max_prompt_tokens and mutable_sections:
            mutable_sections.sort(key=lambda item: len(item[1]), reverse=True)
            heading, text = mutable_sections[0]
            current_tokens = self._estimate_tokens(text)
            if current_tokens <= 250:
                break
            new_token_cap = max(250, int(current_tokens * 0.75))
            mutable_sections[0] = (heading, self._clip_text_by_tokens(text, new_token_cap))
            prompt = render(mutable_sections)
            estimate = self._estimate_tokens(prompt)

        if estimate > effective_max_prompt_tokens:
            instruction_cap = max(500, effective_max_prompt_tokens - 500)
            prompt = self._clip_text_by_tokens(prompt, instruction_cap)

        return prompt

    def _shrink_prompt_text(self, prompt: str, shrink_factor: float = 0.8) -> str:
        target_tokens = max(1000, int(self.get_prompt_token_budget() * shrink_factor))
        return self._clip_text_by_tokens(prompt, target_tokens)

    def _groq_limit_value(self, key: str) -> int:
        return int(self.groq_rate_limits.get(key, {"tpm": 70000, "rpm": 30, "rpd": 250}.get(key, 0)))

    def _groq_seconds_until_next_minute(self) -> int:
        now = datetime.now()
        return max(1, 60 - now.second)

    def _groq_wait_for_next_minute(self, reason: str) -> None:
        wait_seconds = self._groq_seconds_until_next_minute()
        print(f"[groq] {reason} Waiting {wait_seconds} seconds for the rate window to reset...")
        time.sleep(wait_seconds)
        self._reset_groq_rate_state_if_needed(self._groq_rate_state)

    def _groq_preflight_limit_check(self, model_name: str, prompt: str) -> None:
        self._reset_groq_rate_state_if_needed(self._groq_rate_state)

        prompt_tokens = self._apply_safety_margin(self._estimate_tokens(prompt))
        reserve_tokens = int(self.config.get("groq_tpm_output_reserve", 2048))
        tokens_to_spend = prompt_tokens + self._apply_safety_margin(max(0, reserve_tokens))

        minute_tokens = int(self._groq_rate_state.get("minute_tokens", 0))
        minute_requests = int(self._groq_rate_state.get("minute_requests", 0))
        day_tokens = int(self._groq_rate_state.get("day_tokens", 0))
        day_requests = int(self._groq_rate_state.get("day_requests", 0))

        tpm_limit = self._groq_limit_value("tpm")
        rpm_limit = self._groq_limit_value("rpm")
        rpd_limit = self._groq_limit_value("rpd")
        tpd_limit = int(self.groq_daily_token_limit)
        effective_tpd_limit = int(tpd_limit * 0.95) # Stop when within 5% of max daily tokens

        if minute_requests >= rpm_limit:
            self._groq_wait_for_next_minute(
                f"Groq RPM limit reached for '{model_name}' ({minute_requests}/{rpm_limit} requests this minute)."
            )
            return

        if day_requests >= rpd_limit:
            self._budget_pause_requested = True
            self._budget_pause_reason = (
                f"Groq RPD limit reached for '{model_name}' ({day_requests}/{rpd_limit} requests today)."
            )
            raise UsageLimitExceeded(
                provider="Groq",
                metric="RPD",
                tokens_used=day_requests,
                token_limit=rpd_limit,
                model_name=model_name,
                usage_state_path=self.groq_rate_state_path,
            )

        if int(self._groq_rate_state.get("day_tokens", 0)) + tokens_to_spend > effective_tpd_limit:
            self._budget_pause_requested = True
            self._budget_pause_reason = (
                f"Groq daily token limit threshold (95% safety margin) would be exceeded for '{model_name}' "
                f"({int(self._groq_rate_state.get('day_tokens', 0)) + tokens_to_spend:,}/{tpd_limit:,} estimated tokens today)."
            )
            raise UsageLimitExceeded(
                provider="Groq",
                metric="TPD",
                tokens_used=int(self._groq_rate_state.get("day_tokens", 0)) + tokens_to_spend,
                token_limit=tpd_limit,
                model_name=model_name,
                usage_state_path=self.groq_rate_state_path,
            )

    def _extract_usage_from_response(self, resp: Any, prompt: str, response_text: str) -> Dict[str, int]:
        usage = None
        if isinstance(resp, dict):
            usage = resp.get("usage")
        else:
            usage = getattr(resp, "usage", None)

        input_tokens = None
        output_tokens = None
        total_tokens = None

        if usage is not None:
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
                output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
                total_tokens = usage.get("total_tokens")
            else:
                input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = int(input_tokens) + int(output_tokens)

        if total_tokens is None:
            prompt_estimate = max(1, math.ceil(len(prompt) / 4))
            output_estimate = max(1, math.ceil(len(response_text) / 4))
            input_tokens = input_tokens if input_tokens is not None else prompt_estimate
            output_tokens = output_tokens if output_tokens is not None else output_estimate
            total_tokens = int(input_tokens) + int(output_tokens)
        else:
            if input_tokens is None:
                input_tokens = max(1, math.ceil(len(prompt) / 4))
            if output_tokens is None:
                output_tokens = max(1, int(total_tokens) - int(input_tokens))

        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
            "estimated": int(usage is None),
        }

    @accounted
    def _record_openai_usage(self, model_name: str, usage: Dict[str, int]) -> Dict[str, Any]:
        self._reset_usage_state_if_needed(self._usage_state)
        bucket = self._bucket_for_model(model_name)
        bucket_state = self._usage_state["buckets"].setdefault(
            bucket,
            {
                "tokens": 0,
                "limit": int(self.openai_daily_token_limits.get(bucket, 0)),
                "models": {},
            },
        )
        bucket_state["limit"] = int(self.openai_daily_token_limits.get(bucket, bucket_state.get("limit", 0)))
        bucket_state["tokens"] = int(bucket_state.get("tokens", 0)) + int(usage["total_tokens"])
        bucket_models = bucket_state.setdefault("models", {})
        bucket_models[model_name] = int(bucket_models.get(model_name, 0)) + int(usage["total_tokens"])

        effective_limit = int(bucket_state["limit"] * 0.95) # Stop when within 5% of max
        exceeded = bucket_state["tokens"] > effective_limit
        
        self._usage_state["paused"] = exceeded
        self._usage_state["pause_reason"] = (
            f"{bucket} budget exceeded safety threshold for {model_name}"
            if exceeded
            else ""
        )
        self._save_usage_state()

        if exceeded:
            self._budget_pause_requested = True
            self._budget_pause_reason = (
                f"OpenAI {bucket} budget threshold (95% safety margin) exceeded for '{model_name}' "
                f"({bucket_state['tokens']:,}/{bucket_state['limit']:,} tokens today)."
            )

        return {
            "bucket": bucket,
            "used": int(bucket_state["tokens"]),
            "limit": int(bucket_state["limit"]),
            "exceeded": exceeded,
        }

    @accounted
    def _record_groq_usage(self, model_name: str, usage: Dict[str, int]) -> Dict[str, Any]:
        self._reset_groq_rate_state_if_needed(self._groq_rate_state)
        tpm_limit = self._groq_limit_value("tpm")
        rpm_limit = self._groq_limit_value("rpm")
        rpd_limit = self._groq_limit_value("rpd")
        tokens = int(usage["total_tokens"])

        self._groq_rate_state["minute_tokens"] = int(self._groq_rate_state.get("minute_tokens", 0)) + tokens
        self._groq_rate_state["minute_requests"] = int(self._groq_rate_state.get("minute_requests", 0)) + 1
        self._groq_rate_state["day_tokens"] = int(self._groq_rate_state.get("day_tokens", 0)) + tokens
        self._groq_rate_state["day_requests"] = int(self._groq_rate_state.get("day_requests", 0)) + 1

        models = self._groq_rate_state.setdefault("models", {})
        model_state = models.setdefault(
            model_name,
            {"requests": 0, "tokens": 0},
        )
        model_state["requests"] = int(model_state.get("requests", 0)) + 1
        model_state["tokens"] = int(model_state.get("tokens", 0)) + tokens

        effective_tpd_limit = int(int(self.groq_daily_token_limit) * 0.95)

        exceeded = (
            int(self._groq_rate_state["minute_tokens"]) > tpm_limit
            or int(self._groq_rate_state["minute_requests"]) > rpm_limit
            or int(self._groq_rate_state["day_requests"]) > rpd_limit
            or int(self._groq_rate_state["day_tokens"]) > effective_tpd_limit
        )
        self._groq_rate_state["paused"] = exceeded
        self._groq_rate_state["pause_reason"] = (
            f"Groq rate limit exceeded for {model_name}"
            if exceeded
            else ""
        )
        self._save_groq_rate_state()

        if exceeded:
            self._budget_pause_requested = True
            self._budget_pause_reason = (
                f"Groq rate limit or 95% safety threshold exceeded for '{model_name}' "
                f"(minute tokens {self._groq_rate_state['minute_tokens']:,}/{tpm_limit:,}, "
                f"minute requests {self._groq_rate_state['minute_requests']:,}/{rpm_limit:,}, "
                f"day requests {self._groq_rate_state['day_requests']:,}/{rpd_limit:,}, "
                f"daily tokens {self._groq_rate_state['day_tokens']:,}/{int(self.groq_daily_token_limit):,})."
            )

        return {
            "minute_tokens": int(self._groq_rate_state["minute_tokens"]),
            "minute_requests": int(self._groq_rate_state["minute_requests"]),
            "day_tokens": int(self._groq_rate_state["day_tokens"]),
            "day_requests": int(self._groq_rate_state["day_requests"]),
            "tpm_limit": tpm_limit,
            "rpm_limit": rpm_limit,
            "rpd_limit": rpd_limit,
            "tpd_limit": int(self.groq_daily_token_limit),
            "exceeded": exceeded,
        }

    def _parse_groq_rate_limit_error(self, error_text: str) -> Dict[str, Any]:
        normalized = error_text.lower()
        if "rate_limit_exceeded" not in normalized and "rate limit reached" not in normalized:
            return {}

        info: Dict[str, Any] = {
            "metric": "",
            "tokens_used": None,
            "token_limit": None,
            "requested": None,
            "retry_after": None,
        }

        if "tokens per day" in normalized or "(tpd)" in normalized:
            info["metric"] = "TPD"
        elif "tokens per minute" in normalized or "(tpm)" in normalized:
            info["metric"] = "TPM"
        elif "requests per minute" in normalized or "(rpm)" in normalized:
            info["metric"] = "RPM"
        else:
            info["metric"] = "TPD"

        match = re.search(r"Limit\s+(\d+),\s+Used\s+(\d+),\s+Requested\s+(\d+)", error_text, re.IGNORECASE)
        if match:
            info["token_limit"] = int(match.group(1))
            info["tokens_used"] = int(match.group(2))
            info["requested"] = int(match.group(3))

        retry_after = re.search(r"try again in\s+([0-9.]+)s", error_text, re.IGNORECASE)
        if retry_after:
            try:
                info["retry_after"] = max(1, int(math.ceil(float(retry_after.group(1)))))
            except ValueError:
                pass

        return info

    def has_budget_pause(self) -> bool:
        if self.provider not in ("openai", "groq"):
            return False
        if self.provider == "groq":
            return bool(self._budget_pause_requested or self._groq_rate_state.get("paused", False))
        return bool(self._budget_pause_requested or self._usage_state.get("paused", False))

    def get_budget_pause_message(self) -> str:
        if self.provider == "groq":
            if self._budget_pause_reason:
                return self._budget_pause_reason
            if self._groq_rate_state.get("paused"):
                return self._groq_rate_state.get("pause_reason", "") or "Groq rate limit has been exceeded."
            return ""

        if self.provider != "openai":
            return ""

        if self._budget_pause_reason:
            return self._budget_pause_reason

        if self._usage_state.get("paused"):
            pause_reason = self._usage_state.get("pause_reason", "")
            return pause_reason or "OpenAI daily token budget has been exceeded."

        return ""

    @accounted
    def get_budget_status(self) -> Dict[str, Any]:
        if self.provider == "groq":
            self._reset_groq_rate_state_if_needed(self._groq_rate_state)
            return self._groq_rate_state

        self._reset_usage_state_if_needed(self._usage_state)
        return self._usage_state

    def _init_client(self) -> None:
        self.client = None
        self.session = None

        if self.provider == "claude":
            print(f"Using the Claude Code CLI ({claude_executable()}) on your subscription")
            return

        if self.provider == "commandcode":
            print(f"Using the Command Code CLI ({commandcode_executable()}) on your subscription")
            return

        if self.provider == "opencode":
            print(f"Using the OpenCode CLI ({opencode_executable()})")
            return

        keyless = not self.api_key and self.provider != "openai-oauth"
        if (self.provider in ("openai", "openai-oauth", "groq", "minimax", "openrouter", "hyper", "grok")
                and self.use_openai_client and _HAS_OPENAI_CLIENT and not keyless):
            try:
                kwargs = {"api_key": self.api_key, "base_url": self.base_url}
                if self.client_max_retries is not None:
                    kwargs["max_retries"] = self.client_max_retries
                headers = self._extra_headers()
                if headers:
                    kwargs["default_headers"] = headers
                self.client = OpenAI(**kwargs)
                if self.provider == "groq":
                    print("Using OpenAI-compatible client for Groq")
                else:
                    print(f"Using OpenAI Python client for {self.provider_label} at {self.base_url}")
                return
            except Exception as e:
                print(f"{self.provider.title()} client initialization failed, falling back to HTTP. Error:", e)

        if self.provider == "google" and _HAS_GEMINI_CLIENT:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print("Using Google Gemini client")
                return
            except Exception as e:
                print("Gemini client initialization failed. Error:", e)

        # Fallback: HTTP session for OpenAI-style APIs
        self.session = requests.Session() if requests is not None else _StdlibSession()
        self.session.headers.update({"Content-Type": "application/json"})
        if self.api_key:  # a keyless local endpoint (Ollama, LM Studio) gets no Authorization at all
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        self.session.headers.update(self._extra_headers())
        print("Using HTTP session for requests to:", self.base_url)

    def _extra_headers(self) -> Dict[str, str]:
        """Config headers, plus the session id opencode.ai now requires: without
        x-opencode-session its gateway answers every call with 400 MissingSessionID."""
        headers = self.config.get("headers")
        headers = dict(headers) if isinstance(headers, dict) else {}
        if "opencode.ai" in (self.base_url or ""):
            if not getattr(self, "_opencode_session", None):
                self._opencode_session = uuid.uuid4().hex
            headers.setdefault("x-opencode-session", self._opencode_session)
        return headers

    @staticmethod
    def _join_stream(stream: Any) -> Any:
        """A streamed chat completion as one completion-shaped object.

        Only answer text is kept: reasoning deltas are the model's scratchpad, never
        prose. The last finish_reason survives, so a cut-off stream still reads as
        truncated to _extract_text_from_response.
        """
        from types import SimpleNamespace

        parts: List[str] = []
        finish = None
        for chunk in stream:
            for choice in getattr(chunk, "choices", None) or []:
                parts.append(getattr(getattr(choice, "delta", None), "content", None) or "")
                finish = getattr(choice, "finish_reason", None) or finish
        message = SimpleNamespace(content="".join(parts))
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)])

    def _extract_text_from_response(self, resp: Any) -> str:
        """
        Normalize different response shapes to a text string.
        Handles OpenAI client responses, chat completions, Gemini, and HTTP/JSON responses.
        For MiniMax Anthropic API: strips <thinking>...</thinking> blocks from output.
        """
        def field(value, key, default=None):
            return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)

        reasons = [field(resp, "status"), field(resp, "stop_reason")]
        for item in (field(resp, "choices", []) or []) + (field(resp, "candidates", []) or []):
            reason = field(item, "finish_reason")
            reasons.append(getattr(reason, "name", reason))
        if any(str(reason).lower() in {"length", "max_tokens", "incomplete", "failed", "cancelled",
                                       "content_filter", "safety", "recitation"} for reason in reasons):
            raise IncompleteGenerationError("Provider returned incomplete or blocked text; original retained")
        texts: List[str] = []

        # OpenAI Python client convenience property (if present)
        try:
            if hasattr(resp, "output_text"):
                texts.append(str(resp.output_text))
            elif hasattr(resp, "output") and isinstance(resp.output, list) and resp.output:
                for item in resp.output:
                    if isinstance(item, dict) and "content" in item and isinstance(item["content"], list):
                        for c in item["content"]:
                            if isinstance(c, dict):
                                # MiniMax Anthropic API content blocks may have text or type
                                if c.get("type") == "text" and "text" in c:
                                    texts.append(c["text"])
                                elif "text" in c:
                                    texts.append(c["text"])
                    elif isinstance(item, str):
                        texts.append(item)
        except Exception:
            pass

        # Chat completion-style object responses
        try:
            choices = getattr(resp, "choices", None)
            if choices:
                for choice in choices:
                    message = getattr(choice, "message", None)
                    if message is None and isinstance(choice, dict):
                        message = choice.get("message")

                    content = getattr(message, "content", None)
                    if content is None and isinstance(message, dict):
                        content = message.get("content")

                    if isinstance(content, str):
                        texts.append(content)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "text" and "text" in item:
                                    texts.append(item["text"])
                                elif "text" in item:
                                    texts.append(item["text"])
                            else:
                                texts.append(str(item))
        except Exception:
            pass

        # Gemini response
        try:
            if hasattr(resp, "text") and not texts:
                texts.append(str(resp.text))
        except Exception:
            pass

        # HTTP response JSON
        try:
            if isinstance(resp, dict) and not texts:
                if "output" in resp:
                    out = resp["output"]
                    if isinstance(out, list):
                        for o in out:
                            if isinstance(o, dict) and "content" in o and isinstance(o["content"], list):
                                for c in o["content"]:
                                    if isinstance(c, dict):
                                        if c.get("type") == "text" and "text" in c:
                                            texts.append(c["text"])
                                        elif "text" in c:
                                            texts.append(c["text"])
                            elif isinstance(o, str):
                                texts.append(o)
                if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                    first = resp["choices"][0]
                    if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                        content = first["message"]["content"]
                        if isinstance(content, str):
                            texts.append(content)
                        elif isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict):
                                    if c.get("type") == "text" and "text" in c:
                                        texts.append(c["text"])
                                    elif "text" in c:
                                        texts.append(c["text"])
                    elif "text" in first:
                        texts.append(first["text"])
                # MiniMax Anthropic API: content is a list of blocks
                # Blocks can have: type="text"/"thinking", or thinking key directly (extended thinking)
                if not texts and "content" in resp:
                    if isinstance(resp["content"], list):
                        for c in resp["content"]:
                            if isinstance(c, dict):
                                # Skip extended thinking blocks (they contain reasoning, not answer)
                                if c.get("type") == "thinking" or "thinking" in c:
                                    continue
                                if c.get("type") == "text" and "text" in c:
                                    texts.append(c["text"])
                                elif "text" in c and isinstance(c["text"], str):
                                    texts.append(c["text"])
                    elif isinstance(resp["content"], str) and resp["content"].strip():
                        texts.append(resp["content"])
                    elif isinstance(resp["content"], dict):
                        cc = resp["content"]
                        if cc.get("type") == "thinking" or "thinking" in cc:
                            pass  # skip thinking
                        elif cc.get("type") == "text" and "text" in cc:
                            texts.append(cc["text"])
                        elif "text" in cc and isinstance(cc["text"], str):
                            texts.append(cc["text"])
        except Exception:
            pass

        result = "\n".join(texts)
        # Strip <thinking>...</thinking> blocks (MiniMax extended thinking)
        result = re.sub(r"<thinking>.*?</thinking>", "", result, flags=re.DOTALL).strip()
        return result

    def _max_output(self, model: str) -> int:
        """The model's output limit from the config's `models` catalogue; 0 if unknown.

        The Claude Code CLI's catalogue says 0 on purpose: it takes no cap.
        """
        models = self.config.get("models") if isinstance(getattr(self, "config", None), dict) else None
        if not isinstance(models, dict):
            return 0
        info = models.get(model) or models.get(str(model).lower()) or {}
        try:
            return max(0, int(info.get("max_output") or 0)) if isinstance(info, dict) else 0
        except (TypeError, ValueError):
            return 0

    def _default_completion_tokens(self, model_type: str) -> int:
        env_key = {
            "writing": "AI_WRITING_COMPLETION_TOKENS",
            "review": "AI_REVIEW_COMPLETION_TOKENS",
            "planning": "AI_PLANNING_COMPLETION_TOKENS",
        }.get(model_type, "AI_DEFAULT_COMPLETION_TOKENS")
        env_val = os.getenv(env_key)
        if env_val:
            try:
                return max(256, int(env_val))
            except ValueError:
                pass
        if self.provider == "groq":
            defaults = {
                "writing": int(self.config.get("groq_writing_completion_tokens", 3072)),
                "review": int(self.config.get("groq_review_completion_tokens", 1536)),
                "planning": int(self.config.get("groq_planning_completion_tokens", 1024)),
            }
            return max(256, defaults.get(model_type, int(self.config.get("groq_default_completion_tokens", 2048))))

        defaults = {
            "writing": int(self.config.get("writing_completion_tokens", 4096)),
            "review": int(self.config.get("review_completion_tokens", 2048)),
            "planning": int(self.config.get("planning_completion_tokens", 1024)),
        }
        return max(256, defaults.get(model_type, int(self.config.get("default_completion_tokens", 2048))))

    def generate_content(
        self,
        prompt: str,
        model_type: str = "writing",
        max_retries: int = 5,
        max_completion_tokens: Optional[int] = None,
        model: Optional[str] = None,
        wait_for_limits: bool = True,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """One completion. A provider usage limit is waited out, never returned.

        wait_for_limits=False raises ProviderLimitReached instead of waiting, for
        callers that must answer promptly (a served report, a game turn).
        `system` is a system message on chat endpoints and is prepended to the
        prompt elsewhere; `temperature` reaches chat endpoints only. A config with
        "stream": true streams chat replies (slow reasoning models outlive a
        gateway's idle timeout on a non-streamed request).

        Waits until the reset time the notice names; otherwise retries every
        LIMIT_RETRY seconds, with a LIMIT_PAUSE wait every LIMIT_TRIES-th time when
        the notice is a subscription window (LIMIT_WINDOW_RE) rather than a rate limit,
        indefinitely -- callers keep their progress on disk, so waiting loses
        nothing. Metered budget stops (UsageLimitExceeded and friends) still
        raise so the caller's own pause-and-save logic runs. The wait sits
        outside @accounted so it never holds the shared ledger lock.
        """
        limited = 0
        while True:
            try:
                text = self._generate_content_once(prompt, model_type, max_retries, max_completion_tokens, model,
                                                   system, temperature)
                if len(text) < 500 and REFUSAL_RE.search(text):
                    raise RuntimeError(f"[{self.provider_label}] {' '.join(text.split())[:300]}")
                if len(text) >= 500 or not LIMIT_NOTICE_RE.search(text):
                    self._note_model_used(model, model_type)
                    return text
                notice = text
            except (DailyTokenBudgetExceeded, UsageLimitExceeded, UsageStateError):
                raise
            except Exception as exc:
                if not LIMIT_ERROR_RE.search(str(exc)):
                    raise
                notice = f"{type(exc).__name__}: {exc}"
                status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
            else:
                status = None
            if not wait_for_limits:
                raise ProviderLimitReached(f"[{self.provider_label}] {' '.join(notice.split())[:300]}",
                                           status if isinstance(status, int) else None)
            limited += 1
            window = LIMIT_WINDOW_RE.search(notice) and limited % LIMIT_TRIES == 0
            wait = limit_reset_wait(notice) or (LIMIT_PAUSE if window else LIMIT_RETRY)
            resume = datetime.fromtimestamp(time.time() + wait).strftime("%H:%M")
            print(f"[{self.provider_label}] usage limit ({' '.join(notice.split())[:120]}); "
                  f"try {limited}, waiting until {resume}")
            time.sleep(wait)

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Embedding vectors for `texts`, in input order, from an OpenAI-compatible
        /embeddings endpoint (LM Studio, OpenAI, gateways). One request; errors raise."""
        model_to_use = model or self.writing_model
        if self.client is not None and hasattr(self.client, "embeddings"):
            resp = self.client.embeddings.create(model=model_to_use, input=list(texts), timeout=self.timeout)
            items = [(getattr(d, "index", i), list(d.embedding)) for i, d in enumerate(resp.data)]
        else:
            if self.session is None:
                raise RuntimeError(f"{self.provider_label} has no HTTP endpoint for embeddings")
            r = self.session.post(self.base_url.rstrip("/") + "/embeddings",
                                  json={"model": model_to_use, "input": list(texts)}, timeout=self.timeout)
            r.raise_for_status()
            items = [(d.get("index", i), list(d["embedding"])) for i, d in enumerate(r.json()["data"])]
        return [vector for _index, vector in sorted(items, key=lambda pair: pair[0])]

    def _note_model_used(self, model: Optional[str], model_type: str) -> None:
        """Append [provider, model] to AI_MODELS_USED_PATH once, so a book can credit every model that wrote it."""
        path = os.getenv("AI_MODELS_USED_PATH")
        if not path:
            return
        entry = [self.provider_label, str(model or (self.writing_model if model_type == "writing" else self.review_model))]
        if entry in getattr(self, "_models_noted", []):
            return
        try:
            with open(path, encoding="utf-8") as f:
                used = json.load(f)
        except (OSError, ValueError):
            used = []
        if entry not in used:
            used.append(entry)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(used, f, indent=2)
            os.replace(tmp, path)
        self._models_noted = used

    @accounted
    def _generate_content_once(
        self,
        prompt: str,
        model_type: str = "writing",
        max_retries: int = 5,
        max_completion_tokens: Optional[int] = None,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        retry_delays = [3, 8, 20, 45, 90]
        attempt = 0
        last_error = None
        self.last_usage = None
        # Chat endpoints take the system prompt as its own message and the CLIs as
        # their own flag; every other route (Gemini, the responses API) gets it
        # ahead of the prompt.
        native_system = self.provider in ("openai-oauth", "openrouter", "hyper", "grok", "claude", "commandcode", "opencode")
        request_prompt = f"{system}\n\n{prompt}" if system and not native_system else prompt
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": request_prompt}]

        # `model` overrides the role default: the humanness judges have to run on
        # a model that did not write the text, or the verdict measures nothing.
        model_to_use = model or (self.writing_model if model_type == "writing" else self.review_model)
        # Always ask for the model's whole output allowance. A cap never makes a
        # reply shorter -- the prompt does that -- it only cuts it off, and a cut
        # reply is an IncompleteGenerationError that costs a full retry. A
        # caller's cap is therefore a floor, never a ceiling.
        ceiling = self._max_output(model_to_use)
        completion_tokens = ceiling or max(max_completion_tokens or 0, self._default_completion_tokens(model_type))
        if self.config.get("cap_is_ceiling") and max_completion_tokens:
            # Opt-in for consumers on metered or player-chosen endpoints: the caller's
            # cap is sent as given (never above the model's own limit).
            completion_tokens = min(max_completion_tokens, ceiling) if ceiling else max_completion_tokens
        payload = {"model": model_to_use, "input": request_prompt}

        while attempt < max_retries:
            try:
                if self.provider == "groq":
                    self._groq_preflight_limit_check(model_to_use, request_prompt)
                if self.provider == "openai":
                    self._reset_usage_state_if_needed(self._usage_state)
                    bucket = self._bucket_for_model(model_to_use)
                    bucket_state = self._usage_state["buckets"].get(bucket, {})
                    effective_limit = int(int(bucket_state.get("limit", 0)) * 0.95)
                    if int(bucket_state.get("tokens", 0)) > effective_limit:
                        self._budget_pause_requested = True
                        raise DailyTokenBudgetExceeded(
                            bucket=bucket,
                            tokens_used=int(bucket_state.get("tokens", 0)),
                            token_limit=int(bucket_state.get("limit", 0)),
                            model_name=model_to_use,
                            usage_state_path=self.usage_state_path,
                        )

                if self.provider == "claude":
                    text = claude_chat(model_to_use, request_prompt, timeout=self.cli_timeout, system=system)
                    if text.strip():
                        return text
                    print(f"[claude] returned empty content on attempt {attempt + 1}")

                elif self.provider == "commandcode":
                    text = commandcode_chat(model_to_use, request_prompt, timeout=self.cli_timeout, system=system)
                    if text.strip():
                        return text
                    print(f"[commandcode] returned empty content on attempt {attempt + 1}")

                elif self.provider == "opencode":
                    return opencode_chat(model_to_use, request_prompt, timeout=self.cli_timeout, system=system,
                                         max_output=completion_tokens)

                elif self.provider == "google" and self.client is not None:
                    # Gemini API
                    try:
                        resp = self.client.models.generate_content(
                            model=model_to_use, contents=request_prompt
                        )
                    except Exception as e:
                        last_error = e
                        print(f"[gemini] client call failed on attempt {attempt + 1}: {e}")
                        raise
                    text = self._extract_text_from_response(resp)
                    if text and text.strip():
                        return text
                    print(f"[gemini] returned empty content on attempt {attempt + 1}")

                elif self.client is not None and self.provider == "minimax":
                    # OpenAI Python client (MiniMax OpenAI-compatible API)
                    try:
                        resp = self.client.chat.completions.create(
                            model=model_to_use,
                            messages=[{"role": "user", "content": request_prompt}],
                            max_completion_tokens=completion_tokens,
                            timeout=self.timeout,
                        )
                    except Exception as e:
                        error_str = str(e).lower()
                        is_401 = (
                            getattr(e, "status_code", None) == 401
                            or "401" in error_str
                            or "unauthorized" in error_str
                            or "authorized_error" in error_str
                        )
                        if is_401:
                            retry_ok = self._handle_401_auth_error(str(e))
                            if retry_ok:
                                continue
                        last_error = e
                        print(f"[{self.provider_label}_client] client call failed on attempt {attempt + 1}: {e}")
                        raise
                    text = self._extract_text_from_response(resp)
                    if text and text.strip():
                        return text
                    print(f"[{self.provider_label}_client] returned empty content on attempt {attempt + 1}")

                elif self.client is not None and self.provider == "openai":
                    # OpenAI Python client
                    try:
                        client_kwargs = {
                            "model": model_to_use,
                            "input": request_prompt,
                            "timeout": self.timeout,
                        }
                        if self.provider == "openai":
                            client_kwargs["service_tier"] = "flex"
                        client_kwargs["max_output_tokens"] = completion_tokens
                        client_kwargs.update(self._reasoning_options(model_type, responses=True))
                        resp = self.client.responses.create(**client_kwargs)
                    except Exception as e:
                        last_error = e
                        print(f"[{self.provider_label}_client] client call failed on attempt {attempt + 1}: {e}")
                        raise
                    text = self._extract_text_from_response(resp)
                    if text and text.strip():
                        usage = self._extract_usage_from_response(resp, prompt, text)
                        if self.provider == "openai":
                            budget_info = self._record_openai_usage(model_to_use, usage)
                            if budget_info["exceeded"]:
                                print(
                                    f"⚠️ OpenAI {budget_info['bucket']} usage is now "
                                    f"{budget_info['used']:,}/{budget_info['limit']:,} tokens today. "
                                    "Progress has been cached and the next OpenAI request will pause."
                                )
                        return text
                    print(f"[{self.provider_label}_client] returned empty content on attempt {attempt + 1}")

                elif self.client is not None and self.provider == "groq":
                    try:
                        resp = self.client.chat.completions.create(
                            model=model_to_use,
                            messages=[{"role": "user", "content": request_prompt}],
                            max_completion_tokens=completion_tokens,
                            timeout=self.timeout,
                        )
                    except Exception as e:
                        last_error = e
                        print(f"[groq_client] client call failed on attempt {attempt + 1}: {e}")
                        raise
                    text = self._extract_text_from_response(resp)
                    if text and text.strip():
                        usage = self._extract_usage_from_response(resp, prompt, text)
                        rate_info = self._record_groq_usage(model_to_use, usage)
                        if rate_info["exceeded"]:
                            print(
                                f"⚠️ Groq rate usage is now minute tokens {rate_info['minute_tokens']:,}/{rate_info['tpm_limit']:,}, "
                                f"minute requests {rate_info['minute_requests']:,}/{rate_info['rpm_limit']:,}, "
                                f"day requests {rate_info['day_requests']:,}/{rate_info['rpd_limit']:,}. "
                                "Progress has been cached and the next Groq request will pause."
                            )
                        return text
                    print(f"[groq_client] returned empty content on attempt {attempt + 1}")

                elif self.client is not None and self.provider in ("openai-oauth", "openrouter", "hyper", "grok"):
                    try:
                        # hyper.charm.land is OpenAI-compatible on the older
                        # `max_tokens` spelling only; the newer name is a 400.
                        # xAI's API takes the older spelling too.
                        token_arg = self.config.get("token_param") or (
                            "max_tokens" if self.provider in ("hyper", "grok") else "max_completion_tokens")
                        options = {} if temperature is None else {"temperature": temperature}
                        if self.config.get("stream"):
                            options["stream"] = True
                        resp = self.client.chat.completions.create(
                            model=model_to_use,
                            messages=messages,
                            timeout=self.timeout,
                            **{token_arg: completion_tokens},
                            **options,
                            **self._reasoning_options(model_type),
                        )
                        if self.config.get("stream"):
                            resp = self._join_stream(resp)
                    except Exception as e:
                        last_error = e
                        print(f"[{self.provider_label}_client] client call failed on attempt {attempt + 1}: {e}")
                        raise
                    text = self._extract_text_from_response(resp)
                    if text and text.strip():
                        # Only counts the provider reported; consumers keep estimates separate.
                        usage = getattr(resp, "usage", None)
                        if usage is not None:
                            self.last_usage = {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                                               "completion_tokens": getattr(usage, "completion_tokens", None)}
                        return text
                    finish_reason = None
                    try:
                        finish_reason = resp.choices[0].finish_reason
                    except Exception:
                        pass
                    print(f"[{self.provider_label}_client] returned empty content on attempt {attempt + 1} (finish_reason={finish_reason})")

                else:
                    # HTTP endpoint (OpenAI-compatible)
                    if self.session is None:
                        raise RuntimeError("HTTP session is not initialized")

                    if self.provider == "minimax":
                        url = self.base_url.rstrip("/") + "/v1/messages"
                        payload = {
                            "model": model_to_use,
                            "messages": [{"role": "user", "content": request_prompt}],
                            "max_tokens": completion_tokens,
                            "thinking_budget": 0,  # Disable extended thinking to get direct text response
                        }
                    elif self.provider == "groq":
                        payload = {"model": model_to_use, "input": request_prompt}
                        url = self.base_url.rstrip("/") + "/responses"
                    elif self.provider in ("openai-oauth", "openrouter", "hyper", "grok"):
                        payload = {
                            "model": model_to_use,
                            "messages": messages,
                            self.config.get("token_param") or "max_tokens": completion_tokens,
                        }
                        if temperature is not None:
                            payload["temperature"] = temperature
                        url = self.base_url.rstrip("/") + "/chat/completions"
                    else:
                        url = self.base_url.rstrip("/") + "/responses"
                        payload = {"model": model_to_use, "input": request_prompt,
                                   "max_output_tokens": completion_tokens}
                    options = self._reasoning_options(model_type, responses=url.endswith('/responses'))
                    payload.update(options.get('extra_body', options))
                    try:
                        r = self.session.post(url, json=payload, timeout=self.timeout)
                    except _TRANSPORT_ERRORS as e:
                        last_error = e
                        print(f"[http] request failed on attempt {attempt + 1}: {e}")
                        raise

                    if r.status_code == 200:
                        try:
                            data = r.json()
                        except ValueError:
                            print(f"[http] response not JSON on attempt {attempt + 1}")
                            data = {}
                        text = self._extract_text_from_response(data)
                        if text and text.strip():
                            usage = self._extract_usage_from_response(data, prompt, text)
                            if self.provider == "openai":
                                budget_info = self._record_openai_usage(model_to_use, usage)
                                if budget_info["exceeded"]:
                                    print(
                                        f"⚠️ OpenAI {budget_info['bucket']} usage is now "
                                        f"{budget_info['used']:,}/{budget_info['limit']:,} tokens today. "
                                        "Progress has been cached and the next OpenAI request will pause."
                                    )
                            elif self.provider == "groq":
                                rate_info = self._record_groq_usage(model_to_use, usage)
                                if rate_info["exceeded"]:
                                    print(
                                        f"⚠️ Groq rate usage is now minute tokens {rate_info['minute_tokens']:,}/{rate_info['tpm_limit']:,}, "
                                        f"minute requests {rate_info['minute_requests']:,}/{rate_info['rpm_limit']:,}, "
                                        f"day requests {rate_info['day_requests']:,}/{rate_info['rpd_limit']:,}. "
                                        "Progress has been cached and the next Groq request will pause."
                                    )
                            return text
                        print(f"[http] returned empty content on attempt {attempt + 1}")
                    else:
                        if r.status_code in (429, 502, 503, 504):
                            if self.provider == "groq":
                                groq_rate_limit = self._parse_groq_rate_limit_error(r.text)
                                if groq_rate_limit and groq_rate_limit.get("metric") == "TPD":
                                    tokens_used = int(groq_rate_limit.get("tokens_used") or 0)
                                    token_limit = int(groq_rate_limit.get("token_limit") or self.groq_daily_token_limit)
                                    self._budget_pause_requested = True
                                    self._budget_pause_reason = (
                                        f"Groq TPD limit reached for '{model_to_use}' "
                                        f"({tokens_used:,}/{token_limit:,} tokens today)."
                                    )
                                    raise UsageLimitExceeded(
                                        provider="Groq",
                                        metric="TPD",
                                        tokens_used=tokens_used,
                                        token_limit=token_limit,
                                        model_name=model_to_use,
                                        usage_state_path=self.groq_rate_state_path,
                                    )
                            # Kept as the last error so a caller whose retries run out sees the status.
                            last_error = TransportError(f"HTTP {r.status_code}: {r.text[:300]}", r.status_code)
                            print(f"[http] transient HTTP error {r.status_code} - will retry (attempt {attempt + 1})")
                        elif r.status_code == 401:
                            # 401 means auth failure — prompt for API key and save to .env
                            if self._handle_401_auth_error():
                                continue
                            r.raise_for_status()
                        else:
                            print(f"[http] HTTP error {r.status_code}: {r.text}")
                            r.raise_for_status()

            except (DailyTokenBudgetExceeded, UsageLimitExceeded, UsageStateError):
                raise
            except IncompleteGenerationError as e:
                # Truncated or blocked output is never returned, but the next
                # attempt may finish; a larger budget helps when the cap cut it.
                last_error = e
                completion_tokens = min(completion_tokens * 2, ceiling) if ceiling else completion_tokens * 2
                print(f"[{self.provider_label}] Incomplete output on attempt {attempt + 1}; "
                      f"retrying with {completion_tokens} completion tokens")
            except Exception as e:
                error_text = str(e).lower()
                request_too_large = "request_too_large" in error_text or "request entity too large" in error_text
                if request_too_large:
                    raise ValueError("Prompt exceeds provider limits; refusing to silently remove manuscript context") from e
                if self.provider == "groq":
                    groq_rate_limit = self._parse_groq_rate_limit_error(str(e))
                    if groq_rate_limit:
                        metric = groq_rate_limit.get("metric", "")
                        if metric == "TPD":
                            tokens_used = int(groq_rate_limit.get("tokens_used") or 0)
                            token_limit = int(groq_rate_limit.get("token_limit") or self.groq_daily_token_limit)
                            self._budget_pause_requested = True
                            self._budget_pause_reason = (
                                f"Groq TPD limit reached for '{model_to_use}' "
                                f"({tokens_used:,}/{token_limit:,} tokens today)."
                            )
                            raise UsageLimitExceeded(
                                provider="Groq",
                                metric="TPD",
                                tokens_used=tokens_used,
                                token_limit=token_limit,
                                model_name=model_to_use,
                                usage_state_path=self.groq_rate_state_path,
                            )
                        retry_after = groq_rate_limit.get("retry_after")
                        if retry_after and attempt < max_retries - 1:
                            print(
                                f"[groq] Rate limit reached on attempt {attempt + 1}; "
                                f"waiting {retry_after} seconds before retrying."
                            )
                            time.sleep(int(retry_after))
                            attempt += 1
                            continue
                status = getattr(e, "status_code", None)
                if status in (401, 402, 403, 404) or REFUSAL_RE.search(str(e)):
                    # A refusal, not an outage (opencode-zen's free models now only
                    # answer OpenCode's own client; an unknown model id; no funds):
                    # retrying cannot change the answer.
                    print(f"[{self.provider_label}] {model_to_use} refused the request "
                          f"({f'HTTP {status}' if status else e}); choose another provider or model")
                    raise
                last_error = e
                print(f"[{self.provider_label}] Error on attempt {attempt + 1}: {e}")

            if attempt + 1 >= max_retries:
                break  # no attempt left to wait for
            delay = retry_delays[min(attempt, len(retry_delays)-1)]
            print(f"Waiting {delay} seconds before retry...")
            time.sleep(delay)
            attempt += 1

        print("CRITICAL ERROR: Failed to generate content after all retries")
        if last_error:
            print("Last error:", last_error)
            raise last_error
        raise EmptyGenerationError(f"{self.provider_label} {model_to_use} returned no text after {max_retries} attempt(s)")


if __name__ == "__main__":
# ponytail: self-check — print the live opencode sync; asserts that the
    # shape matches what _resolve_api_key and cli._load_catalogue expect.
    sync = load_opencode_go_sync()
    print(
        "opencode sync:",
        f"key={'present' if sync['api_key'] else 'missing'}, models={len(sync['models'])}",
    )
    assert "api_key" in sync and isinstance(sync["api_key"], str), "api_key missing/wrong type"
    assert isinstance(sync["models"], dict), "models missing/wrong type"
    for mid, info in sync["models"].items():
        assert isinstance(mid, str) and mid == mid.lower(), f"model id not lowercased: {mid!r}"
        assert "max_output" in info and isinstance(info["max_output"], int), f"{mid}: max_output bad"
        assert "context" in info and isinstance(info["context"], int), f"{mid}: context bad"
    print("ok")
