"""mathforge - multi-agent math research pipeline.

Proposes original, computationally-checkable conjectures, tries hard to kill
them with independently-written code, screens the survivors for novelty against
arXiv, Crossref and OpenAlex, proves them, has a referee
attack the proof, re-verifies the proof's own lemmas with a second script written
from the statement alone, formalizes the result in Lean 4 against Mathlib, and
emits a paper. Runs one seed or churns out papers continuously.

Credentials and model access are reused verbatim from the shared ai-suite package
(ai_suite.AIService), which reads the opencode CLI's auth.json, so no new key
handling lives here.

SECURITY: every agent-written script is executed with subprocess. That is
arbitrary code execution on this machine. Scripts run inside the per-run output
directory with a wall-clock timeout, but there is no sandbox. Run this only in
an environment where that is acceptable.

Usage:
    python mathforge.py "additive structure of squarefree numbers"
    python mathforge.py "seed topic" --conjectures 5 --workers 3
    python mathforge.py --resume math_output/additive-structure-of-squarefree-numbers
    python mathforge.py --forever --resume         # finish the newest run, then carry on
    python mathforge.py --forever --workers 4        # picks its own topics, paper after paper
    python mathforge.py --runs 10 --pause 300        # ten papers, five minutes apart
    python mathforge.py --forever --publish          # public gist per machine-checked result
    python mathforge.py --setup-lean     # one-off: Mathlib project for Lean checking

Every run lands in math_output/<slug>/ (state.json, generated scripts, paper.md)
and is appended to math_output/index.md and index.json.

Statuses a conjecture can end in:
    machine-refuted  a counterexample survived an independent re-check and Lean
                     proved the negation of the claim with no `sorry`
    refuted          a counterexample survived the independent re-check; no Lean
    inconclusive     the search crashed, timed out, or its witness was rejected
                     by the independent re-check
    known            novelty referee named it in the literature
    provisional      proved, but the referee or the independent check objected
    verified         referee accepted and the independent script re-derived it
    machine-verified Lean 4 + Mathlib accepted the proof with no `sorry`
    error            the pipeline itself crashed on this conjecture (API or parsing
                     failure); nothing is cached, so --resume retries it

With --publish, the two statuses that rest on a machine verdict rather than on a
model's opinion -- `machine-verified` and `machine-refuted` -- are posted as public
GitHub gists through the `gh` CLI. Nothing is published without that flag.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The AI client, opencode credential loading and usage accounting are the shared
# ai-suite package: the sibling checkout when present (AI_SUITE_DIR overrides the
# location), else the copy vendored into this repository.
AI_SUITE = Path(os.getenv("AI_SUITE_DIR") or HERE.parent / "ai-suite")
if AI_SUITE.is_dir():
    sys.path.insert(0, str(AI_SUITE))

try:
    from ai_suite import AIService, choose_ai, exit_on_ctrl_c, load_local_env  # noqa: E402
except ImportError as exc:  # pragma: no cover - configuration error, not logic
    raise SystemExit(
        f"cannot import the shared ai_suite package (looked in {AI_SUITE} and {HERE})\n"
        "set AI_SUITE_DIR to the ai-suite checkout."
    ) from exc

OUTPUT_ROOT = Path(os.getenv("MATHFORGE_OUTPUT") or HERE / "math_output")
CODE_TIMEOUT = 300
HEARTBEAT = 60  # seconds between "still running" lines on a long stage
LEAN_TIMEOUT = 900
MAX_CODE_REPAIRS = 3
MAX_JSON_RETRIES = 3  # re-asks after a reply with no usable JSON, before the stage errors
DEFAULT_LEAN_PROJECT = Path.home() / "mathforge-lean"
# written by setup_lean once `lake build` succeeds; a project without it is a
# half-finished setup, and the next run resumes it instead of using it
LEAN_READY = ".mathforge-ready"
ELAN_BIN = Path(os.getenv("ELAN_HOME") or Path.home() / ".elan") / "bin"
# opencode models. Pro does the proposing, proving and formalizing; flash is the
# referee, where throughput matters more than depth.
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_REVIEW_MODEL = "deepseek-v4-flash"
# The shared AIService defaults to 4096/2048 completion tokens, sized for prose;
# every stage here (proofs, Lean files, papers) is longer than a chapter. Note
# that the opencode proxy IGNORES the requested cap -- measured: a request
# capped at 800 came back with 4304 completion tokens -- so on that provider
# this is advisory only and the real ceiling is the model's own (deepseek-v4:
# 384k output on a 1M context, per the opencode model sync).
DEFAULT_MAX_TOKENS = 32000
# Which leaves reasoning effort as the knob that actually bites. An empty reply
# with finish_reason=length is not a small budget, it is the model spending its
# whole output allowance thinking and never starting the answer. Measured on one
# prompt: default 6357 reasoning tokens, high 5255, medium 5247, minimal 3948,
# low 3614. `xhigh` is accepted by the openai-oauth proxy's gpt-5.x models and
# rejected by providers that do not know it, so it stays opt-in.
DEFAULT_EFFORT = "high"
EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "provider-default")
USER_AGENT = "mathforge/1.0 (automated novelty check; contact: local user)"
ARXIV_DELAY = 3.0  # arXiv asks for one request every 3 seconds
SEARCH_ROWS = 5

RULES = """\
Hard constraints on every conjecture you produce:
- It must be ORIGINAL. Do not restate a named theorem, a textbook exercise, a
  known identity, or a famous open problem (Collatz, Goldbach, twin primes,
  Riemann, ABC, Erdos-Straus, ...). Aim for a statement a specialist would call
  "plausible, small, and apparently new", not "famous".
- It must be FALSIFIABLE BY BRUTE FORCE: over explicit finite objects (integers,
  finite groups, graphs, words, lattice points, partitions, matrices over small
  fields), so a program can search for a counterexample in minutes.
- It must quantify over an INFINITE family (every n >= 1, every prime p, every
  graph in an unbounded class). A claim confined to a finite range ("for n <= 10",
  "for 5 <= n <= 8") or a single exhibited example is a computation, not a
  theorem: the search decides it outright and there is nothing left to prove.
  The search space below is a finite SLICE of the infinite claim.
- It must be FULLY FORMAL: every symbol quantified, every constant explicit. No
  "for sufficiently large n" without a stated bound. No undefined notation.
- It must be PROVABLE by elementary means (induction, counting, pigeonhole,
  generating functions, elementary number theory, linear algebra). If the only
  plausible proof needs deep machinery, it is out of scope.
"""


# "For n <= 10, enumerate ...", "For all 4x4 matrices ...": a finite task, not an area
BOUNDED_SEED = re.compile(
    r"^\s*for\s+(?:all\s+|every\s+)?(?:n\s*(?:<=|≤|<)|\d+\s*[x×]\s*\d+|"
    r".{0,40}?on at most \d+ (?:vertices|nodes|edges|elements|letters|points))", re.I)


# Fenced block with any (or no) language tag: ```python / ```lean4 / ```json / ```
FENCE = r"```[A-Za-z0-9_+.-]*[ \t]*\r?\n(.*?)```"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "run"


def _json_block(text: str):
    """Pull JSON out of an LLM reply, salvaging the common malformed shapes.

    Beyond the clean cases (fenced, or the whole reply) this scans for top-level
    values with `raw_decode`, which recovers concatenated objects and NDJSON --
    models drop the enclosing array brackets often -- and keeps the complete
    objects out of a reply truncated mid-object.
    """
    fenced = re.search(FENCE, text, re.S)
    for chunk in ([fenced.group(1)] if fenced else []) + [text]:
        try:
            return json.loads(chunk)
        except Exception:
            pass

    decoder = json.JSONDecoder()
    values, i = [], 0
    while i < len(text):
        if text[i] not in "[{":
            i += 1
            continue
        try:
            value, i = decoder.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        values.append(value)

    if not values:
        raise ValueError(f"no JSON found in reply: {text[:400]}")
    if len(values) == 1:
        return values[0]
    if all(isinstance(v, list) for v in values):
        return [item for v in values for item in v]
    return values


def _json_object(text: str, key: str) -> dict:
    """The first JSON object in a reply that carries `key`.

    Salvage can return a list (concatenated objects, an echoed example before
    the real answer); calling `.get` on that crashed the whole conjecture.
    """
    parsed = _json_block(text)
    for value in parsed if isinstance(parsed, list) else [parsed]:
        if isinstance(value, dict) and key in value:
            return value
    raise ValueError(f"no JSON object with {key!r} in reply: {text[:400]}")


def _code_block(text: str) -> str:
    blocks = re.findall(FENCE, text, re.S)
    return (blocks[0] if blocks else text).strip()


# reentrant: a print under the lock reaches _LiveStdout.write, which takes it again
_PRINT_LOCK = threading.RLock()

# On a terminal, running steps share one status line redrawn in place below the
# log; piped to a file, heartbeats stay appended lines so the log keeps them.
LIVE = sys.stdout.isatty()
_live: dict = {}  # heartbeat token -> "c3 prove 2.1m"
_live_width = 0
_line_open = False  # someone else's output left a line unfinished (e.g. an input() prompt)


def _live_draw(text: str) -> None:
    """Overwrite the status line with `text` ("" clears it). Caller holds _PRINT_LOCK.
    Plain \\r and spaces, no ANSI: a legacy Windows console prints escapes raw."""
    global _live_width
    if text and _line_open:
        return  # drawing now would overwrite a half-written line
    text = text[: shutil.get_terminal_size().columns - 1]
    out = getattr(sys.stdout, "inner", sys.stdout)
    out.write("\r" + " " * _live_width + "\r" + text)
    out.flush()
    _live_width = len(text)


class _LiveStdout:
    """sys.stdout on a terminal. Every write -- ours, ai_suite's retry notices,
    anyone's print -- clears the status line first and redraws it after a newline,
    so nothing lands on the status line's row.
    ponytail: stdout only; a stderr traceback can still share the row."""

    def __init__(self, inner):
        self.inner = inner

    def write(self, s: str) -> int:
        global _line_open
        with _PRINT_LOCK:
            if _live_width:
                _live_draw("")
            n = self.inner.write(s)
            if s:
                _line_open = not s.endswith("\n")
            if _live:
                _live_draw("  |  ".join(_live.values()))
            return n

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _emit(text: str) -> None:
    """One permanent line, serialized across workers."""
    with _PRINT_LOCK:
        print(text, flush=True)


def log(msg: str, tag: str = "") -> None:
    """One timestamped line, serialized across workers.

    Stages are minutes apart and up to `--workers` of them run at once, so every
    line carries a clock and the conjecture it belongs to; without both, parallel
    output is unreadable and a long silence is indistinguishable from a hang.
    """
    _emit(f"{time.strftime('%H:%M:%S')}  {tag:<4} {msg}")


# Compact by default: sub-step chatter only with --verbose (or MATHFORGE_VERBOSE=1).
VERBOSE = os.getenv("MATHFORGE_VERBOSE", "") == "1"


def vlog(msg: str, tag: str = "") -> None:
    """A log line only --verbose wants: per-query, per-attempt, stage start/cached."""
    if VERBOSE:
        log(msg, tag)


def progress_bar(done: int, total: int, width: int = 20) -> str:
    filled = width * done // max(1, total)
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total}"


@contextlib.contextmanager
def live_bar(label: str, total: int):
    """Yields step() -> done count. On a terminal the bar leads the status line and
    is redrawn in place on every step; piped, it draws nothing (callers log lines)."""
    token, done = object(), [0]

    def draw() -> None:
        text = f"{label} {progress_bar(done[0], total)}".strip()
        if token not in _live:  # first draw: put the bar ahead of running steps
            rest = dict(_live)
            _live.clear()
            _live[token] = text
            _live.update(rest)
        _live[token] = text
        _live_draw("  |  ".join(_live.values()))

    def step() -> int:
        with _PRINT_LOCK:
            done[0] += 1
            if LIVE:
                draw()
            return done[0]

    if LIVE:
        with _PRINT_LOCK:
            draw()
    try:
        yield step
    finally:
        if LIVE:
            with _PRINT_LOCK:
                _live.pop(token, None)
                _live_draw("  |  ".join(_live.values()))


def rule(title: str = "") -> None:
    _emit(f"\n{('── ' + title + ' ').ljust(78, '─') if title else '─' * 78}\n")


def _dur(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds < 90 else f"{seconds / 60:.1f}m"


VERDICT_MARKERS = (
    "COUNTEREXAMPLE", "SANITY FAILED", "ALL CHECKS PASSED", "CHECK FAILED", "TIMEOUT", "error:", "Error",
)


# lines _clip never drops: everything a verdict function looks for
CLIP_KEEP = ("NO COUNTEREXAMPLE", "COUNTEREXAMPLE", "SANITY FAILED", "ALL CHECKS PASSED",
             "REFUTATION CONFIRMED", "REFUTATION REJECTED",
             "CHECK FAILED", "declaration uses 'sorry'", "depends on axioms",
             "does not depend on any axioms")


def _verdict_line(output: str, limit: int = 88) -> str:
    """The most informative line of a script's output, for a one-line log."""
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    for line in reversed(lines):
        if any(m in line for m in VERDICT_MARKERS):
            return line[:limit]
    return lines[-1][:limit] if lines else "(no output)"


def _tag(key: str) -> str:
    """`c3.falsify` -> `c3`; run-level keys get no tag."""
    head = key.split(".")[0]
    return head if re.fullmatch(r"c\d+", head) else ""


@contextlib.contextmanager
def heartbeat(label: str, tag: str = "", every: float = 0):
    """Tick while a step runs. A single model call can take minutes and prints
    nothing, which is indistinguishable from a hang; this says which step owns
    the silence and how long it has held it."""
    # compact mode ticks five times less often; it still proves the run is alive.
    # A live status line costs no scrollback, so it ticks every second.
    live = LIVE
    every = every or (1 if live else HEARTBEAT if VERBOSE else 5 * HEARTBEAT)
    stop = threading.Event()
    token = object()

    def tick():
        started = time.time()
        while not stop.wait(every):
            waited = time.time() - started
            if not live:
                log(f"{label}: still running ({_dur(waited)})", tag)
                continue
            with _PRINT_LOCK:
                if stop.is_set():  # the block ended while we waited for the lock
                    break
                _live[token] = f"{tag + ' ' if tag else ''}{label} {_dur(waited)}"
                _live_draw("  |  ".join(_live.values()))

    threading.Thread(target=tick, daemon=True).start()
    try:
        yield
    finally:
        stop.set()
        if live:
            with _PRINT_LOCK:
                if _live.pop(token, None) is not None:
                    _live_draw("  |  ".join(_live.values()))


class Run:
    """On-disk, resumable run state. One directory per research run."""

    def __init__(self, path: Path):
        # absolute: generated scripts run with cwd set to this directory, so a
        # relative path here would be resolved against itself twice
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = self.path / "state.json"
        self.data = json.loads(self.file.read_text(encoding="utf-8")) if self.file.exists() else {}
        self.lock = threading.Lock()

    def save(self):
        with self.lock:
            # atomic: this file is the resume cache, and a crash mid-write would
            # corrupt it and take every finished stage with it
            tmp = self.file.with_name(self.file.name + ".tmp")
            tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            # Windows: an antivirus or indexer holding state.json for a moment
            # makes os.replace raise WinError 5; seen in the selftest. Retry briefly
            # rather than lose a finished stage.
            for attempt in range(10):
                try:
                    os.replace(tmp, self.file)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.2 * (attempt + 1))

    def stage(self, key, produce):
        """Run `produce()` once ever; cached under `key` across restarts.

        Every long-running step goes through here, so this is also where the
        start/finish logging lives -- one place instead of a print at each call
        site.
        """
        name = key.split(".", 1)[-1]
        if key in self.data:
            vlog(f"{name}: cached", _tag(key))
            return self.data[key]
        started = time.time()
        vlog(f"{name}: start", _tag(key))
        try:
            with heartbeat(name, _tag(key)):
                value = produce()
        except Exception as exc:
            log(f"{name}: FAILED after {_dur(time.time() - started)} — {type(exc).__name__}: {exc}", _tag(key))
            raise
        vlog(f"{name}: done in {_dur(time.time() - started)}", _tag(key))
        with self.lock:
            self.data[key] = value
        self.save()
        return value


def _clip(output: str, limit: int = 8000) -> str:
    """Keep both ends of a long output, plus every verdict line in between.

    Every verdict (classify_search, check_passed, lean_verdict) is read from
    this string, so a marker cut out of the middle -- a `CHECK FAILED` among
    chatty lemma output, a `declaration uses 'sorry'` among 600 lines of Lean
    warnings -- would flip a failure into a pass.
    """
    if len(output) <= limit:
        return output
    head, tail = output[: limit // 4], output[-(limit * 3 // 4):]
    middle = output[limit // 4: -(limit * 3 // 4)]
    # capped per marker, so a script printing ten thousand witnesses stays small
    # while one rare marker still survives next to them
    kept, seen = [], dict.fromkeys(CLIP_KEEP, 0)
    for ln in middle.splitlines():
        # the negative marker contains the positive one: without its own key, 20
        # `NO COUNTEREXAMPLE` lines used up the cap and hid a real witness
        positive = ln.replace("NO COUNTEREXAMPLE", "")
        hit = [m for m in CLIP_KEEP if seen[m] < 20
               and (m in ln if m == "NO COUNTEREXAMPLE" else m in positive)]
        for m in hit:
            seen[m] += 1
        if hit:
            kept.append(ln[:300])
    return head + "\n... [output truncated; verdict lines kept] ...\n" + "\n".join(kept) + "\n" + tail


def run_code(code: str, workdir: Path, name: str, timeout: int = CODE_TIMEOUT):
    """Execute an agent-written script. Returns (exit_code, combined_output)."""
    script = workdir / name
    script.write_text(code, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            # a script that prints "≤" under a non-UTF-8 console codepage, or
            # writes raw bytes, must not crash the pipeline while decoding
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            timeout=timeout,
            cwd=str(workdir),
        )
        return proc.returncode, _clip(proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT: script exceeded {timeout}s wall clock."


def _http_get(url: str, timeout: int = 30) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        # arXiv's front end answers 406 to Python's TLS client on any uncached
        # query, whatever the headers; curl (shipped with Windows 10+) gets through
        if exc.code == 406 and shutil.which("curl"):
            return subprocess.run(
                ["curl", "-sSf", "-A", USER_AGENT, "--max-time", str(timeout), url],
                capture_output=True, check=True,
            ).stdout.decode("utf-8", "replace")
        # OpenAlex throttles with 429 / 503: one wait and retry, then give up
        if exc.code not in (429, 503):
            raise
        wait = exc.headers.get("Retry-After", "")  # seconds, or an HTTP date we ignore
        time.sleep(min(int(wait), 60) if wait.isdigit() else 10)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def _clean(text: str, limit: int = 700) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", text or "").split())[:limit]


def search_arxiv(query: str, rows: int = SEARCH_ROWS) -> list:
    url = (
        "https://export.arxiv.org/api/query?search_query=all:"
        + urllib.parse.quote_plus(query)
        + f"&start=0&max_results={rows}"
    )
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = []
    for entry in ET.fromstring(_http_get(url)).findall("a:entry", ns):
        entries.append(
            {
                "source": "arXiv",
                "title": _clean(entry.findtext("a:title", "", ns), 300),
                "abstract": _clean(entry.findtext("a:summary", "", ns)),
                "url": _clean(entry.findtext("a:id", "", ns), 200),
            }
        )
    return entries


def search_crossref(query: str, rows: int = SEARCH_ROWS) -> list:
    url = (
        f"https://api.crossref.org/works?rows={rows}&select=title,abstract,DOI"
        "&query.bibliographic=" + urllib.parse.quote_plus(query)
    )
    items = json.loads(_http_get(url)).get("message", {}).get("items", [])
    return [
        {
            "source": "Crossref",
            "title": _clean(" ".join(item.get("title") or []), 300),
            "abstract": _clean(item.get("abstract", "")),
            "url": f"https://doi.org/{item['DOI']}" if item.get("DOI") else "",
        }
        for item in items
    ]


def search_openalex(query: str, rows: int = SEARCH_ROWS) -> list:
    """OpenAlex: keyless, and far broader than Crossref for mathematics."""
    mail = os.getenv("OPENALEX_MAIL_ADDRESS", "")
    # anonymous search gets 429s whenever OpenAlex is under load; a free key avoids that
    key = os.getenv("OPENALEX_API_KEY", "")
    url = (
        f"https://api.openalex.org/works?per-page={rows}"
        "&select=id,title,abstract_inverted_index,doi"
        + (f"&mailto={urllib.parse.quote_plus(mail)}" if mail else "")
        + (f"&api_key={urllib.parse.quote_plus(key)}" if key else "")
        + "&search=" + urllib.parse.quote_plus(query)
    )
    entries = []
    for work in json.loads(_http_get(url)).get("results", []):
        # OpenAlex ships abstracts as {word: [positions]}; rebuild the text
        index = work.get("abstract_inverted_index") or {}
        words = sorted((pos, word) for word, spots in index.items() for pos in spots)
        entries.append(
            {
                "source": "OpenAlex",
                "title": _clean(work.get("title") or "", 300),
                "abstract": _clean(" ".join(w for _, w in words)),
                "url": work.get("doi") or work.get("id") or "",
            }
        )
    return entries


def search_semantic_scholar(query: str, rows: int = SEARCH_ROWS) -> list:
    """Semantic Scholar. Only used when S2_API_KEY is set: keyless access is
    rate-limited hard enough that it mostly returns 429s under a running loop."""
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search?limit={rows}"
        "&fields=title,abstract,url&query=" + urllib.parse.quote_plus(query)
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "x-api-key": os.getenv("S2_API_KEY", "")}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8", "replace"))
    return [
        {
            "source": "SemanticScholar",
            "title": _clean(p.get("title") or "", 300),
            "abstract": _clean(p.get("abstract") or ""),
            "url": p.get("url") or "",
        }
        for p in payload.get("data", [])
    ]


class _RateLimiter:
    """Enforce one backend's request spacing across every worker thread.

    arXiv asks for one request every 3 seconds globally, not per client; a
    `time.sleep` inside each thread would still let `--workers 4` fire four
    arXiv calls at once. The lock serializes the gap on shared wall-clock time.
    """

    def __init__(self, gap: float):
        self.gap = gap
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self) -> None:
        with self.lock:
            pause = self.last + self.gap - time.monotonic()
            if pause > 0:
                time.sleep(pause)
            self.last = time.monotonic()


# shared across threads and across literature() calls: the spacing requirement
# is global, not per call
_LIMITERS: dict = {}


def _backends() -> list:
    """(callable, minimum-gap) pairs. arXiv asks for 3s spacing; the rest are polite."""
    backends = [(search_arxiv, ARXIV_DELAY), (search_crossref, 0.0), (search_openalex, 0.0)]
    if os.getenv("S2_API_KEY"):
        backends.append((search_semantic_scholar, 1.0))
    return backends


def literature(queries: list, rows: int = SEARCH_ROWS, tag: str = "") -> dict:
    """Retrieve hits for the given queries across every available backend.

    Best-effort: every backend failure is recorded and the rest still run, so a
    dropped network degrades novelty screening to memory-only instead of
    killing the pipeline. Returns {"hits": [...], "errors": [...]}.
    """
    hits, errors, seen = [], [], set()
    for i, query in enumerate(queries, 1):
        vlog(f'  search {i}/{len(queries)}: "{query[:70]}"', tag)
        for backend, delay in _backends():
            if delay:
                _LIMITERS.setdefault(backend.__name__, _RateLimiter(delay)).wait()
            try:
                found = backend(query, rows)
            except Exception as exc:  # network, XML, JSON, rate limit
                errors.append(f"{backend.__name__}('{query}'): {type(exc).__name__}: {exc}")
                vlog(f"    {backend.__name__}: {type(exc).__name__}: {str(exc)[:80]}", tag)
                found = []
            for hit in found:
                key = hit["title"].lower()
                if key and key not in seen:
                    seen.add(key)
                    hit["matched_query"] = query
                    hits.append(hit)
    return {"hits": hits, "errors": errors}


def canon_negatives(output: str) -> str:
    """Spell every negative marker `NO COUNTEREXAMPLE`. Scripts printed
    `NO-COUNTEREXAMPLE` and `NO_COUNTEREXAMPLES`; the leftover `COUNTEREXAMPLE`
    read as a witness and two clean searches were published as refutations."""
    return re.sub(r"\bNO[\s_-]+COUNTER[\s_-]?EXAMPLES?\b", "NO COUNTEREXAMPLE", output, flags=re.I)


def classify_search(exit_code: int, output: str) -> str:
    """refuted | clean | inconclusive, from an adversarial search script's output.

    `NO COUNTEREXAMPLE` contains `COUNTEREXAMPLE`, so the negative marker is
    removed before looking for the positive one. Both markers present means the
    script ignored its brief, which is inconclusive, not a refutation.
    """
    output = canon_negatives(output)
    if exit_code != 0 or "SANITY FAILED" in output:
        return "inconclusive"
    clean = "NO COUNTEREXAMPLE" in output
    refuted = "COUNTEREXAMPLE" in output.replace("NO COUNTEREXAMPLE", "")
    if refuted != clean:
        return "refuted" if refuted else "clean"
    return "inconclusive"


def check_passed(check: dict) -> bool:
    """The independent check passed only if it exited clean, printed the pass
    marker, and never printed the failure one. A script that catches its own
    assertion and carries on to `ALL CHECKS PASSED` has not passed."""
    output = check.get("output", "")
    return check.get("exit_code") == 0 and "ALL CHECKS PASSED" in output and "CHECK FAILED" not in output


def refutation_confirmed(check: dict | None) -> bool:
    """The review model's re-check of a witness, read like check_passed."""
    output = (check or {}).get("output", "")
    return ((check or {}).get("exit_code") == 0 and "REFUTATION CONFIRMED" in output
            and "REFUTATION REJECTED" not in output)


def faithful(lean: dict | None) -> bool:
    """A FAITHFUL back-translation the judge does not itself call trivialized."""
    faith = (lean or {}).get("faithfulness") or {}
    return faith.get("verdict") == "FAITHFUL" and str(faith.get("trivialized")).lower() != "true"


def find_lean_project(explicit: str | None = None) -> Path | None:
    """Locate a Lake project with Mathlib available. None means "skip Lean"."""
    # an explicit path is the only candidate: a typo must not fall back silently
    for candidate in ((explicit,) if explicit else (os.getenv("MATHFORGE_LEAN_PROJECT"), DEFAULT_LEAN_PROJECT)):
        if not candidate:
            continue
        path = Path(candidate)
        if (path / "lakefile.toml").exists() or (path / "lakefile.lean").exists():
            return path
    return None


def run_lean(code: str, project: Path, workdir: Path, name: str, timeout: int = LEAN_TIMEOUT):
    """Elaborate a Lean 4 file against a Lake project. Returns (exit_code, output).

    The file lives inside the project so `lake env` puts Mathlib on the path;
    a copy is kept in the run directory as the paper's artifact.
    """
    scratch = project / "MathForge"
    scratch.mkdir(exist_ok=True)
    lean_file = scratch / f"{name}.lean"
    lean_file.write_text(code, encoding="utf-8")
    (workdir / f"{name}.lean").write_text(code, encoding="utf-8")
    lake = _lake() or "lake"
    try:
        proc = subprocess.run(
            [lake, "env", "lean", str(lean_file)],
            capture_output=True,
            text=True,
            encoding="utf-8",  # Lean's messages are full of ∀, ℕ, ⁻¹
            errors="replace",
            timeout=timeout,
            cwd=str(project),
        )
        return proc.returncode, _clip(proc.stdout + proc.stderr)
    except FileNotFoundError:
        return -2, "lake not found on PATH"
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT: lean exceeded {timeout}s wall clock."


# What a classical Mathlib proof may rest on. Anything else -- sorryAx, a
# smuggled `axiom`, Lean.ofReduceBool from native_decide -- is not a proof.
STANDARD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
# `theorem «a name»`, `theorem main.{u}`, `set_option … in theorem`, `nonrec theorem`
_DECL = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*(?:(?:set_option\s+\S+\s+\S+|open\b[^\n]*?)\s+in\s+)*"
    r"(?:(?:private|protected|noncomputable|nonrec)\s+)*(?:theorem|lemma)\s+(«[^»\n]+»|[^\s:({\[«]+)")
_IDENT = re.compile(r"(?:«[^»\n]+»|[A-Za-z_Ͱ-Ͽἀ-῿℀-⅏][\w'!?₀-ₜ.]*)")


def _uncommented(code: str) -> str:
    """The source without comments, line numbers kept. A `lemma 2 gives ...` in a
    block comment is not a declaration, and a one-line `/-- doc -/ theorem t`
    is one. Nested block comments close early here; whatever leaks out only
    ever makes the declaration count disagree, which fails safe."""
    code = re.sub(r"/-.*?-/", lambda m: "\n" * m.group(0).count("\n"), code, flags=re.S)
    return re.sub(r"--[^\n]*", "", code)


def _theorems(code: str) -> tuple[list, int]:
    """(qualified theorem names the probe can print, `theorem`/`lemma` keywords seen).

    The two disagree when a declaration has a shape the parser does not know;
    lean_verdict then refuses sorry_free rather than trusting a partial probe.
    """
    names, spaces = [], []
    text = _uncommented(code)
    for line in text.splitlines():
        if m := re.match(r"^\s*namespace\s+(\S+)", line):
            spaces.append(m.group(1))
        elif (m := re.match(r"^\s*end\s+(\S+)", line)) and spaces and spaces[-1] == m.group(1):
            spaces.pop()
        elif m := _DECL.match(line):
            name = re.sub(r"\.(?:\{[^}]*\})?$", "", m.group(1))  # `main.{u}` stops at the brace
            if not _IDENT.fullmatch(name):
                continue
            names.append(name[len("_root_."):] if name.startswith("_root_.") else ".".join(spaces + [name]))
    return names, len(re.findall(r"(?<![\w.])(?:theorem|lemma)\b", text))


def _axiom_probe(code: str) -> str:
    """`#print axioms` for every theorem, appended to the file before elaboration.

    The regex over the source cannot see an axiom behind a doc comment or a
    `set_option ... in`, nor `native_decide`; Lean's own report can. Names are
    qualified by the enclosing `namespace` blocks, since the probe runs at the
    end of the file.
    """
    # ponytail: parsed names, not Lean's own environment walk (a `run_cmd` over
    # `getEnv`); switch to that once it can be tested against a real Lean install
    # pp.fullNames: under `open Classical` Lean prints `choice`, not the standard
    # `Classical.choice`, and a sorry-free proof read as resting on a foreign axiom
    return "".join(f"\nset_option pp.fullNames true in\n#print axioms {n}" for n in _theorems(code)[0])


def _run_lean_probed(code: str, project: Path, workdir: Path, name: str):
    """run_lean with the probe appended. An error that sits only on the probe's
    own lines is not the model's to repair -- it cannot see those lines -- so it
    counts as compiled, and the missing report keeps it from sorry_free."""
    rc, out = run_lean(code + _axiom_probe(code), project, workdir, name)
    # Only a name the probe could not resolve is the probe's fault. A file that
    # stops mid-proof (a truncated completion) also errors on the probe line --
    # `unexpected token '#print'` -- and that one still goes back for repair.
    errors = re.findall(r"\.lean:(\d+):\d+: error:?[ \t]*([^\r\n]*)", out)
    if (rc > 0 and errors and "unexpected" not in out
            and all(int(n) > len(code.splitlines()) and re.match(r"unknown (?:constant|identifier)", msg)
                    for n, msg in errors)):
        rc, out = 0, out + "\n(axiom probe could not print a theorem; not counted as verified)"
    return rc, out


def lean_verdict(code: str, exit_code: int, output: str) -> dict:
    """Machine verdict on a Lean elaboration run.

    The elaborator prints `declaration uses 'sorry'` for sorry proofs, but an
    `axiom` declaration compiles silently and would smuggle an unproved
    statement past a sorry count of zero -- so axioms are counted and disqualify
    `sorry_free` just like the elaborator's own warning.
    """
    # modifiers and attributes do not change what an axiom is: `private axiom`,
    # `@[simp] axiom` compile just as silently
    axioms = len(re.findall(
        r"^\s*(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|unsafe|scoped)\s+)*axiom\b",
        code, re.M,
    ))
    compiles = exit_code == 0
    # the `#print axioms` report from _axiom_probe: at least one theorem must
    # report, and every axiom it rests on must be a standard one
    reports = re.findall(r"depends on axioms: \[([^\]]*)\]", output)
    reports += re.findall(r"does not depend on any axioms", output)
    names, declared = _theorems(code)
    unexpected = sorted({a.strip() for r in reports if r != "does not depend on any axioms"
                         for a in r.split(",") if a.strip()} - STANDARD_AXIOMS)
    return {
        "compiles": compiles,
        # every declared theorem parsed and reported: a theorem the probe could
        # not name is a theorem nobody checked
        "sorry_free": (compiles and axioms == 0 and not unexpected and declared == len(names)
                       and len(reports) >= declared > 0
                       and "declaration uses 'sorry'" not in output),
        "sorries": len(re.findall(r"\bsorry\b", code)),
        "axioms": axioms,
        "unexpected_axioms": unexpected,
    }


def _lake() -> str | None:
    """`lake` on PATH, else in elan's own bin. The VS Code Lean extension
    installs elan there without touching an already-open shell's PATH, so the
    directory is put on this process's PATH for lake's own child processes."""
    found = shutil.which("lake")
    if not found and (found := shutil.which("lake", path=str(ELAN_BIN))):
        os.environ["PATH"] = f"{ELAN_BIN}{os.pathsep}{os.environ.get('PATH', '')}"
    return found


def install_elan() -> bool:
    """Run the official elan installer unattended (no prompt, stable toolchain)."""
    if os.name == "nt":
        # bool parameters of a .ps1 only bind from -Command, not from -File
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
               "$f = Join-Path $env:TEMP 'elan-init.ps1'; "
               "Invoke-WebRequest https://elan.lean-lang.org/elan-init.ps1 -OutFile $f -UseBasicParsing; "
               "& $f -NoPrompt 1 -DefaultToolchain stable; exit $LASTEXITCODE"]
    else:
        cmd = ["sh", "-c", "curl -sSfL https://elan.lean-lang.org/elan-init.sh | sh -s -- -y --default-toolchain stable"]
    print("lake not found; installing elan (the Lean toolchain manager)...")
    try:
        subprocess.run(cmd, check=False)
    except OSError as exc:
        print(f"  elan install failed: {exc}")
    return _lake() is not None


def setup_lean(project: Path) -> int:
    """Install elan if needed and build a Mathlib-backed Lake project. Unattended
    and resumable: every step is safe to repeat, so an interrupted download is
    picked up by the next call. Downloads several GB of Mathlib cache."""
    if (project / LEAN_READY).exists():
        print(f"Lean project ready at {project}")
        return 0
    lake = _lake() or (install_elan() and _lake())
    if not lake:
        print("could not install elan automatically; install it from https://lean-lang.org/install/ "
              "(or open a .lean file in VS Code with the Lean 4 extension) and re-run --setup-lean")
        return 1
    project.parent.mkdir(parents=True, exist_ok=True)
    print(f"Setting up the Mathlib project at {project} (this downloads several GB)...")
    steps = [([lake, "exe", "cache", "get"], project), ([lake, "build"], project)]
    if not find_lean_project(str(project)):
        # the toolchain Mathlib pins, not whatever `stable` is today: a mismatch
        # makes the downloaded cache useless and `lake build` compile Mathlib
        steps.insert(0, ([lake, "+leanprover-community/mathlib4:lean-toolchain", "new", project.name, "math"],
                         project.parent))
    for cmd, cwd in steps:
        print(f"  $ {' '.join(cmd[1:])}")
        result = subprocess.run(cmd, cwd=str(cwd))
        if result.returncode != 0:
            print(f"  failed with exit {result.returncode}; re-run --setup-lean to resume")
            return result.returncode
    (project / LEAN_READY).write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    print(f"Lean project ready at {project}.")
    return 0


def set_reasoning_effort(ai: AIService, effort: str, review_effort: str | None = None) -> bool:
    return ai.set_reasoning_effort(effort, review_effort)



class Forge:
    def __init__(self, ai: AIService, run: Run, lean_project: Path | None = None, search: bool = True):
        self.ai = ai
        self.run = run
        self.lean_project = lean_project
        self.search = search
        # set by the CLI before AIService is built; recorded so a gist can say
        # which model did what. A resume keeps the first pair for the run-level
        # record; code artifacts carry their own `model`, so they stay exact.
        self.models = {"writing": os.getenv("AI_WRITING_MODEL", ""), "review": os.getenv("AI_REVIEW_MODEL", "")}
        if all(self.models.values()):
            run.data.setdefault("models", {"work": self.models["writing"], "review": self.models["review"]})

    def ask(self, prompt: str, model_type: str = "writing") -> str:
        # the client's default is five retries of the identical prompt, and a
        # call here can take fifteen minutes: a model that thinks itself out of
        # its output allowance would do so for over an hour before giving up.
        # A usage limit is waited out inside AIService.generate_content, which
        # never returns the provider's limit notice as a reply.
        return self.ai.generate_content(prompt, model_type=model_type, max_retries=2)

    def ask_json(self, prompt: str, key: str, model_type: str = "writing") -> dict:
        """`ask` + `_json_object`, re-asking up to MAX_JSON_RETRIES times when
        the reply carries no JSON.

        Live runs got replies like "Suspicious proof -- I'll brute-force the
        counting formula" and nothing else: the model announced a tool call it
        cannot make and ended its turn, which errored the whole conjecture.
        """
        reply = self.ask(prompt, model_type=model_type)
        for attempt in range(MAX_JSON_RETRIES + 1):
            try:
                return _json_object(reply, key)
            except ValueError:
                if attempt == MAX_JSON_RETRIES:
                    raise
            reply = self.ask(
                prompt + "\n\nYou have no tools and cannot run code. Do all checking in "
                "your head, then reply with the JSON only.",
                model_type=model_type,
            )

    def write_and_run(
        self, brief: str, name: str, lang: str = "python", markers=(), model_type: str = "writing"
    ) -> dict:
        """Ask an agent for code, run it, hand failures back until it runs clean.

        Repairs cover crashes and, when `markers` is given, a run that exits 0
        without reporting a verdict -- otherwise a conjecture dies for a
        formatting slip. The verdict itself is never "fixed": a clean run that
        refutes the conjecture is a result, not a bug.
        """
        if lang == "lean":
            runner = lambda src: _run_lean_probed(src, self.lean_project, self.run.path, name)  # noqa: E731
        else:
            runner = lambda src: run_code(src, self.run.path, f"{name}.py")  # noqa: E731

        # `c3_falsify` -> tag `c3`, step `falsify`
        tag, _, step = name.partition("_")
        step = step or name

        started = time.time()
        vlog(f"  {step}: writing {lang} ({model_type} model)", tag)
        code = _code_block(self.ask(brief, model_type=model_type))
        vlog(f"  {step}: {len(code.splitlines())} lines written in {_dur(time.time() - started)}", tag)
        for attempt in range(MAX_CODE_REPAIRS + 1):
            label = f"  {step}: running" + (f" (repair {attempt})" if attempt else "")
            vlog(f"{label}, up to {LEAN_TIMEOUT if lang == 'lean' else CODE_TIMEOUT}s", tag)
            started = time.time()
            rc, out = runner(code)
            vlog(f"  {step}: exit {rc} in {_dur(time.time() - started)} — {_verdict_line(out)}", tag)
            silent = bool(markers) and rc == 0 and not any(m in out for m in markers)
            if (rc == 0 and not silent) or attempt == MAX_CODE_REPAIRS:
                return {"code": code, "exit_code": rc, "output": out, "repairs": attempt,
                        "model": self.models.get(model_type, "")}
            vlog(f"  {step}: {'no verdict printed' if silent else f'failed (exit {rc})'}, asking for a repair", tag)
            complaint = (
                "ran but never printed a verdict line. It must print exactly one of: "
                + " or ".join(f"`{m}`" for m in markers)
                if silent
                else f"failed with exit code {rc}"
            )
            # the repair stays on the model that wrote the code: routing it to the
            # work model would let the independent check be rewritten by the same
            # model whose proof it is checking
            code = _code_block(
                self.ask(
                    f"Your {lang} code {complaint}. Fix it and return the complete "
                    f"corrected file in one ```{lang} fence. Change only what is needed; "
                    "do NOT weaken the search, soften the test, or replace a proof with "
                    "`sorry` to silence an error.\n\n"
                    f"CODE:\n```{lang}\n{code}\n```\n\nEXIT CODE {rc}, OUTPUT:\n{out}",
                    model_type=model_type,
                )
            )

    # ---------------- agents ----------------

    def propose_one(self, seed: str, nth: int, count: int):
        """One conjecture, one model call. None if the reply was unusable.

        On the review model on purpose: proposing is recall and taste, not
        derivation, and keeping it off the work model leaves the prover facing a
        statement it did not author.
        """
        try:
            reply = self.run.stage(
                f"conjecture{nth}",
                lambda: self.ask(
                    f"You are a research mathematician hunting for small NEW theorems in: {seed}\n\n"
                    f"{RULES}\n"
                    f"You are proposer {nth} of {count} working independently on this seed. "
                    "Pick an angle the others are unlikely to pick and propose exactly ONE "
                    "conjecture. Before writing it, silently check it against the literature "
                    "you know; discard anything you can name. Prefer a statement that "
                    "combines two structures in a way you have not seen combined.\n\n"
                    "Do not survey the area first and do not weigh many candidates: settle on "
                    "one early and spend the reply making it precise.\n\n"
                    "Return ONLY a JSON object:\n"
                    '{"title": "...", "headline": "the claim itself as one plain sentence of at '
                    'most 20 words a reader grasps at a glance, e.g. every tree on p vertices has '
                    'an even number of X", "statement": "precise natural-language statement '
                    'with all quantifiers", "notation": "definitions of every symbol used", '
                    '"search_space": "the explicit finite family a program should search for a '
                    'counterexample, with concrete bounds", "why_plausible": "the heuristic or '
                    'partial argument", "why_new": "why you believe this is not in the literature"}',
                    model_type="review",
                ),
            )
            # salvage can surface a list or nested fragments; a conjecture
            # without a statement is not one.
            parsed = _json_block(reply)
            candidates = parsed if isinstance(parsed, list) else [parsed]
            found = next((c for c in candidates if isinstance(c, dict) and c.get("statement")), None)
            if found is None:
                raise ValueError(f"no statement in reply: {reply[:200]}")
            return found
        except Exception as exc:
            vlog(f"proposer {nth}/{count} produced nothing: {type(exc).__name__}: {exc}")
            # an unusable reply must not stay cached, or every --resume would
            # re-read the same garbage instead of asking again
            with self.run.lock:
                dropped = self.run.data.pop(f"conjecture{nth}", None) is not None
            if dropped:
                self.run.save()
            return None

    def propose(self, seed: str, count: int, workers: int = 1) -> list:
        """Fan out one call per conjecture.

        Asking for the whole batch in one reply is the longest completion in the
        pipeline, which is exactly where a reasoning model runs out of budget and
        returns nothing at all -- losing the batch, and with it the run. Separate
        calls are short, run in parallel, and a failure costs one conjecture.
        """
        def fan_out(nths, step):
            def one(n):
                c = self.propose_one(seed, n, count)
                step()
                return c
            if workers > 1:
                with ThreadPoolExecutor(max_workers=min(workers, len(nths))) as pool:
                    return list(pool.map(one, nths))
            return [one(n) for n in nths]

        nths = list(range(1, count + 1))
        with live_bar("proposing", count) as step:
            proposals = dict(zip(nths, fan_out(nths, step)))
        # a garbled or truncated reply is often a one-off: ask those proposers once more
        failed = [n for n, c in proposals.items() if c is None]
        if failed:
            vlog(f"asking {len(failed)} proposer(s) that produced nothing once more")
            with live_bar("re-proposing", len(failed)) as step:
                proposals.update(zip(failed, fan_out(failed, step)))

        # independent proposers land on the same idea now and then
        conjectures, seen = [], set()
        for c in proposals.values():
            key = _slug(str(c.get("title") or c["statement"])) if c else ""
            if key and key not in seen:
                seen.add(key)
                c["id"] = f"c{len(conjectures) + 1}"
                conjectures.append(c)
        if not conjectures:
            raise ValueError(f"all {count} proposers failed to return a usable conjecture")
        return conjectures

    def falsify(self, c: dict) -> dict:
        """Adversarial agent: sees the claim, not the proposer's reasoning."""
        return self.write_and_run(
            "You are an adversarial verifier. Your ONLY goal is to destroy the following "
            "claim by finding an explicit counterexample. You are rewarded for breaking it, "
            "not for confirming it.\n\n"
            f"CLAIM: {c['statement']}\n"
            f"NOTATION: {c.get('notation', '')}\n"
            f"SEARCH SPACE: {c.get('search_space', '')}\n\n"
            "Write ONE self-contained Python 3 script (stdlib, plus numpy/sympy if truly "
            "needed) that searches exhaustively over the largest slice of that space it can "
            "clear in under 4 minutes. Requirements:\n"
            "- Implement the definitions from scratch. Do not assume the claim anywhere.\n"
            "- Include at least two independent sanity checks on your own implementation "
            "(known small values, a brute-force cross-check of any clever routine). Print "
            "them. If a sanity check fails, print SANITY FAILED and exit.\n"
            "- Only search inside the claim's hypotheses (every bound such as `p >= 7`, every "
            "side condition). A witness outside them refutes nothing.\n"
            "- Compute the claimed side literally from the statement's own formula or "
            "property, and compare against THAT, never against a related quantity (another "
            "group's count, a neighbouring formula). A counterexample is a case where your "
            "brute-force value differs from the claim's value.\n"
            "- Before reporting a witness, recompute both sides for it a second time by the "
            "plainest brute force available and report it only if they still differ.\n"
            "- On finding a counterexample print exactly `COUNTEREXAMPLE:` followed by the "
            "witness and the two sides of the failing relation, then exit.\n"
            "- If the search completes clean, print exactly `NO COUNTEREXAMPLE` (these two "
            "words, a space, no hyphen) followed by the exact ranges checked and the number "
            "of cases tested.\n"
            "- Exit code 0 in both cases. Never print both markers.\n"
            "Return only the script in one ```python fence.",
            f"{c['id']}_falsify",
            markers=("COUNTEREXAMPLE", "SANITY FAILED"),
        )

    def confirm_refutation(self, c: dict, search_output: str) -> dict:
        """Re-check the falsifier's witness from the statement alone.

        On the review model: the falsifier is the work model, and its scripts
        have compared against the wrong quantity, mis-read a witness, and used a
        witness outside the hypotheses -- each printed a COUNTEREXAMPLE that was
        published. One script's word is not a refutation.
        """
        return self.write_and_run(
            "You are an independent referee checking a claimed counterexample. You do not "
            "trust the searcher: its script may have mis-implemented a definition, compared "
            "against the wrong quantity, or used a witness outside the hypotheses.\n\n"
            f"CLAIM: {c['statement']}\nNOTATION: {c.get('notation', '')}\n\n"
            f"SEARCH OUTPUT (the reported witness is in here):\n{search_output[-2000:]}\n\n"
            "Write ONE self-contained Python 3 script that implements every definition from "
            "scratch, takes the reported witness, checks that it satisfies EVERY hypothesis "
            "of the claim, and evaluates the claim's conclusion at it by plain brute force, "
            "computing the claimed side literally from the statement. Do not search for new "
            "witnesses. Print exactly `REFUTATION CONFIRMED:` followed by the witness and "
            "both sides if it meets the hypotheses and violates the conclusion; otherwise "
            "print `REFUTATION REJECTED:` followed by the reason. Exit code 0 either way; no "
            "bare `assert`. Return only the script in one ```python fence.",
            f"{c['id']}_confirm",
            markers=("REFUTATION CONFIRMED", "REFUTATION REJECTED"),
            model_type="review",
        )

    def lean_refute(self, c: dict, witness: str) -> dict:
        """Prove the negation of the whole claim in Lean by exhibiting the witness."""
        result = self.write_and_run(
            "You are formalizing a counterexample in Lean 4 with Mathlib.\n\n"
            f"CLAIM (informal, believed FALSE): {c['statement']}\nNOTATION: {c.get('notation', '')}\n\n"
            f"CONFIRMED WITNESS:\n{witness[-1500:]}\n\n"
            "Write ONE self-contained Lean 4 file that:\n"
            "- opens with `import Mathlib` and any `open` clauses you need;\n"
            "- states `theorem refutation : ¬ (<the claim, formalized faithfully with all "
            "its quantifiers and hypotheses>)`. The part inside the `¬` must be the claim "
            "itself, not an instance and not a weakened variant: negating a stronger or "
            "different statement proves nothing about this one;\n"
            "- proves it by exhibiting the witness, then settling the finite computation "
            "with `decide`, `norm_num`, `simp` or `rfl`. Do not use `native_decide` or "
            "`sorry`; either one means the counterexample is not machine-checked.\n"
            "- Add a comment `-- FAITHFULNESS:` explaining how each informal quantifier and "
            "condition maps to the Lean statement.\n"
            "Return only the Lean file in one ```lean fence.",
            f"{c['id']}_refute",
            lang="lean",
        )
        result.update(lean_verdict(result["code"], result["exit_code"], result["output"]))
        return result

    def novelty(self, c: dict) -> dict:
        """Two steps: the model writes the queries, then judges what came back.

        Retrieval is what turns "the model does not recall this" into a claim
        with sources attached. When the search is unavailable the verdict is
        still produced, flagged as memory-only.
        """
        cid = c["id"]
        queries = []
        if self.search:
            vlog("  novelty: writing search queries", cid)
            try:
                queries = self.ask_json(
                        "Write literature search queries that would surface prior work on "
                        "this statement, if any exists.\n\n"
                        f"STATEMENT: {c['statement']}\nNOTATION: {c.get('notation', '')}\n\n"
                        "Use the vocabulary a paper on this would use, not the phrasing "
                        "above: name the objects, the invariants, and the technique. Vary "
                        "generality -- one query for the exact statement, one for the "
                        "general family it belongs to, one for the technique.\n\n"
                        'Return ONLY JSON: {"queries": ["...", "...", "..."]}',
                    "queries",
                    model_type="review",
                )["queries"]
                queries = [str(q) for q in queries if str(q).strip()][:4] if isinstance(queries, list) else []
            except Exception as exc:
                vlog(f"  novelty: query generation failed: {exc}", cid)

        found = literature(queries, tag=cid) if queries else {"hits": [], "errors": ["search disabled"]}
        vlog(f"  novelty: {len(found['hits'])} hits, {len(found['errors'])} backend errors; judging", cid)

        def judge(hits: list, queries_run: list) -> dict:
            digest = (
                "\n\n".join(f"[{h['source']}] {h['title']}\n{h['url']}\n{h['abstract']}" for h in hits)
                or "(nothing retrieved)"
            )
            return self.ask_json(
                    "You are a referee deciding whether a statement is already known. You "
                    "have your own knowledge AND the search results below.\n\n"
                    f"STATEMENT: {c['statement']}\nNOTATION: {c.get('notation', '')}\n\n"
                    f"QUERIES RUN: {json.dumps(queries_run)}\n"
                    f"SEARCH RESULTS ({len(hits)} hits):\n{digest}\n\n"
                    "Be harsh. Count it as KNOWN if it is a special case, a restatement, or "
                    "a direct corollary of a named result, a standard identity, or a routine "
                    "textbook exercise -- whether you recall it yourself or see it above. "
                    "Count it as UNCLEAR if it smells familiar but you cannot pin a source. "
                    "Only say APPARENTLY_NEW if neither your knowledge nor the retrieved work "
                    "covers it. A retrieved paper counts against novelty only if it really "
                    "implies the statement; a merely related title does not.\n\n"
                    'Return ONLY JSON: {"verdict": "KNOWN"|"UNCLEAR"|"APPARENTLY_NEW", '
                    '"closest_known_results": ["..."], "matching_hits": ["url or title of any '
                    'retrieved item that covers the statement"], "reasoning": "...", '
                    '"followup_queries": ["sharper queries to settle this, if unsure"], '
                    '"search_terms": ["terms a human should still check by hand"]}',
                "verdict",
                model_type="review",
            )

        verdict = judge(found["hits"], queries)
        rounds = 1
        # keyword-only, single-shot novelty screening is the documented weak point
        # of this kind of pipeline: one refinement round on an undecided verdict.
        followups = verdict.get("followup_queries")
        followups = [str(q) for q in followups if q and str(q) not in queries][:3] if isinstance(followups, list) else []
        if self.search and verdict.get("verdict") == "UNCLEAR" and followups:
            vlog("  novelty: UNCLEAR, second search round", cid)
            more = literature(followups, tag=cid)
            queries = queries + followups
            found = {
                "hits": found["hits"] + more["hits"],
                "errors": found["errors"] + more["errors"],
            }
            verdict = judge(found["hits"], queries)
            rounds = 2

        verdict["queries"] = queries
        verdict["rounds"] = rounds
        verdict["retrieved"] = found["hits"]
        verdict["search_errors"] = found["errors"]
        verdict["evidence_base"] = "memory-only" if not found["hits"] else "retrieval"
        vlog(f"  novelty: {verdict.get('verdict')} after {rounds} round(s)", cid)
        return verdict

    def prove(self, c: dict, evidence: str) -> str:
        return self.ask(
            "You are a mathematician writing a complete, rigorous proof for publication.\n\n"
            f"THEOREM: {c['statement']}\nNOTATION: {c.get('notation', '')}\n"
            f"HEURISTIC: {c.get('why_plausible', '')}\n"
            f"COMPUTATIONAL EVIDENCE (a hostile search found no counterexample):\n{evidence}\n\n"
            "Write the proof in Markdown with LaTeX. State every lemma separately and prove "
            "each one. Justify every step; no 'clearly', no 'it is easy to see', no appeal "
            "to the computation as if it were a proof. If you cannot close the argument, say "
            "so explicitly under a heading `## GAP` and state precisely what is missing — an "
            "honest gap is worth far more than a fake proof."
        )

    def referee(self, c: dict, proof: str) -> dict:
        return self.ask_json(
            "You are a hostile referee. Find the error. Most submitted proofs contain one.\n\n"
            f"THEOREM: {c['statement']}\nNOTATION: {c.get('notation', '')}\n\n"
            f"PROOF:\n{proof}\n\n"
            "Check every step independently; re-derive the computations yourself. Look for: "
            "unjustified interchange, hidden assumptions on parameters, induction whose base "
            "case or step does not close, division by a possibly-zero quantity, quantifier "
            "swaps, a lemma used outside its stated hypotheses.\n\n"
            'Return ONLY JSON: {"verdict": "VALID"|"GAPS"|"WRONG", "issues": [{"location": '
            '"lemma/step", "problem": "...", "severity": "fatal"|"major"|"minor"}], '
            '"summary": "..."}',
            "verdict",
            model_type="review",
        )

    def repair(self, c: dict, proof: str, report: dict) -> str:
        return self.ask(
            "A referee found problems with your proof. Fix them or admit the gap.\n\n"
            f"THEOREM: {c['statement']}\n\nPROOF:\n{proof}\n\n"
            f"REFEREE REPORT:\n{json.dumps(report, indent=2)}\n\n"
            "Return the complete revised proof in Markdown. If an issue cannot be fixed, "
            "keep the rest and mark the unresolved part under `## GAP`."
        )

    def independent_check(self, c: dict, proof: str) -> dict:
        """Second verification: tests the proof's own lemmas, not just the theorem.

        Deliberately on the review model: the prover and the falsifier run on the
        work model, so routing this one elsewhere makes the two executable
        verdicts come from different models instead of sharing blind spots.
        """
        return self.write_and_run(
            "You are an independent verifier reproducing a submitted result. You do not "
            "trust the author.\n\n"
            f"THEOREM: {c['statement']}\nNOTATION: {c.get('notation', '')}\n\n"
            f"PROOF:\n{proof}\n\n"
            "Write ONE self-contained Python 3 script that tests the theorem AND each "
            "intermediate lemma of that proof numerically/symbolically on many small cases. "
            "Implement every definition from scratch — do not copy the author's code or "
            "reasoning, and never assume a lemma while testing it. Give each lemma its own "
            "check with a printed label; do NOT use bare `assert` -- a failing assertion "
            "crashes the script, and a crash is sent back for repair, which would 'fix' "
            "the verdict. Print exactly `ALL CHECKS PASSED` at the end if everything "
            "holds, or `CHECK FAILED:` with the lemma and the witness at the first failure "
            "and stop. Exit code 0 either way. Return only "
            "the script in one ```python fence.",
            f"{c['id']}_check",
            markers=("ALL CHECKS PASSED", "CHECK FAILED"),
            model_type="review",
        )

    def lean(self, c: dict, proof: str) -> dict:
        """Formalize in Lean 4 + Mathlib. The only stage that can't be bluffed."""
        result = self.write_and_run(
            "You are formalizing a result in Lean 4 with Mathlib.\n\n"
            f"THEOREM (informal): {c['statement']}\nNOTATION: {c.get('notation', '')}\n\n"
            f"INFORMAL PROOF:\n{proof}\n\n"
            "Write ONE self-contained Lean 4 file that:\n"
            "- opens with `import Mathlib` and any `open` clauses you need;\n"
            "- states the theorem formally, as faithfully as possible. The statement is "
            "the part that matters most: a formalization that is easier than the informal "
            "claim is worthless, so do not weaken hypotheses, do not add extra ones, and "
            "do not special-case the conclusion.\n"
            "- proves it, following the informal proof. Use Mathlib lemmas; `decide`, "
            "`omega`, `norm_num`, `simp` and `interval_cases` are fine.\n"
            "- If a step defeats you, leave exactly that step as `sorry` rather than "
            "distorting the statement. Be honest: a faithful statement with a `sorry` is a "
            "useful result; a mangled statement that compiles is a lie.\n"
            "- Add a comment `-- FAITHFULNESS:` explaining how each informal quantifier and "
            "condition maps to the Lean statement.\n"
            "Return only the Lean file in one ```lean fence.",
            f"{c['id']}_lean",
            lang="lean",
        )
        result.update(lean_verdict(result["code"], result["exit_code"], result["output"]))
        return result

    def faithfulness(self, c: dict, lean_code: str, negated: bool = False) -> dict:
        """Back-translate the Lean statement and compare it to the informal one.

        Lean certifies only what the Lean statement says, and a formalization
        that quietly weakens the claim still compiles. There is no mechanical
        oracle for this, so it runs on the review model -- a different model from
        the one that wrote the Lean -- and is treated as a screen, not a proof.
        """
        return self.ask_json(
                "Judge whether a Lean 4 formalization says the same thing as an informal "
                "statement. Do not check the proof; only the statement.\n\n"
                f"INFORMAL: {c['statement']}\nNOTATION: {c.get('notation', '')}\n\n"
                f"LEAN FILE:\n```lean\n{lean_code}\n```\n\n"
                + ("The file proves the NEGATION of the claim. Judge the statement inside the "
                   "outer `¬` against the informal claim: it is FAITHFUL only if it is the "
                   "claim itself. Negating a stronger claim, or a single instance, proves "
                   "nothing about this one, so call that DIVERGENT.\n\n" if negated else "")
                + "First translate the Lean theorem statement back into plain English on its "
                "own terms, without looking at the informal wording for cues. Then compare. "
                "Watch for the standard failure: hypotheses added or strengthened, the "
                "conclusion weakened or special-cased, a quantifier narrowed to a finite "
                "range, a `Fin n` or `Nat` subtlety that changes the claim, or a definition "
                "restated in a way that makes the theorem trivial.\n\n"
                'Return ONLY JSON: {"backtranslation": "the Lean statement in English", '
                '"verdict": "FAITHFUL"|"NARROWER"|"DIVERGENT", "differences": ["..."], '
                '"trivialized": true|false, "reasoning": "..."}',
            "verdict",
            model_type="review",
        )

    def next_seed(self, history: list) -> str:
        """Pick the next research area in continuous mode, avoiding past ground."""
        # Seeds written as bounded computations ("For n <= 10, enumerate ...") got
        # `verified` tallies for results that were computations, not theorems; shown
        # as-is they teach the model that this phrasing is what succeeds.
        recent = json.dumps([
            {**h, "tally": "bounded computation, not a theorem -- do not imitate"}
            if BOUNDED_SEED.match(str(h.get("seed") or "")) else h
            for h in history[-25:]
        ], indent=1, ensure_ascii=False) if history else "(none yet)"
        seed = str(self.ask_json(
            "Choose the next topic for an automated math research run. The pipeline can "
            "only keep results that are (a) checkable by brute force over explicit finite "
            "objects and (b) provable by elementary means, so pick an area rich in small "
            "concrete objects: integer sequences, finite words, graphs on few vertices, "
            "partitions, lattice paths, finite groups or rings, matrices over small fields, "
            "combinatorial designs, polynomial identities over Z.\n\n"
            "Runs so far, with what each yielded — do not repeat a topic, and prefer areas "
            "unlike those where everything came back `known`:\n"
            f"{recent}\n\n"
            "Give one narrow, specific area, not a broad field: a sentence naming the "
            "objects and the kind of relation to look for. Name an AREA, not a computation: "
            "never 'for n <= 10, enumerate X and find the smallest n such that ...'. Seeds "
            "written that way produced conjectures confined to the enumerated range, which "
            "the search settles outright and which are therefore not theorems.\n\n"
            'Return ONLY JSON: {"seed": "...", "why": "..."}',
            "seed",
            model_type="review",
        )["seed"]).strip()
        if not seed:
            raise ValueError("empty seed in reply")
        return seed

    def paper(self, seed: str, results: list) -> str:
        return self.ask(
            "Write a short research paper in Markdown with LaTeX from the verified results "
            "below. Sections: Title, Abstract, 1. Introduction (context and what is new), "
            "2. Notation, 3. Results (each theorem with its full proof, verbatim from the "
            "verified proof), 4. Computational verification (what was searched, what ranges, "
            "what the independent check confirmed, and for each theorem the Lean 4 status: "
            "not attempted / failed to compile / compiles with `sorry` / machine-checked "
            "sorry-free, together with the back-translation faithfulness verdict on the "
            "Lean statement), 5. Limitations and open questions.\n\n"
            "Be scrupulously honest. In a `Novelty` subsection of section 5, list the exact "
            "search queries that were run against arXiv, Crossref and OpenAlex, say how many "
            "hits came back, and state that this is a bounded automated search over those "
            "indexes -- not a literature review, and blind to books, journals outside them, and "
            "anything phrased differently. Where `evidence_base` is `memory-only` the search "
            "failed and novelty rests on model recall alone: say so. Also state that a "
            "sorry-free Lean proof certifies the Lean statement, which still has to be read "
            "against the informal one, and carry every remaining `## GAP` forward into "
            "section 5 instead of hiding it.\n\n"
            f"SEED TOPIC: {seed}\n\nRESULTS:\n{json.dumps(results, indent=2)}"
        )


def pipeline(forge: Forge, c: dict) -> dict:
    """One conjecture: falsify -> novelty -> prove -> referee -> independent check."""
    cid = c["id"]
    run = forge.run
    started = time.time()
    vlog(f"{c.get('title', '(untitled)')}", cid)
    vlog(f"  claim: {str(c.get('statement', ''))[:150]}", cid)

    falsification = run.stage(f"{cid}.falsify", lambda: forge.falsify(c))
    # every reader downstream looks for the canonical spelling
    falsification = {**falsification, "output": canon_negatives(falsification["output"])}
    search = classify_search(falsification["exit_code"], falsification["output"])
    if search != "clean":
        extra = {}
        detail = _counterexample_line(falsification["output"]) or _verdict_line(falsification["output"])
        if search == "refuted":
            vlog(f"  search reports a counterexample — {detail}", cid)
            confirmation = run.stage(f"{cid}.confirm", lambda: forge.confirm_refutation(c, falsification["output"]))
            extra["confirmation"] = confirmation
            if not refutation_confirmed(confirmation):
                # a witness that does not survive a second model's re-check says
                # the search script was wrong, which leaves no evidence either way
                search, detail = "inconclusive", f"witness rejected — {_verdict_line(confirmation['output'])}"
        status = "inconclusive"
        if search == "refuted":
            status = "refuted"
            detail = _verdict_line(extra["confirmation"]["output"])
            if forge.lean_project:
                lean = run.stage(f"{cid}.lean_refute", lambda: forge.lean_refute(c, extra["confirmation"]["output"]))
                if lean["compiles"] and not lean.get("faithfulness"):
                    lean = {**lean, "faithfulness": run.stage(
                        f"{cid}.refute_faithfulness", lambda: forge.faithfulness(c, lean["code"], negated=True))}
                extra["lean"] = lean
                if lean.get("sorry_free") and faithful(lean):
                    status = "machine-refuted"
                vlog(f"  lean refutation: {'sorry-free' if lean.get('sorry_free') else 'not checked'}, "
                    f"faithfulness={(lean.get('faithfulness') or {}).get('verdict', 'n/a')}", cid)
        vlog(f"  {status.upper()} — {detail}", cid)
        vlog(f"  finished as `{status}` in {_dur(time.time() - started)}", cid)
        return {**c, "status": status, "falsification": falsification, **extra}
    vlog(f"  survived the search — {_verdict_line(falsification['output'])}", cid)

    # a backend that failed (arXiv 406 throttling, OpenAlex 429) left the verdict
    # resting on partial retrieval: drop it so --resume searches again
    if forge.search and (run.data.get(f"{cid}.novelty") or {}).get("search_errors"):
        run.data.pop(f"{cid}.novelty", None)
    novelty = run.stage(f"{cid}.novelty", lambda: forge.novelty(c))
    if novelty.get("verdict") == "KNOWN":
        vlog(f"  KNOWN — closest: {'; '.join(map(str, novelty.get('closest_known_results') or []))[:150]}", cid)
        vlog(f"  finished as `known` in {_dur(time.time() - started)}", cid)
        return {**c, "status": "known", "novelty": novelty, "falsification": falsification}

    proof = run.stage(f"{cid}.proof", lambda: forge.prove(c, falsification["output"]))
    if "## GAP" in proof:
        vlog("  the prover declared a GAP in its own proof", cid)
    report = run.stage(f"{cid}.referee", lambda: forge.referee(c, proof))
    vlog(f"  referee: {report.get('verdict')} — {str(report.get('summary', ''))[:120]}", cid)
    if report.get("verdict") != "VALID":
        vlog("  sending it back for repair", cid)
        proof = run.stage(f"{cid}.proof2", lambda: forge.repair(c, proof, report))
        report = run.stage(f"{cid}.referee2", lambda: forge.referee(c, proof))
        vlog(f"  referee (round 2): {report.get('verdict')}", cid)

    check = run.stage(f"{cid}.check", lambda: forge.independent_check(c, proof))
    passed = check_passed(check)

    if not forge.lean_project:
        vlog("  lean: skipped (no Mathlib project)", cid)
    lean = run.stage(f"{cid}.lean", lambda: forge.lean(c, proof)) if forge.lean_project else None
    # its own stage: a garbled faithfulness reply used to throw away a
    # fifteen-minute Lean run with it. Older runs cached it inside the Lean stage.
    if lean and lean["compiles"] and not lean.get("faithfulness"):
        lean = {**lean, "faithfulness": run.stage(f"{cid}.faithfulness", lambda: forge.faithfulness(c, lean["code"]))}

    # a sorry-free proof of a statement that drifted from the informal claim
    # certifies the wrong theorem, so faithfulness gates the top status -- and a
    # statement the judge itself calls trivialized is not faithful to anything
    if lean and lean.get("sorry_free") and faithful(lean):
        status = "machine-verified"
    elif report.get("verdict") == "VALID" and passed:
        status = "verified"
    else:
        status = "provisional"
    if lean:
        if lean["sorry_free"]:
            state = "sorry-free"
        elif lean["compiles"] and "sorryAx" in (lean.get("unexpected_axioms") or []):
            state = "compiles+sorry"   # Lean reports a sorry as the sorryAx axiom
        elif lean["compiles"] and (lean.get("axioms") or lean.get("unexpected_axioms")):
            state = "compiles+axiom"
        elif lean["compiles"]:
            state = "compiles+sorry"
        else:
            state = "failed to compile"
        vlog(f"  lean: {state}, faithfulness={(lean.get('faithfulness') or {}).get('verdict', 'n/a')}", cid)
    vlog(f"  independent check: {'passed' if passed else 'did not pass'}", cid)
    vlog(f"  finished as `{status}` in {_dur(time.time() - started)}", cid)
    return {
        **c,
        "status": status,
        "novelty": novelty,
        "falsification": falsification,
        "proof": proof,
        "referee": report,
        "independent_check": check,
        "lean": lean,
    }


def run_one(forge: Forge, c: dict) -> dict:
    """pipeline(), but a crash costs one conjecture instead of the run.

    A single unparseable model reply inside pipeline() would propagate through
    pool.map and mark the whole run failed; weeks-long sessions should not die
    that way. The failed conjecture is recorded honestly as `error`, nothing is
    cached for it (so --resume retries), and the rest continue.
    """
    try:
        return pipeline(forge, c)
    except Exception as exc:
        log(f"  pipeline error — {type(exc).__name__}: {exc}", c["id"])
        vlog("  finished as `error` (not cached; --resume retries it)", c["id"])
        return {**c, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _for_paper(r: dict) -> dict:
    """Drop generated source from a result; keep verdicts and truncated output."""
    slim = {k: v for k, v in r.items() if k not in ("falsification", "independent_check", "lean")}
    for key in ("falsification", "independent_check", "lean"):
        artifact = r.get(key)
        if isinstance(artifact, dict):
            slim[key] = {k: v for k, v in artifact.items() if k != "code"}
            slim[key]["output"] = str(artifact.get("output", ""))[-1500:]
    novelty = r.get("novelty")
    if isinstance(novelty, dict):
        slim["novelty"] = dict(novelty, retrieved=[
            f"{h.get('source')}: {h.get('title')} {h.get('url')}" for h in novelty.get("retrieved", [])
        ])
    return slim


def negative_results(seed: str, results: list) -> str:
    """Record what died and why.

    A failed counterexample search is the evidence that supports a conjecture and
    it is almost never written down; a found counterexample is a small true fact
    about the objects. Both are cheap to keep and are thrown away by default.
    """
    lines = [f"# Negative results: {seed}", "", "Conjectures this run did not keep.", ""]
    for r in results:
        if r["status"] in ("machine-verified", "verified", "provisional"):
            continue
        lines += [f"## {r.get('title', r['id'])} — {r['status']}", "", f"**Statement.** {r.get('statement', '')}", ""]
        if r.get("notation"):
            lines += [f"**Notation.** {r['notation']}", ""]
        output = (r.get("falsification") or {}).get("output", "")
        if r["status"] in REFUTED:
            lines += [f"**Counterexample.** `{_witness(r)}`", ""]
        elif r["status"] == "known":
            novelty = r.get("novelty") or {}
            lines += [
                f"**Survived search, already known.** {novelty.get('reasoning', '')}",
                "",
                "Closest known results: " + "; ".join(map(str, novelty.get("closest_known_results") or [])),
                "",
                "Ranges searched without a counterexample: `"
                + " ".join(ln for ln in output.splitlines() if "NO COUNTEREXAMPLE" in ln).strip()
                + "`",
                "",
            ]
        else:
            if r["status"] == "error":
                lines += [f"**Pipeline error.** {r.get('error', '')}", ""]
            else:
                lines += [f"**Search inconclusive.** Last output:\n\n```\n{output[-800:]}\n```", ""]
    return "\n".join(lines) + "\n"


REFUTED = ("machine-refuted", "refuted")
# a refutation publishes only once Lean has checked it: a Python witness alone
# published four false counterexamples before the re-check and Lean existed
PUBLISH_STATUSES = ("machine-verified", "machine-refuted")
DISCLAIMER = """\
## How this was produced

Fully automated: every statement, script, proof and formalization here was
written by language models in a pipeline (mathforge), with no human in the loop.
Read it as a machine-checked artifact, not as a reviewed paper.

Only two outcomes are published, both resting on Lean 4 + Mathlib rather than on
a model's opinion of its own work:

- **machine-refuted** — an adversarial script found a witness, a second model's
  script re-checked it against the statement from scratch, and Lean elaborated a
  sorry-free proof of the negation of the claim.
- **machine-verified** — Lean elaborated the proof with no `sorry`.

In both cases a second model back-translated the Lean statement and judged it
faithful to the informal one. Lean certifies the Lean statement; the
back-translation is a screening filter, not an oracle, so read the attached
`.lean` file against the statement above.

Novelty screening is a bounded automated search over arXiv, Crossref and
OpenAlex with model-written queries. It is blind to books, to journals outside
those indexes, and to anything phrased differently: a result here may well be a
rediscovery. Corrections welcome in the comments.
"""


def publishable(r: dict) -> bool:
    return r.get("status") in PUBLISH_STATUSES


def _counterexample_line(output: str) -> str:
    """The witness line from a falsification run, without the negative marker."""
    for line in canon_negatives(output).splitlines():
        if "COUNTEREXAMPLE:" in line and "NO COUNTEREXAMPLE" not in line:
            return line.strip()
    return ""


def _witness(r: dict) -> str:
    """The re-checked witness when there is one, else the searcher's own line."""
    confirmed = (r.get("confirmation") or {}).get("output", "")
    line = next((ln.strip() for ln in confirmed.splitlines() if "REFUTATION CONFIRMED" in ln), "")
    line = line or _counterexample_line((r.get("falsification") or {}).get("output", ""))
    # the marker is for the parser; a reader wants the witness
    return re.sub(r"^.*?(?:REFUTATION CONFIRMED|COUNTEREXAMPLE):?\s*", "", line)


def headline(r: dict) -> str:
    """What the gist title leads with: `Refuted: <claim>` / `Proved: <claim>`."""
    claim = str(r.get("headline") or r.get("title") or r["id"]).strip().rstrip(".")
    return f"{'Refuted' if r.get('status') in REFUTED else 'Proved'}: {claim}"


def _lean_lines(lean: dict) -> list:
    faith = lean.get("faithfulness") or {}
    lines = [
        f"- Lean 4 + Mathlib: compiled, sorry-free (`{lean.get('sorries', 0)}` occurrences of "
        "the token in the source, none reported by the elaborator).",
        f"- Back-translation of the Lean statement: **{faith.get('verdict', 'n/a')}**. "
        f"{faith.get('backtranslation', '')}",
    ]
    if faith.get("differences"):
        lines.append("- Noted differences: " + "; ".join(map(str, faith["differences"])))
    return lines


def _model_lines(r: dict, models: dict | None) -> list:
    """Who did what. A code artifact names its own model; the rest falls back to
    the run's work/review pair, and anything unrecorded says so."""
    models = models or {}

    def who(artifact_key: str | None, role: str) -> str:
        artifact = r.get(artifact_key) if artifact_key else None
        return ((artifact or {}).get("model") if isinstance(artifact, dict) else "") or models.get(role) or "not recorded"

    if r.get("status") in REFUTED:
        rows = [("Proposed the claim", None, "review"), ("Counterexample search", "falsification", "work"),
                ("Independent re-check of the witness", "confirmation", "review")]
    else:
        rows = [("Proposed the claim", None, "review"), ("Counterexample search", "falsification", "work"),
                ("Wrote the proof", None, "work"), ("Referee", None, "review"),
                ("Independent check script", "independent_check", "review")]
    if r.get("lean"):
        rows += [("Lean formalization", "lean", "work"), ("Faithfulness judge", None, "review")]
    return [f"- {label}: {who(key, role)}" for label, key, role in rows]


def _publication(seed: str, r: dict, models: dict | None = None) -> str:
    """The gist body: verdict first, then the claim, the evidence, the caveats."""
    cid = r["id"]
    refuted = r["status"] in REFUTED
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    witness = _witness(r) if refuted else ""
    verdict = (f"**Verdict: FALSE.** Counterexample: `{witness}`" if refuted
               else "**Verdict: TRUE.** Lean 4 + Mathlib accepted a sorry-free proof.")
    lines = [f"# {headline(r)}", "", verdict, ""]
    if r.get("title") and r.get("headline"):
        lines += [f"*{r['title']}*", ""]
    lines += [
        f"*Automated run on the seed topic \"{seed}\", {stamp}. Status: `{r['status']}`.*",
        "",
        "## Statement",
        "",
        r.get("statement", ""),
        "",
    ]
    if r.get("notation"):
        lines += ["## Notation", "", r["notation"], ""]
    lean = r.get("lean") or {}

    if refuted:
        lines += [
            "## Why it is false",
            "",
            f"`{witness}`",
            "",
            "## Machine verification",
            "",
            "- Adversarial search (`*_falsify.py`) reported the witness.",
            "- Independent re-check (`*_confirm.py`, a different model, definitions "
            "re-implemented from the statement): confirmed.",
            *(_lean_lines(lean) if lean else ["- Lean: not run."]),
            "",
            "Re-check output (tail):",
            "",
            "```",
            (r.get("confirmation") or {}).get("output", "")[-1500:].strip(),
            "```",
            "",
        ]
    else:
        check = r.get("independent_check") or {}
        novelty = r.get("novelty") or {}
        lines += [
            "## Proof",
            "",
            r.get("proof", ""),
            "",
            "## Machine verification",
            "",
            *_lean_lines(lean),
            f"- Independent script (written from the proof, definitions re-implemented from "
            f"scratch): {'passed' if check_passed(check) else 'did not pass; see attached output'}.",
            f"- Adversarial search found no counterexample: "
            f"`{next((ln.strip() for ln in (r.get('falsification') or {}).get('output', '').splitlines() if 'NO COUNTEREXAMPLE' in ln), '')}`",
            f"- Novelty verdict: **{novelty.get('verdict', 'n/a')}** "
            f"({novelty.get('evidence_base', 'n/a')}, {len(novelty.get('retrieved') or [])} hits over "
            f"{len(novelty.get('queries') or [])} queries). {novelty.get('reasoning', '')}",
            "",
        ]

    lines += ["## Models", "", *_model_lines(r, models), ""]
    return "\n".join(lines + [DISCLAIMER])


def _gist_url(stdout: str) -> str:
    """`gh gist create` prints progress on stderr and the URL last on stdout."""
    return next((ln.strip() for ln in reversed(stdout.splitlines()) if ln.strip().startswith("https://")), "")


def publish_gist(run: Run, seed: str, r: dict) -> dict:
    """Post one result as a public gist via `gh`. Never raises: publishing is a
    side effect of research, and a failed post must not lose the result."""
    if not shutil.which("gh"):
        return {"error": "gh not on PATH; install the GitHub CLI to publish"}
    cid = r["id"]
    # a gist lists its files alphabetically: README.md sorts above c1_*.py, so
    # the write-up is what a reader sees first
    body = run.path / f"{cid}_gist" / "README.md"
    body.parent.mkdir(exist_ok=True)
    body.write_text(_publication(seed, r, run.data.get("models")), encoding="utf-8")
    # the generated artifacts are the point: a reader can re-run them
    extras = ([f"{cid}_refute.lean", f"{cid}_falsify.py", f"{cid}_confirm.py"] if r["status"] in REFUTED
              else [f"{cid}_lean.lean", f"{cid}_falsify.py", f"{cid}_check.py"])
    files = [body] + [run.path / name for name in extras if (run.path / name).exists()]
    desc = f"{headline(r)} (mathforge, {r['status']})"
    try:
        proc = subprocess.run(
            ["gh", "gist", "create", "--public", "--desc", desc, *(str(f) for f in files)],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(run.path),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    url = _gist_url(proc.stdout)
    if proc.returncode != 0 or not url:
        return {"error": (proc.stdout + proc.stderr)[-500:].strip() or f"exit {proc.returncode}"}
    return {"url": url, "status": r["status"], "title": r.get("title", cid), "files": [f.name for f in files]}


def publish(run: Run, seed: str, results: list) -> list:
    """Publish every qualifying result once. Cached in state.json, so a --resume
    of a published run re-reads the URL instead of posting a duplicate gist."""
    published = []
    for r in results:
        key = f"{r['id']}.gist"
        if not publishable(r):
            stale = (run.data.get(key) or {}).get("url")
            if stale:
                log(f"PUBLIC gist {stale} no longer qualifies (now `{r['status']}`); retract it", r["id"])
            continue
        # a failed post is not a result: drop it so --resume retries instead of
        # caching the error forever
        if not (run.data.get(key) or {}).get("url"):
            run.data.pop(key, None)
        info = run.stage(key, lambda r=r: publish_gist(run, seed, r))
        if info.get("url"):
            log(f"published `{r['status']}`: {info['url']}", r["id"])
            published.append(info)
        else:
            log(f"publish failed: {info.get('error')}", r["id"])
    return published


def publish_verdict(enabled: bool, results: list, published: list) -> str:
    """One end-of-seed line saying whether anything went public, and why not."""
    qualifying = sum(publishable(r) for r in results)
    rule_ = f"only {' / '.join(PUBLISH_STATUSES)} qualify"
    if not enabled:
        return f"publish: OFF (no --publish); {qualifying} result(s) would have qualified"
    if not qualifying:
        return f"publish: NOTHING PUBLISHED, no result qualified ({rule_})"
    if len(published) < qualifying:
        return f"publish: {len(published)}/{qualifying} posted, rest failed; --resume retries them"
    return f"publish: {len(published)} gist(s) posted"


def latest_run_dir() -> Path | None:
    """The most recently touched run, for a bare `--resume`. Underscore-prefixed
    scratch directories (`_scout`, `_selftest`) are not runs."""
    runs = [p.parent for p in OUTPUT_ROOT.glob("*/state.json") if not p.parent.name.startswith("_")]
    return max(runs, key=lambda p: (p / "state.json").stat().st_mtime, default=None)


def unique_run_dir(seed: str) -> Path:
    """A fresh directory per run, so a repeated auto-seed does not resume the old one."""
    slug = _slug(seed)
    candidate = OUTPUT_ROOT / slug
    n = 2
    while (candidate / "state.json").exists():
        candidate = OUTPUT_ROOT / f"{slug}-{n}"
        n += 1
    return candidate


def append_index(summary: dict) -> None:
    """Append one run to the library index, in JSON and as a readable list."""
    index = OUTPUT_ROOT / "index.json"
    entries = []
    if index.exists():
        try:
            entries = json.loads(index.read_text(encoding="utf-8"))
            if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
                raise ValueError("not a list of runs")
        except ValueError:
            # never silently reset the library: keep the damaged file for repair
            damaged = index.with_name(f"index.damaged-{time.time_ns()}.json")
            os.replace(index, damaged)
            log(f"index.json was unreadable; kept as {damaged.name}, starting a new one")
            entries = []
    # a resumed run replaces its own entry instead of being listed twice
    entries = [e for e in entries if e.get("path") != summary.get("path")] + [summary]
    tmp = index.with_name(f"index.json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    os.replace(tmp, index)

    lines = ["# mathforge library", "", f"{len(entries)} runs.", ""]
    for e in entries:
        target = e.get("paper") or e.get("path")
        kept = e.get("tally", {})
        lines.append(f"- `{e.get('finished', '')}` [{e.get('seed', '')}]({target}) — {kept}")
        for p in e.get("published") or []:
            lines.append(f"    - published {p.get('status')}: [{p.get('title')}]({p.get('url')})")
    (OUTPUT_ROOT / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def research_run(forge_for, seed: str, args, run: Run | None = None) -> dict:
    """One seed end to end. Returns the index summary."""
    run = run or Run(unique_run_dir(seed))
    run.data.setdefault("seed", seed)
    run.save()
    forge = forge_for(run)

    started = time.time()
    rule(seed[:74])
    vlog(f"run directory: {run.path}")
    conjectures = run.stage("conjectures", lambda: forge.propose(seed, args.conjectures, args.workers))
    for c in conjectures:
        vlog(f"proposed: {c.get('title', '(untitled)')}", c["id"])
    log(f"{len(conjectures)} conjectures")

    # On a terminal the bar leads the status line, redrawn in place; piped, it
    # prefixes each result line so the log still shows progress.
    with live_bar("", len(conjectures)) as step:
        def one(c):
            r = run_one(forge, c)
            done = step()
            bar = "" if LIVE else progress_bar(done, len(conjectures)) + " "
            log(f"{bar}{r['status']:<16} {c.get('title', '')[:60]}", c["id"])
            return r

        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(one, conjectures))
        else:
            results = [one(c) for c in conjectures]

    run.data["results"] = results
    run.save()

    tally = {}
    for r in results:
        tally[r["status"]] = tally.get(r["status"], 0) + 1

    keepers = [r for r in results if r["status"] in ("machine-verified", "verified", "provisional")]
    if len(keepers) < len(results):
        (run.path / "negative_results.md").write_text(negative_results(seed, results), encoding="utf-8")
        vlog(f"negative results: {run.path / 'negative_results.md'}")

    paper_path = None
    if keepers:
        # a resume can turn an `error` into a keeper; the cached paper was
        # written without it and must be redone
        kept = [f"{r['id']}:{r['status']}" for r in keepers]
        # a paper cached before keepers were recorded cannot be vouched for either
        if run.data.get("paper_keepers") != kept:
            run.data.pop("paper", None)
        run.data["paper_keepers"] = kept
        paper = run.stage("paper", lambda: forge.paper(seed, [_for_paper(r) for r in keepers]))
        (run.path / "paper.md").write_text(paper, encoding="utf-8")
        paper_path = run.path / "paper.md"

    published = publish(run, seed, results) if getattr(args, "publish", False) else []

    elapsed = time.time() - started
    order = ("machine-verified", "verified", "provisional", "known", "machine-refuted", "refuted",
             "inconclusive", "error")
    log(f"done in {_dur(elapsed)}: " + ", ".join(f"{tally[k]} {k}" for k in order if tally.get(k)))
    if paper_path:
        log(f"paper: {paper_path}")
    (log if getattr(args, "publish", False) else vlog)(publish_verdict(getattr(args, "publish", False), results, published))

    def _relative(p: Path | None) -> str | None:
        if p is None:
            return None
        try:
            return p.relative_to(OUTPUT_ROOT).as_posix()
        except ValueError:
            return str(p)

    summary = {
        "seed": seed,
        "finished": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "seconds": round(elapsed),
        "tally": tally,
        "path": _relative(run.path),
        "paper": _relative(paper_path),
        "published": published,
    }
    append_index(summary)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seed", nargs="?", help="research topic to explore")
    ap.add_argument("--conjectures", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1, help="conjectures pursued in parallel")
    ap.add_argument(
        "--provider",
        help="ai-suite provider (opencode-go, claude, hyper, ...). On a terminal the "
             "writer's provider/model menu asks when this is omitted; otherwise the last pick is reused",
    )
    ap.add_argument("--config", help="ai-suite AI config json; skips the provider menu")
    ap.add_argument("--model", help=f"model for proposing, proving, coding (menu default {DEFAULT_MODEL})")
    ap.add_argument("--review-model", help=f"model for referee stages (menu default {DEFAULT_REVIEW_MODEL})")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="completion token budget per call, where the provider honours it "
             "(the opencode proxy does not; default %(default)s)",
    )
    ap.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=EFFORTS,
        help="reasoning effort per call on the work model. An empty reply with "
             "finish_reason=length is the model thinking past its output allowance; lower "
             "this when that happens. `xhigh` only exists on some OpenAI-compatible "
             "providers and is rejected elsewhere (default %(default)s)",
    )
    ap.add_argument(
        "--review-effort",
        default=None,
        choices=EFFORTS,
        help="reasoning effort on the review model (proposing, refereeing, novelty, "
             "independent check). Defaults to --effort",
    )
    ap.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="existing run directory; bare --resume takes the most recent one. With --forever "
             "the run is finished first and then new topics continue as usual",
    )
    ap.add_argument("--runs", type=int, default=1, help="how many papers to attempt")
    ap.add_argument("--forever", action="store_true", help="keep researching until interrupted")
    ap.add_argument("--verbose", action="store_true",
                    help="log every search query, code attempt and stage start (compact by default)")
    ap.add_argument("--pause", type=int, default=0, help="seconds between runs in continuous mode")
    ap.add_argument("--lean-project", help="Lake project with Mathlib (else auto-detected)")
    ap.add_argument("--no-lean", action="store_true", help="run without Lean (required by default); nothing becomes machine-checked or publishable")
    ap.add_argument("--no-search", action="store_true", help="skip the arXiv/Crossref/OpenAlex novelty search")
    ap.add_argument(
        "--publish",
        action="store_true",
        help="post every machine-verified result and every counterexample as a PUBLIC GitHub gist (needs `gh` logged in)",
    )
    ap.add_argument(
        "--setup-lean",
        action="store_true",
        help=f"create a Mathlib Lake project at {DEFAULT_LEAN_PROJECT} and exit (multi-GB download)",
    )
    args = ap.parse_args(argv)
    global VERBOSE
    VERBOSE = VERBOSE or args.verbose

    # stages are minutes apart; keep progress visible when piped to a log
    sys.stdout.reconfigure(line_buffering=True)
    if LIVE:
        sys.stdout = _LiveStdout(sys.stdout)
    # stop at once, not after the model call in flight; the cut stage is simply not cached
    exit_on_ctrl_c(message="stopped; every finished stage is cached on disk, --resume picks it up")

    if args.setup_lean:
        return setup_lean(Path(args.lean_project) if args.lean_project else DEFAULT_LEAN_PROJECT)

    if not (args.seed or args.resume or args.forever or args.runs > 1):
        ap.error("give a seed topic, --resume a run directory, or --forever")

    lean_project = None if args.no_lean else find_lean_project(args.lean_project)
    if args.lean_project and lean_project is None:
        ap.error(f"no lakefile found in {args.lean_project}")
    # Lean is the only gate a model cannot talk its way past; running without it
    # has to be asked for, so a missing or half-built default project is set up
    # (or resumed) here rather than skipped
    if not args.no_lean and (lean_project is None or (
            lean_project == DEFAULT_LEAN_PROJECT and not (lean_project / LEAN_READY).exists())):
        if setup_lean(DEFAULT_LEAN_PROJECT) != 0:
            ap.error("Lean setup failed (see above); re-run to resume it, or pass --no-lean "
                     "to run with no machine checks and nothing publishable")
        lean_project = DEFAULT_LEAN_PROJECT

    load_local_env()
    if args.config:
        args.model = args.model or DEFAULT_MODEL
        args.review_model = args.review_model or DEFAULT_REVIEW_MODEL
    else:
        # The shared provider/model menu, so every AI script offers the
        # same, live-refreshed choices. Picks are remembered per script.
        interactive = sys.stdin.isatty() and not (args.model and args.review_model)
        _, args.config, picked = choose_ai(
            args.provider,
            "review" if interactive else "auto",
            state_file=OUTPUT_ROOT / "provider_state.json",
            roles=("work", "review"),
            defaults=(DEFAULT_MODEL, DEFAULT_REVIEW_MODEL),
            default_provider="opencode-go",
        )
        args.model = args.model or picked[0]
        args.review_model = args.review_model or picked[-1]
    # AIService reads these; setting them here keeps the shared ai-suite
    # config file untouched.
    os.environ["AI_WRITING_MODEL"] = args.model
    os.environ["AI_REVIEW_MODEL"] = args.review_model
    os.environ["AI_WRITING_COMPLETION_TOKENS"] = str(args.max_tokens)
    os.environ["AI_REVIEW_COMPLETION_TOKENS"] = str(args.max_tokens)
    ai = AIService(config_path=args.config)
    review_effort = args.review_effort or args.effort
    effort_supported = set_reasoning_effort(ai, args.effort, review_effort)

    log(f"models      {args.model} (work) / {args.review_model} (review)")
    log(f"effort      {args.effort} (work) / {review_effort} (review)"
        f"{'' if effort_supported else ' -- unsupported by this provider'}, "
        f"{args.max_tokens} tokens/call requested")
    log(f"plan        {'forever' if args.forever else f'{args.runs} run(s)'}, "
        f"{args.conjectures} conjectures each, {args.workers} worker(s)")
    vlog(f"lean        {lean_project or ('disabled (--no-lean)' if args.no_lean else 'no Mathlib project found; run --setup-lean')}")
    vlog(f"novelty     {'arXiv + Crossref + OpenAlex' + (' + Semantic Scholar' if os.getenv('S2_API_KEY') else '') if not args.no_search else 'disabled (--no-search)'}")
    vlog(f"publish     {'PUBLIC gists for machine-verified results and counterexamples' if args.publish else 'off'}")
    vlog(f"library     {OUTPUT_ROOT}")

    def forge_for(run: Run) -> Forge:
        return Forge(ai, run, lean_project, search=not args.no_search)

    if args.resume:
        path = latest_run_dir() if args.resume == "latest" else Path(args.resume)
        if path is None:
            ap.error(f"no previous run to resume under {OUTPUT_ROOT}")
        run = Run(path)
        seed = args.seed or run.data.get("seed")
        if not seed:
            ap.error("resumed run has no seed recorded; pass one explicitly")
        log(f"resuming    {path}")
        research_run(forge_for, seed, args, run=run)
        if not (args.forever or args.runs > 1):
            return 0
        # the resumed seed is spent; the loop below picks fresh topics
        args.seed = None

    history = []
    index = OUTPUT_ROOT / "index.json"
    if index.exists():
        try:
            history = [
                {"seed": e.get("seed"), "tally": e.get("tally")}
                for e in json.loads(index.read_text(encoding="utf-8"))
            ]
        except Exception:
            history = []

    vlog(f"history     {len(history)} previous run(s) on record")
    scout = forge_for(Run(OUTPUT_ROOT / "_scout"))
    _drive(scout, forge_for, args, history)
    vlog(f"library index: {OUTPUT_ROOT / 'index.md'}")
    return 0


def _drive(scout: Forge, forge_for, args, history: list) -> int:
    completed = 0
    while args.forever or completed < args.runs:
        try:
            if completed == 0 and args.seed:
                seed = args.seed
            else:
                started = time.time()
                vlog("choosing the next topic (asking the review model)...")
                with heartbeat("next topic"):
                    seed = scout.next_seed(history)
                vlog(f"topic chosen in {_dur(time.time() - started)}")
        except Exception as exc:
            if getattr(exc, "status_code", None) in (401, 403):
                raise SystemExit(f"the provider refused the request ({exc}); rerun with another --provider or --model")
            log(f"seed selection failed ({type(exc).__name__}: {exc}); retrying in {max(args.pause, 60)}s")
            time.sleep(max(args.pause, 60))  # never hot-loop a failing API
            continue
        try:
            summary = research_run(forge_for, seed, args)
            history.append({"seed": summary["seed"], "tally": summary["tally"]})
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # one bad run must not end a week-long session; the state is on disk
            log(f"run failed on seed '{seed[:60]}': {type(exc).__name__}: {exc}")
            history.append({"seed": seed, "tally": {"failed": 1}})
        completed += 1
        if (args.forever or completed < args.runs) and args.pause:
            vlog(f"{completed} run(s) done; next in {args.pause}s (Ctrl-C to stop)")
            time.sleep(args.pause)
    return completed


def _selftest() -> None:
    """Offline check of the parsing and execution plumbing (no API calls)."""
    assert _slug("Additive Structure of Squarefree Numbers!") == "additive-structure-of-squarefree-numbers"
    assert _json_block('noise ```json\n[{"id": "c1"}]\n``` tail') == [{"id": "c1"}]
    assert _json_block('prose {"verdict": "VALID"} more') == {"verdict": "VALID"}
    # shapes a live run actually produced: bare concatenated objects, no brackets
    assert _json_block('{"id":"c1"}\n{"id":"c2"}') == [{"id": "c1"}, {"id": "c2"}]
    # truncated tail: the complete object survives, the partial one is dropped
    assert _json_block('{"id":"c1"}\n{"id":"c2","x":') == {"id": "c1"}
    assert _json_block("[{\"a\":1}]\n[{\"a\":2}]") == [{"a": 1}, {"a": 2}]
    # an echoed example object before the real answer used to crash `.get`
    assert _json_object('{"a": 1}\n{"verdict": "VALID"}', "verdict") == {"verdict": "VALID"}
    try:
        _json_object('["no", "object"]', "verdict")
        raise AssertionError("a reply without the key was accepted")
    except ValueError:
        pass
    assert check_passed({"exit_code": 0, "output": "lemma 1 ok\nALL CHECKS PASSED"})
    assert not check_passed({"exit_code": 0, "output": "CHECK FAILED: lemma 2\nALL CHECKS PASSED"})
    assert not check_passed({"exit_code": 1, "output": "ALL CHECKS PASSED\nTraceback"})
    clipped = _clip("COUNTEREXAMPLE: n=3\n" + "x" * 20000 + "\nend")
    assert clipped.startswith("COUNTEREXAMPLE: n=3") and clipped.endswith("end") and len(clipped) < 8100
    assert _code_block("text ```python\nprint(1)\n``` text") == "print(1)"
    assert _code_block("```lean4\ntheorem t : 1 = 1 := rfl\n```") == "theorem t : 1 = 1 := rfl"
    assert _code_block("```\nimport Mathlib\n```") == "import Mathlib"
    assert _code_block("bare code") == "bare code"
    assert find_lean_project("D:/definitely/not/here") is None
    # the "NO COUNTEREXAMPLE" substring trap that a live run walked straight into
    assert classify_search(0, "NO COUNTEREXAMPLE, n<=10^6, 3141 cases") == "clean"
    assert classify_search(0, "COUNTEREXAMPLE: n=12, lhs=3 rhs=4") == "refuted"
    # spellings live scripts used; each was published as a refutation
    for clean_line in ("NO-COUNTEREXAMPLE: checked n=1..12", "NO_COUNTEREXAMPLES found", "No counterexample up to 9"):
        assert classify_search(0, clean_line) == "clean", clean_line
    assert classify_search(0, "COUNTEREXAMPLE: n=5\nNO COUNTEREXAMPLE elsewhere") == "inconclusive"
    assert classify_search(0, "SANITY FAILED\nNO COUNTEREXAMPLE") == "inconclusive"
    assert classify_search(1, "NO COUNTEREXAMPLE") == "inconclusive"
    assert classify_search(0, "finished") == "inconclusive"
    good = lean_verdict("theorem t : True := trivial", 0, "'t' does not depend on any axioms")
    assert good == {"compiles": True, "sorry_free": True, "sorries": 0, "axioms": 0,
                    "unexpected_axioms": []}, good
    classical = lean_verdict("theorem t : True := trivial", 0,
                             "'t' depends on axioms: [propext, Classical.choice, Quot.sound]")
    assert classical["sorry_free"], classical
    # what the source regex cannot see, Lean's own report does
    for hidden in ("sorryAx", "Lean.ofReduceBool", "h"):
        v = lean_verdict("theorem t : True := x", 0, f"'t' depends on axioms: [propext, {hidden}]")
        assert not v["sorry_free"] and v["unexpected_axioms"] == [hidden], v
    # no report at all means nothing was checked
    assert not lean_verdict("theorem t : True := trivial", 0, "")["sorry_free"]
    probe = _axiom_probe("namespace A\ntheorem t : True := trivial\nend A\n"
                         "@[simp] private lemma u : True := trivial\ntheorem _root_.v : True := trivial")
    full = "\nset_option pp.fullNames true in\n#print axioms "
    assert probe == f"{full}A.t{full}u{full}v", probe
    # the shapes the re-review broke the first probe with
    tricky = ("/- a comment:\n  lemma 2 gives the bound\n  theorem statement: x -/\n"
              "theorem «my thm» : True := trivial\ntheorem main.{u} : True := trivial\n"
              "set_option maxHeartbeats 400000 in theorem big : True := trivial\n"
              "nonrec theorem nr : True := trivial\n/-- doc -/ theorem doc : True := trivial")
    names, declared = _theorems(tricky)
    assert names == ["«my thm»", "main", "big", "nr", "doc"] and declared == 5, (names, declared)
    # a theorem the probe cannot name is not verified, even if its neighbour reports clean
    shy = "theorem ok : True := trivial\ntheorem 2bad : True := trivial"
    assert _theorems(shy) == (["ok"], 2)
    assert not lean_verdict(shy, 0, "'ok' does not depend on any axioms")["sorry_free"]
    # an error only on the probe's own lines is not sent for repair
    real_run_lean = globals()["run_lean"]
    globals()["run_lean"] = lambda code, *a: (1, "x/MathForge/c1_lean.lean:9:0: error: unknown constant")
    try:
        assert _run_lean_probed("theorem t : True := trivial", HERE, HERE, "c1_lean")[0] == 0
        globals()["run_lean"] = lambda code, *a: (1, "x/MathForge/c1_lean.lean:1:0: error: type mismatch")
        assert _run_lean_probed("theorem t : True := trivial", HERE, HERE, "c1_lean")[0] == 1
        # a truncated last proof errors on the probe line too, and must be repaired
        globals()["run_lean"] = lambda code, *a: (1, "x/MathForge/c1_lean.lean:9:0: error: unexpected token '#print'; expected term")
        assert _run_lean_probed("theorem t : True := by", HERE, HERE, "c1_lean")[0] == 1
    finally:
        globals()["run_lean"] = real_run_lean
    # twenty `NO COUNTEREXAMPLE` lines must not use up the witness's cap
    hidden = _clip("".join(f"n={i}: NO COUNTEREXAMPLE\n" for i in range(600))
                   + "COUNTEREXAMPLE: n=600\n" + "trace\n" * 1500 + "NO COUNTEREXAMPLE")
    assert classify_search(0, hidden) == "inconclusive", classify_search(0, hidden)
    assert not BOUNDED_SEED.match("For all planar graphs of maximum degree at most 4")
    # _clip keeps a verdict line from the middle of a long output
    noisy = _clip("x\n" * 3000 + "declaration uses 'sorry'\n" + "CHECK FAILED: lemma 3\n" + "y\n" * 5000)
    assert "declaration uses 'sorry'" in noisy and "CHECK FAILED: lemma 3" in noisy and len(noisy) < 9000
    assert not check_passed({"exit_code": 0, "output": _clip("ok\n" * 3000 + "CHECK FAILED: l\n"
                                                              + "ok\n" * 5000 + "ALL CHECKS PASSED")})
    # an `axiom` compiles silently and would beat a sorry count of zero
    smuggled = lean_verdict("axiom h : False\ntheorem t : False := h", 0, "")
    assert smuggled["compiles"] and not smuggled["sorry_free"] and smuggled["axioms"] == 1, smuggled
    dressed = lean_verdict("@[simp] private axiom h : False\ntheorem t : False := h", 0, "")
    assert dressed["axioms"] == 1 and not dressed["sorry_free"], dressed
    sorry = lean_verdict("theorem t : True := by sorry", 0, "declaration uses 'sorry'")
    assert not sorry["sorry_free"] and sorry["sorries"] == 1, sorry
    assert not lean_verdict("theorem t : True := trivial", 1, "")["sorry_free"]
    assert _for_paper({"status": "ok", "lean": {"code": "x", "output": "y", "compiles": True}}) == {
        "status": "ok",
        "lean": {"output": "y", "compiles": True},
    }

    tmp = OUTPUT_ROOT / "_selftest"
    shutil.rmtree(tmp, ignore_errors=True)  # the cache test needs a virgin run dir
    tmp.mkdir(parents=True, exist_ok=True)
    rc, out = run_code("print('NO COUNTEREXAMPLE up to 10')", tmp, "t.py")
    assert rc == 0 and "NO COUNTEREXAMPLE" in out, (rc, out)
    rc, out = run_code("raise SystemExit(3)", tmp, "t.py")
    assert rc == 3, rc
    rc, out = run_code("import time; time.sleep(5)", tmp, "t.py", timeout=1)
    assert rc == -1 and "TIMEOUT" in out, (rc, out)

    if not _lake():
        rc, out = run_lean("theorem t : 1 = 1 := rfl", tmp, tmp, "t")
        assert rc == -2 and "lake not found" in out, (rc, out)
        assert (tmp / "t.lean").exists(), "lean source not archived in the run directory"

    class _StubAI:
        def __init__(self, replies):
            self.replies, self.seen = list(replies), []

        def generate_content(self, prompt, **kwargs):
            self.seen.append(prompt)
            return self.replies.pop(0)

    stub = _StubAI(["```python\nprint('searching')\n```", "```python\nprint('COUNTEREXAMPLE: n=1')\n```"])
    result = Forge(stub, Run(tmp)).write_and_run("brief", "stub", markers=("COUNTEREXAMPLE",))
    assert result["repairs"] == 1, result["repairs"]
    assert "COUNTEREXAMPLE: n=1" in result["output"], result["output"]
    assert "never printed a verdict" in stub.seen[1], "silent run was not reported back to the agent"

    # a reply that only announces a tool call is asked again, once, not errored
    stub = _StubAI(["Suspicious proof -- I'll brute-force it.", '{"verdict": "GAPS"}'])
    assert Forge(stub, Run(tmp)).referee({"statement": "s"}, "p") == {"verdict": "GAPS"}
    assert "no tools" in stub.seen[1], stub.seen[1]
    # ... up to MAX_JSON_RETRIES times, then the conjecture errors
    stub = _StubAI(["Checking..."] * MAX_JSON_RETRIES + ['{"verdict": "VALID"}'])
    assert Forge(stub, Run(tmp)).referee({"statement": "s"}, "p") == {"verdict": "VALID"}
    stub = _StubAI(["Checking..."] * (MAX_JSON_RETRIES + 1) + ['{"verdict": "VALID"}'])
    try:
        Forge(stub, Run(tmp)).referee({"statement": "s"}, "p")
        raise AssertionError("more replies without JSON than MAX_JSON_RETRIES were accepted")
    except ValueError:
        pass
    assert len(stub.seen) == MAX_JSON_RETRIES + 1, f"asked {len(stub.seen)} times"

    # the repair of a review-model script stays on the review model
    class _RoleAI(_StubAI):
        def generate_content(self, prompt, **kwargs):
            self.seen.append(kwargs.get("model_type"))
            return self.replies.pop(0)

    roles = _RoleAI(["```python\nraise SystemExit(2)\n```", "```python\nprint('ALL CHECKS PASSED')\n```"])
    Forge(roles, Run(tmp)).write_and_run("brief", "role", markers=("ALL CHECKS PASSED",), model_type="review")
    assert roles.seen == ["review", "review"], roles.seen

    # Consumer uses the public service contract, including distinct roles on one model.
    fake = object.__new__(AIService)
    fake.provider = 'openai'
    assert set_reasoning_effort(fake, 'high', 'low')
    assert fake._reasoning_options('writing', responses=True) == {'reasoning': {'effort': 'high'}}
    assert fake._reasoning_options('review', responses=True) == {'reasoning': {'effort': 'low'}}
    assert set_reasoning_effort(fake, 'provider-default')
    assert fake._reasoning_options('review') == {}

    # proposal fan-out: one call each, duplicates and dud replies dropped, rest kept
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    stub = _StubAI([
        '{"title": "First", "statement": "s1"}',
        "the model ran out of budget mid-thought and said nothing useful",
        '{"title": "First", "statement": "s1 again"}',
        '```json\n{"title": "Second", "statement": "s2"}\n```',
        "still nothing useful",
    ])
    proposed = Forge(stub, Run(tmp)).propose("seed", 4)
    assert [c["id"] for c in proposed] == ["c1", "c2"], proposed
    assert [c["title"] for c in proposed] == ["First", "Second"], proposed
    assert "proposer 4 of 4" in stub.seen[3], stub.seen[3]
    # the dud proposer is asked once more, and only it
    assert len(stub.seen) == 5 and "proposer 2 of 4" in stub.seen[4], stub.seen[4:]
    # the dud reply is not cached, so a resume asks proposer 2 again
    assert "conjecture2" not in Run(tmp).data and "conjecture1" in Run(tmp).data
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    # a relative run dir used to make scripts resolve against their own cwd twice
    relative = Run(Path(tmp.relative_to(Path.cwd())) if tmp.is_relative_to(Path.cwd()) else tmp)
    assert relative.path.is_absolute(), relative.path
    rc, out = run_code("print('ok')", relative.path, "rel.py")
    assert rc == 0 and "ok" in out, (rc, out)

    # literature retrieval: parse canned payloads, no network
    atom = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>http://arxiv.org/abs/1234.5678</id><title>On  binomial
      sums</title><summary>We prove a  congruence.</summary></entry></feed>"""
    crossref = json.dumps(
        {"message": {"items": [{"title": ["A related paper"], "abstract": "<p>Body</p>", "DOI": "10/x"}]}}
    )
    openalex = json.dumps(
        {"results": [{"title": "Inverted abstract paper",
                      "abstract_inverted_index": {"second": [1], "First": [0], "third": [2]},
                      "doi": "https://doi.org/10/y", "id": "https://openalex.org/W1"}]}
    )

    def _canned(url, timeout=30):
        if "arxiv" in url:
            return atom
        if "openalex" in url:
            return openalex
        return crossref

    real_get = globals()["_http_get"]
    globals()["_http_get"] = _canned
    real_arxiv_delay = globals()["ARXIV_DELAY"]
    globals()["ARXIV_DELAY"] = 0.0  # keep the selftest off the real 3s arXiv spacing
    _LIMITERS.clear()
    try:
        found = literature(["binomial sums", "binomial sums"], rows=1)  # repeated on purpose
    finally:
        globals()["_http_get"] = real_get
        globals()["ARXIV_DELAY"] = real_arxiv_delay
        _LIMITERS.clear()
    assert found["errors"] == [], found["errors"]
    assert [h["title"] for h in found["hits"]] == [
        "On binomial sums", "A related paper", "Inverted abstract paper",
    ], found["hits"]
    assert found["hits"][0]["url"] == "http://arxiv.org/abs/1234.5678"
    assert found["hits"][1]["abstract"] == "Body", found["hits"][1]
    # inverted index must come back in position order, not dict order
    assert found["hits"][2]["abstract"] == "First second third", found["hits"][2]

    globals()["_http_get"] = lambda url, timeout=30: (_ for _ in ()).throw(OSError("no network"))
    try:
        offline = literature(["x"], rows=1)
    finally:
        globals()["_http_get"] = real_get
        _LIMITERS.clear()
    assert offline["hits"] == [] and len(offline["errors"]) == len(_backends()), offline

    # a throttled request (OpenAlex 429) is retried once after Retry-After
    import io

    class _Reply(io.BytesIO):
        __enter__ = lambda self: self
        __exit__ = lambda self, *a: None

    attempts, real_open = [], urllib.request.urlopen

    def _throttled(request, timeout=30):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "0"}, None)
        return _Reply(b"ok")

    urllib.request.urlopen = _throttled
    try:
        assert _http_get("https://x") == "ok" and len(attempts) == 2, attempts
    finally:
        urllib.request.urlopen = real_open

    # the spacing requirement is global: back-to-back waits pay the gap
    limiter = _RateLimiter(0.1)
    started = time.time()
    limiter.wait()
    limiter.wait()
    assert time.time() - started >= 0.1

    # status ladder: a trivialized statement is not machine-verified, and
    # faithfulness is its own stage so a bad reply does not cost the Lean run
    class _FakeForge:
        lean_project = True
        search = True

        def __init__(self, run, faith):
            self.run, self.faith, self.lean_runs = run, faith, 0

        falsify = staticmethod(lambda c: {"exit_code": 0, "output": "NO COUNTEREXAMPLE n<=9"})
        lean_refute = staticmethod(lambda c, w: {"code": "theorem refutation", "compiles": True,
                                                 "sorry_free": True, "axioms": 0})
        novelty = staticmethod(lambda c: {"verdict": "APPARENTLY_NEW"})
        prove = staticmethod(lambda c, e: "proof")
        referee = staticmethod(lambda c, p: {"verdict": "VALID"})
        independent_check = staticmethod(lambda c, p: {"exit_code": 0, "output": "ALL CHECKS PASSED"})

        def lean(self, c, proof):
            self.lean_runs += 1
            return {"code": "theorem t", "compiles": True, "sorry_free": True, "axioms": 0}

        def faithfulness(self, c, code, negated=False):
            if isinstance(self.faith, Exception):
                raise self.faith
            return self.faith

    for faith, expected in (
        ({"verdict": "FAITHFUL", "trivialized": False}, "machine-verified"),
        ({"verdict": "FAITHFUL", "trivialized": "true"}, "verified"),
        ({"verdict": "NARROWER"}, "verified"),
    ):
        shutil.rmtree(tmp, ignore_errors=True)
        got = pipeline(_FakeForge(Run(tmp), faith), {"id": "c1", "statement": "s"})["status"]
        assert got == expected, (faith, got)
    shutil.rmtree(tmp, ignore_errors=True)
    flaky = _FakeForge(Run(tmp), ValueError("garbled"))
    assert run_one(flaky, {"id": "c1", "statement": "s"})["status"] == "error"
    flaky.faith = {"verdict": "FAITHFUL"}
    assert pipeline(flaky, {"id": "c1", "statement": "s"})["status"] == "machine-verified"
    assert flaky.lean_runs == 1, "the Lean run was redone because faithfulness failed"
    # novelty cached with backend errors is searched again on resume; a clean one is kept
    flaky.run.data["c1.novelty"] = {"verdict": "UNCLEAR", "search_errors": ["search_arxiv: 406"]}
    pipeline(flaky, {"id": "c1", "statement": "s"})
    assert flaky.run.data["c1.novelty"] == {"verdict": "APPARENTLY_NEW"}, flaky.run.data["c1.novelty"]
    flaky.run.data["c1.novelty"] = {"verdict": "UNCLEAR", "search_errors": []}
    pipeline(flaky, {"id": "c1", "statement": "s"})
    assert flaky.run.data["c1.novelty"]["verdict"] == "UNCLEAR"

    # a reported witness is refuted only after a second script re-checks it, and
    # machine-refuted only once Lean proves the negation faithfully
    for confirm_out, lean_on, expected in (
        ("REFUTATION REJECTED: p=5 is outside p >= 7", True, "inconclusive"),
        ("REFUTATION CONFIRMED: p=7", False, "refuted"),
        ("REFUTATION CONFIRMED: p=7", True, "machine-refuted"),
    ):
        shutil.rmtree(tmp, ignore_errors=True)
        liar = _FakeForge(Run(tmp), {"verdict": "FAITHFUL"})
        liar.lean_project = lean_on
        liar.falsify = lambda c: {"exit_code": 0, "output": "COUNTEREXAMPLE: p=5"}
        liar.confirm_refutation = lambda c, out, o=confirm_out: {"exit_code": 0, "output": o}
        got = pipeline(liar, {"id": "c1", "statement": "s"})
        assert got["status"] == expected, (confirm_out, lean_on, got["status"])
    shutil.rmtree(tmp, ignore_errors=True)

    # one crashed conjecture must not kill the run
    real_pipeline = globals()["pipeline"]
    boom, fine = {"id": "c1", "statement": "s"}, {"id": "c2", "statement": "s"}

    def _maybe_boom(forge, c):
        if c is boom:
            raise ValueError("model gibberish")
        return {**c, "status": "verified"}

    globals()["pipeline"] = _maybe_boom
    try:
        isolated = [run_one(None, c) for c in (boom, fine)]
    finally:
        globals()["pipeline"] = real_pipeline
    assert isolated[0]["status"] == "error" and "ValueError" in isolated[0]["error"], isolated[0]
    assert isolated[1]["status"] == "verified", isolated[1]

    negatives = negative_results(
        "seed",
        [
            {"id": "c1", "title": "Kept", "status": "verified", "statement": "s"},
            {"id": "c2", "title": "Broken", "status": "refuted", "statement": "s",
             "falsification": {"output": "sanity ok\nCOUNTEREXAMPLE: n=7, lhs=1 rhs=2"}},
            {"id": "c3", "title": "Old news", "status": "known", "statement": "s",
             "novelty": {"reasoning": "it is Wilson", "closest_known_results": ["Wilson"]},
             "falsification": {"output": "NO COUNTEREXAMPLE up to 10^6"}},
            {"id": "c4", "title": "Crashed", "status": "error", "statement": "s",
             "error": "ValueError: model gibberish"},
        ],
    )
    assert "Kept" not in negatives, "a surviving conjecture leaked into the negative record"
    assert "**Counterexample.** `n=7, lhs=1 rhs=2`" in negatives
    assert "Wilson" in negatives and "NO COUNTEREXAMPLE up to 10^6" in negatives
    assert "ValueError: model gibberish" in negatives, "an errored conjecture was not recorded"

    # bounded computations are flagged in the history the seed chooser reads
    for bounded in ("For n ≤ 10, enumerate all Motzkin paths", "For all 4x4 matrices over F_2",
                    "For all connected 3-regular graphs on at most 10 vertices, test"):
        assert BOUNDED_SEED.match(bounded), bounded
    assert not BOUNDED_SEED.match("additive structure of squarefree numbers")
    assert not BOUNDED_SEED.match("Forbidden patterns in binary words")
    seen_prompts = []
    scout = Forge(type("AI", (), {"generate_content": lambda self, p, **k: (seen_prompts.append(p), '{"seed": "x"}')[1]})(), Run(tmp))
    scout.next_seed([{"seed": "For n ≤ 10, enumerate X", "tally": {"verified": 3}}])
    assert "do not imitate" in seen_prompts[0] and '"verified": 3' not in seen_prompts[0]

    # a resume that turns an error into a keeper rewrites the paper
    shutil.rmtree(tmp, ignore_errors=True)
    papers = []

    class _PaperForge:
        def __init__(self, run):
            self.run = run

        def propose(self, seed, n, w):
            return [{"id": "c1", "statement": "s"}, {"id": "c2", "statement": "t"}]

        def paper(self, seed, keep):
            papers.append([r["id"] for r in keep])
            return "paper"

    outcomes = {"c1": "verified", "c2": "error"}
    real_run_one, real_root = globals()["run_one"], globals()["OUTPUT_ROOT"]
    globals()["run_one"] = lambda forge, c: {**c, "status": outcomes[c["id"]]}
    globals()["OUTPUT_ROOT"] = tmp
    try:
        opts = argparse.Namespace(conjectures=2, workers=1, publish=False)
        research_run(_PaperForge, "seed", opts, run=Run(tmp / "p"))
        outcomes["c2"] = "verified"
        research_run(_PaperForge, "seed", opts, run=Run(tmp / "p"))
        research_run(_PaperForge, "seed", opts, run=Run(tmp / "p"))  # unchanged: cached
    finally:
        globals()["run_one"], globals()["OUTPUT_ROOT"] = real_run_one, real_root
    assert papers == [["c1"], ["c1", "c2"]], papers
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    # progress logging helpers
    assert _dur(12.4) == "12s" and _dur(200) == "3.3m"
    assert _tag("c12.falsify") == "c12" and _tag("paper") == "" and _tag("conjectures") == ""
    assert _verdict_line("checking...\nCOUNTEREXAMPLE: n=7\ndone") == "COUNTEREXAMPLE: n=7"
    assert _verdict_line("a\nb") == "b" and _verdict_line("  \n") == "(no output)"
    assert _verdict_line("x" * 200).endswith("x") and len(_verdict_line("x" * 200)) == 88

    ticks = []
    real_log, real_live = globals()["log"], LIVE
    globals()["log"] = lambda msg, tag="": ticks.append(msg)
    globals()["LIVE"] = False  # piped: heartbeats are appended log lines
    try:
        with heartbeat("slow", every=0.05):
            time.sleep(0.2)
        quiet_after_exit = len(ticks)
        time.sleep(0.15)  # the ticker must stop with the block, not outlive it
    finally:
        globals()["log"], globals()["LIVE"] = real_log, real_live
    assert ticks and "still running" in ticks[0], ticks
    assert len(ticks) == quiet_after_exit, ticks

    # terminal: heartbeats redraw one status line, and leave nothing behind
    import io
    screen = io.StringIO()
    globals()["LIVE"] = True
    try:
        with contextlib.redirect_stdout(_LiveStdout(screen)):
            with heartbeat("prove", "c4", every=0.05):
                time.sleep(0.2)
                seen = dict(_live)
                log("result line", "c4")
                print("[opencode] retry notice")  # foreign output, as ai_suite prints it
                shown = screen.getvalue()
                print("Enter key: ", end="")  # an open prompt: the ticker must not draw over it
                before = screen.getvalue()
                time.sleep(0.15)
                assert screen.getvalue() == before, screen.getvalue()[len(before):]
                print()
    finally:
        globals()["LIVE"] = real_live
    assert list(seen.values())[0].startswith("c4 prove "), seen
    assert not _live, _live
    # each permanent line starts clean: the status line was wiped before it
    for line in ("result line", "[opencode] retry notice"):
        head = shown.split(line)[0].rsplit("\n", 1)[-1].rsplit("\r", 1)[-1]  # what the row shows
        assert "prove" not in head, repr(head)
    assert "c4 prove" in shown.rsplit("\n", 1)[-1], repr(shown[-60:])  # redrawn below it

    # live_bar: leads the status line, redrawn in place per step, gone at the end
    screen = io.StringIO()
    globals()["LIVE"] = True
    try:
        with contextlib.redirect_stdout(_LiveStdout(screen)):
            _live["other"] = "conjectures 3s"
            with live_bar("proposing", 2) as step:
                assert list(_live.values())[0] == "proposing [....................] 0/2", _live
                assert step() == 1
                assert list(_live.values())[0].endswith("1/2"), _live
            assert list(_live.values()) == ["conjectures 3s"], _live
            _live.pop("other")
    finally:
        globals()["LIVE"] = real_live
    assert "\n" not in screen.getvalue(), repr(screen.getvalue())  # all in place, no new lines

    # publishing: only machine verdicts qualify, and a failed post is retried
    assert publishable({"status": "machine-verified"}) and publishable({"status": "machine-refuted"})
    assert not any(publishable({"status": s}) for s in ("verified", "provisional", "known", "inconclusive", "refuted"))
    mixed = [{"status": "verified"}, {"status": "machine-refuted"}]
    assert publish_verdict(False, mixed, []).startswith("publish: OFF") and "1 result" in publish_verdict(False, mixed, [])
    assert "NOTHING PUBLISHED" in publish_verdict(True, [{"status": "verified"}, {"status": "known"}], [])
    assert "0/1 posted" in publish_verdict(True, mixed, [])
    assert publish_verdict(True, mixed, [{"url": "u"}]) == "publish: 1 gist(s) posted"
    assert _counterexample_line("checks ok\nNO COUNTEREXAMPLE here\nCOUNTEREXAMPLE: n=7") == "COUNTEREXAMPLE: n=7"
    assert _counterexample_line("NO COUNTEREXAMPLE up to 10^6") == ""
    assert _gist_url("Creating gist\nhttps://gist.github.com/u/abc123\n") == "https://gist.github.com/u/abc123"
    assert _gist_url("nothing here") == ""

    negation = _publication("seed", {
        "id": "c1", "title": "Broken", "headline": "Every n is odd.", "status": "machine-refuted",
        "statement": "s", "notation": "n",
        "falsification": {"output": "sanity ok\nCOUNTEREXAMPLE: n=5, lhs=1 rhs=2"},
        "confirmation": {"exit_code": 0, "output": "REFUTATION CONFIRMED: n=7, lhs=1 rhs=2"},
        "lean": {"sorries": 0, "faithfulness": {"verdict": "FAITHFUL"}},
    })
    # verdict first, and the re-checked witness rather than the searcher's
    assert negation.startswith("# Refuted: Every n is odd\n\n**Verdict: FALSE.**"), negation[:120]
    assert "REFUTATION CONFIRMED: n=7" in negation and "n=5" not in negation and "rediscovery" in negation
    assert headline({"id": "c3", "title": "T", "status": "machine-verified"}) == "Proved: T"
    proved = _publication("seed", {
        "id": "c2", "title": "Kept", "status": "machine-verified", "statement": "s",
        "proof": "PROOF BODY", "falsification": {"output": "NO COUNTEREXAMPLE up to 10^6"},
        "independent_check": {"output": "ALL CHECKS PASSED"},
        "novelty": {"verdict": "APPARENTLY_NEW", "evidence_base": "retrieval", "retrieved": [], "queries": ["q"]},
        "lean": {"sorries": 0, "faithfulness": {"verdict": "FAITHFUL", "backtranslation": "BT"}},
    })
    assert "PROOF BODY" in proved and "FAITHFUL" in proved and "NO COUNTEREXAMPLE up to 10^6" in proved
    # an artifact's own model wins over the run pair; nothing recorded says so
    credited = _publication("seed", {"id": "c1", "status": "machine-refuted", "statement": "s",
                                     "confirmation": {"output": "", "model": "checker-x"}},
                            {"work": "work-y", "review": "review-z"})
    assert "- Independent re-check of the witness: checker-x" in credited
    assert "- Counterexample search: work-y" in credited and "- Proposed the claim: review-z" in credited
    assert "- Wrote the proof: not recorded" in proved

    pub_run = Run(OUTPUT_ROOT / "_publishtest")
    posts = []
    real_publish_gist = globals()["publish_gist"]
    globals()["publish_gist"] = lambda run, seed, r: (
        posts.append(r["id"]),
        {"error": "offline"} if len(posts) == 1 else {"url": "https://gist.github.com/x", "title": r["id"]},
    )[1]
    try:
        one = {"id": "c1", "status": "machine-refuted", "statement": "s"}
        assert publish(pub_run, "seed", [one, {"id": "c2", "status": "known"}]) == []
        assert publish(pub_run, "seed", [one])[0]["url"] == "https://gist.github.com/x"
        assert publish(pub_run, "seed", [one])  # cached, no third post
        assert posts == ["c1", "c1"], posts
    finally:
        globals()["publish_gist"] = real_publish_gist
        shutil.rmtree(pub_run.path, ignore_errors=True)

    # continuous mode bookkeeping
    real_root = globals()["OUTPUT_ROOT"]
    globals()["OUTPUT_ROOT"] = tmp
    try:
        first = unique_run_dir("A Topic")
        assert first == tmp / "a-topic", first
        first.mkdir(parents=True, exist_ok=True)
        (first / "state.json").write_text("{}", encoding="utf-8")
        assert unique_run_dir("A Topic") == tmp / "a-topic-2"
        assert latest_run_dir() == first, latest_run_dir()
        newer = tmp / "b-topic"
        newer.mkdir(parents=True, exist_ok=True)
        (newer / "state.json").write_text("{}", encoding="utf-8")
        os.utime(newer / "state.json", (time.time() + 60, time.time() + 60))
        assert latest_run_dir() == newer, latest_run_dir()
        scratch = tmp / "_scout"  # scratch dirs are not runs
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / "state.json").write_text("{}", encoding="utf-8")
        os.utime(scratch / "state.json", (time.time() + 120, time.time() + 120))
        assert latest_run_dir() == newer, latest_run_dir()
        append_index({"seed": "A Topic", "finished": "now", "tally": {"verified": 1}, "path": "a-topic", "paper": "a-topic/paper.md"})
        append_index({"seed": "B Topic", "finished": "later", "tally": {"known": 2}, "path": "b-topic", "paper": None})
        assert len(json.loads((tmp / "index.json").read_text(encoding="utf-8"))) == 2
        listing = (tmp / "index.md").read_text(encoding="utf-8")
        assert "a-topic/paper.md" in listing and "b-topic" in listing, listing
        # a resumed run replaces its entry rather than listing twice
        append_index({"seed": "A Topic", "finished": "again", "tally": {"verified": 2}, "path": "a-topic", "paper": None})
        entries = json.loads((tmp / "index.json").read_text(encoding="utf-8"))
        assert [e["finished"] for e in entries] == ["later", "again"], entries
        # a damaged index is set aside, never silently overwritten
        (tmp / "index.json").write_text("{broken", encoding="utf-8")
        append_index({"seed": "C", "finished": "x", "tally": {}, "path": "c", "paper": None})
        assert len(json.loads((tmp / "index.json").read_text(encoding="utf-8"))) == 1
        assert list(tmp.glob("index.damaged-*.json")), "damaged index was discarded"
        (tmp / "index.json").write_text('{"valid": "but not a list"}', encoding="utf-8")
        append_index({"seed": "D", "finished": "x", "tally": {}, "path": "d", "paper": None})
        assert len(json.loads((tmp / "index.json").read_text(encoding="utf-8"))) == 1
    finally:
        globals()["OUTPUT_ROOT"] = real_root

    run = Run(tmp)
    calls = []
    for _ in range(2):
        run.stage("k", lambda: (calls.append(1), "v")[1])
    assert calls == [1], "stage recomputed a cached value"
    # save() is atomic: valid JSON on disk, no temp file left behind
    json.loads(run.file.read_text(encoding="utf-8"))
    assert not list(tmp.glob("state.json.tmp")), "temp file left behind"
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        try:
            raise SystemExit(main())
        except KeyboardInterrupt:
            # covers --resume too; the stage that was cut is simply not cached
            log("stopped; every finished stage is cached on disk, --resume picks it up")
            raise SystemExit(130)
