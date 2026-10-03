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
    python mathforge.py --forever --publish          # public results-repo folder per machine-checked result
    python mathforge.py --publish-existing           # publish results already on disk, then exit
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
model's opinion -- `machine-verified` and `machine-refuted` -- are published as
folders of a public GitHub repository (with a Palomar-ready Lake project where the
Lean file splits cleanly) through the `gh` CLI. Nothing is published without that flag.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
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
DEFAULT_REVIEW_MODEL = "deepseek-v4.1-flash"
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
DEFAULT_EFFORT = "medium"  # the work model: proving, falsifying, formalizing
DEFAULT_REVIEW_EFFORT = "high"  # the review model: the gates are worth the extra thinking
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "provider-default")
USER_AGENT = "mathforge/1.0 (automated novelty check; contact: local user)"
ARXIV_DELAY = 3.0  # arXiv asks for one request every 3 seconds
AIRAXIV_DELAY = 1.0  # a small site; its novelty search is a page fetch
SEARCH_ROWS = 5

RULES = """\
Hard constraints on every conjecture you produce:
- It must be ORIGINAL. Do not restate a named theorem, a textbook exercise, a
  known identity, or a famous open problem (Collatz, Goldbach, twin primes,
  Riemann, ABC, Erdos-Straus, ...) in full. A tractable special case, analogue
  or strengthening of an open question that specialists work on is welcome, and
  is the best kind of target.
- It must MATTER. Ask what a specialist would do with it. Prefer an exact formula,
  a bijection, a sharp extremal bound with the extremal objects characterized, a
  structure or classification theorem, or an unexpected link between two
  invariants. Avoid a parity or divisibility curiosity about a count refined by
  arbitrarily chosen statistics, and anything whose whole proof is one obvious
  symmetry (a reversal involution, a free group action): those are true, new and
  worthless. Be adventurous: a bold claim that may be refuted is worth more than
  a safe one nobody would cite.
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
- It must be PROVABLE by an argument that can be written out in full (induction,
  bijections, transfer matrices, generating functions, linear algebra, character
  sums, elementary number theory, extremal or probabilistic counting). A long,
  multi-step proof is welcome. It is out of scope only if the one plausible proof
  rests on machinery that cannot be reproduced on the page.
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


def _short(text: str, limit: int) -> str:
    """`text` cut to `limit` characters, the cut marked with an ellipsis."""
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _named(stem: str, text: str) -> str:
    """A log line about one package: `name: text`."""
    return f"{_short(stem, 44)}: {text}"


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


def search_airaxiv(query: str, rows: int = SEARCH_ROWS) -> list:
    """AiraXiv's public paper search, an HTML page. It holds AI-generated papers no other index here covers,
    this pipeline's own earlier uploads among them, so a result already there is not written up twice."""
    # ponytail: the site matches the query as one phrase, so a model-written query rarely hits; the fallback
    # asks by its longest word alone. Use the site's recommender (MCP search_related_papers) if that proves thin
    words = sorted(re.findall(r"[\w-]{6,}", query), key=len, reverse=True)
    for q in dict.fromkeys([query, *words[:1]]):
        page = _http_get("https://airaxiv.com/papers/?q=" + urllib.parse.quote_plus(q))
        hits = []
        for item in re.findall(r'(?s)<li class="paper-item">(.*?)</li>', page)[:rows]:
            def field(name: str) -> str:
                found = re.search(rf'(?s)class="{name}"[^>]*>(.*?)</(?:div|span)>', item)
                return html.unescape(re.sub(r"<[^>]+>", "", found.group(1))).strip() if found else ""
            paper_id = field("paper-card-id")
            hits.append({"source": "AiraXiv", "title": _clean(field("paper-title"), 300),
                         "abstract": _clean(field("paper-card-abstract")),
                         "url": f"https://airaxiv.com/papers/view/{paper_id}/" if paper_id else ""})
        if hits:
            return hits
    return []


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
    backends = [(search_arxiv, ARXIV_DELAY), (search_crossref, 0.0), (search_openalex, 0.0),
                (search_airaxiv, AIRAXIV_DELAY)]
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


def resolve_efforts(flag: str | None, review_flag: str | None, asked: bool, env=os.environ) -> tuple[str, str]:
    """(work, review) effort. A flag wins. Otherwise the shared menu's pick for that
    role's model: when the menu was asked, no pick means the provider's own default;
    unattended, no remembered pick means the built-in default.
    """
    # ponytail: an unattended run cannot tell "never asked" from a remembered "default"
    # pick, so both get the built-in default; pass --effort provider-default to force it.
    blank = ("provider-default",) * 2 if asked else (DEFAULT_EFFORT, DEFAULT_REVIEW_EFFORT)
    return (flag or env.get("AI_WRITING_EFFORT") or blank[0],
            review_flag or env.get("AI_REVIEW_EFFORT") or blank[1])


def set_reasoning_effort(ai: AIService, effort: str, review_effort: str | None = None) -> bool:
    return ai.set_reasoning_effort(effort, review_effort)



class Forge:
    def __init__(self, ai: AIService, run: Run, lean_project: Path | None = None, search: bool = True):
        self.ai = ai
        self.run = run
        self.lean_project = lean_project
        self.search = search
        # set by the CLI before AIService is built; recorded so a publication can say
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
                    f"You are a research mathematician hunting for NEW theorems that matter in: {seed}\n\n"
                    f"{RULES}\n"
                    f"You are proposer {nth} of {count} working independently on this seed. "
                    "Pick an angle the others are unlikely to pick and propose exactly ONE "
                    "conjecture. Before writing it, silently check it against the literature "
                    "you know; discard anything you can name. Prefer a statement that "
                    "combines two structures in a way you have not seen combined, or that "
                    "settles a case of a question people in the area actually ask. Aim at the "
                    "most important statement you believe is true, not the safest one.\n\n"
                    "Do not survey the area first and do not weigh many candidates: settle on "
                    "one early and spend the reply making it precise.\n\n"
                    "Return ONLY a JSON object:\n"
                    '{"title": "...", "headline": "the claim itself as one plain sentence of at '
                    'most 20 words a reader grasps at a glance, e.g. every tree on p vertices has '
                    'an even number of X", "statement": "precise natural-language statement '
                    'with all quantifiers", "notation": "definitions of every symbol used", '
                    '"search_space": "the explicit finite family a program should search for a '
                    'counterexample, with concrete bounds", "why_plausible": "the heuristic or '
                    'partial argument", "why_new": "why you believe this is not in the literature", '
                    '"why_it_matters": "what open question, known theorem or line of work this '
                    'advances, and what a specialist would do with it"}',
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
            "- states the theorem formally as `theorem main_theorem`, as faithfully as possible. The statement is "
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
            "objects and (b) provable by an argument that can be written out in full, so "
            "pick an area rich in small concrete objects: integer sequences, finite words, "
            "graphs on few vertices, partitions, lattice paths, finite groups or rings, "
            "matrices over small fields, combinatorial designs, polynomial identities over Z.\n\n"
            "Be adventurous and aim at importance. Choose an area where current research "
            "has open questions (a conjecture with unsettled cases, an extremal problem "
            "with a gap between bounds, an enumeration nobody has a formula for, a "
            "structure nobody has classified) and where a tractable special case, analogue "
            "or exact answer would be cited. Name that open question in the seed. Do not "
            "choose a familiar object refined by arbitrary statistics: those runs yielded "
            "parity and divisibility curiosities that are new and of no interest.\n\n"
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

    def classify(self, r: dict) -> dict:
        """arXiv and MSC 2020 codes for a published result's formalization.yaml.
        Metadata, not a verdict; a reply that fails the format falls back to
        combinatorics, where nearly every mathforge seed lives."""
        fallback = {"arxiv": ["math.CO"], "msc2020": ["05A99"]}
        try:
            got = self.ask_json(
                "Classify this result. Reply with JSON only: {\"arxiv\": [1-2 official arXiv math "
                "categories such as \"math.CO\"], \"msc2020\": [1-3 five-character MSC 2020 codes such "
                "as \"05A15\"]}.\n\n"
                f"STATEMENT: {r.get('statement', '')}\nNOTATION: {r.get('notation', '')}",
                "arxiv", model_type="review")
        except (ValueError, RuntimeError):
            return fallback
        arxiv = [c for c in got.get("arxiv") or [] if isinstance(c, str) and re.fullmatch(r"math\.[A-Z]{2}", c)][:2]
        msc = [c for c in got.get("msc2020") or [] if isinstance(c, str) and re.fullmatch(r"\d\d[A-Z-]\d\d", c)][:8]
        return {"arxiv": arxiv or fallback["arxiv"], "msc2020": msc or fallback["msc2020"]}

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

    def revise(self, paper: str, report: str) -> str:
        """The paper rewritten against a referee report. The evidence is what it was: nothing has been
        re-run, so the rewrite may explain better but may not claim more."""
        return self.ask(
            "Revise the research paper below in response to the referee report. Return the complete "
            "revised paper in Markdown with LaTeX, keeping its sections and their numbering, and nothing "
            "else: no preamble, no code fence around it.\n\n"
            "You may change the exposition: motivation and context, definitions made precise, notation, "
            "steps of a proof written out, the place of the result among known ones, the limitations. "
            "Mention a work only if the report or the paper names it, in no more detail than they give: "
            "invent no reference, title, year or theorem number, and say that it was not consulted.\n\n"
            "A proof may gain steps and clearer wording but must lose none: it was checked as written, "
            "so do not compress, merge or re-derive it, whatever the report says about length.\n\n"
            "You may NOT change what the paper establishes: the statement of each theorem, what was "
            "searched and over which ranges, the Lean status and faithfulness verdict of each result, the "
            "novelty disclaimer and its queries. No computation, formalization or literature search has "
            "been run since the paper was written. Where the report asks for something that would need "
            "one (larger ranges, a sorry-free Lean proof, a wider literature review, a stronger theorem), "
            "do not claim it: record it in section 5 as open, attributed to the review. Where the report "
            "finds a real error in a proof, say so plainly under a `## GAP` heading in section 5 and "
            "state that the result is in doubt; do not argue it away. If the report calls a result "
            "elementary or of limited interest and that is fair, say so in the introduction.\n\n"
            "End with a section `6. Changes in response to review`: one bullet per point of the report, "
            "saying what changed or why it could not.\n\n"
            f"REFEREE REPORT:\n{report}\n\nPAPER:\n{paper}"
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


def paper_models(results: list, models: dict | None, revised_by: str | None = None) -> str:
    """The paper's closing `## Models` section: who proposed, proved and checked each result, and who
    rewrote the paper against a review, which need not be the model that wrote it."""
    lines = ["## Models", ""]
    for r in results:
        lines += [f"**{r.get('title', r['id'])}** (`{r['status']}`)", "", *_model_lines(r, models), ""]
    if revised_by:
        lines += ["**Revision in response to the AiraXiv AI review** (exposition only; no result was re-checked)", "",
                  f"- Revised against the review: {revised_by}", ""]
    return "\n".join(lines)


def negative_results(seed: str, results: list, models: dict | None = None) -> str:
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
            lines += [f"**Counterexample.** `{_witness(r)}`", "", f"**Models.** {models_used(r, models)}", ""]
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

- **machine-refuted** — an adversarial script found a witness, a separate
  model call's script re-checked it against the statement from scratch, and Lean elaborated a
  sorry-free proof of the negation of the claim.
- **machine-verified** — Lean elaborated the proof with no `sorry`.

In both cases a separate model call back-translated the Lean statement and judged it
faithful to the informal one. Lean certifies the Lean statement; the
back-translation is a screening filter, not an oracle, so read the attached
`.lean` file against the statement above.

Novelty screening is a bounded automated search over arXiv, Crossref and
OpenAlex with model-written queries. It is blind to books, to journals outside
those indexes, and to anything phrased differently: a result here may well be a
rediscovery. Corrections welcome as GitHub issues.
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
    """What a publication's title leads with: `Refuted: <claim>` / `Proved: <claim>`."""
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


def _model_rows(r: dict, models: dict | None) -> list:
    """Who did what, as (role, model) pairs. A code artifact names its own model; the rest falls back to
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
    return [(label, who(key, role)) for label, key, role in rows]


def _model_lines(r: dict, models: dict | None) -> list:
    return [f"- {label}: {model}" for label, model in _model_rows(r, models)]


def models_used(r: dict, models: dict | None) -> str:
    """The distinct models behind a result, in pipeline order, for a one-line credit."""
    return ", ".join(dict.fromkeys(m for _label, m in _model_rows(r, models) if m != "not recorded")) or "not recorded"


def _publication(seed: str, r: dict, models: dict | None = None) -> str:
    """The publication's README: verdict first, then the claim, the evidence, the caveats."""
    cid = r["id"]
    refuted = r["status"] in REFUTED
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    witness = _witness(r) if refuted else ""
    verdict = (f"**Verdict: FALSE.** Counterexample: `{witness}`" if refuted
               else "**Verdict: TRUE.** Lean 4 + Mathlib accepted a sorry-free proof.")
    lines = [f"# {headline(r)}", "", verdict, "", f"**Models:** {models_used(r, models)} (roles under *Models* below)", ""]
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
            "- Independent re-check (`*_confirm.py`, a separate model call, definitions "
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


RESULTS_REPO = os.getenv("MATHFORGE_RESULTS_REPO", "mathforge-results")  # `name` or `owner/name`
RESULTS_CHECKOUT = Path(os.getenv("MATHFORGE_RESULTS_DIR") or Path.home() / "mathforge-results")
PALOMAR_FORM = "https://submit.palomar-registry.org/"
_REPO_LOCK = threading.Lock()
# top-level Lean commands and the declaration keywords a chunk is classified by
_TOP = re.compile(r"^(?:@\[|/-|--|set_option\b|private\b|protected\b|noncomputable\b|nonrec\b|theorem\b|lemma\b|"
                  r"example\b|def\b|abbrev\b|instance\b|structure\b|inductive\b|class\b|namespace\b|section\b|"
                  r"end\b|open\b|variable\b|universe\b|attribute\b|notation\b|macro\b|import\b|#)")
_KIND = re.compile(r"(?<![\w.])(import|theorem|lemma|example|def|abbrev|instance|structure|inductive|class|"
                   r"namespace|section|end|open|variable|universe|attribute|notation|macro|#\w+)\b")


def _gh(*args: str, cwd: Path | None = None, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, cwd=str(cwd) if cwd else None)


def _lean_chunks(code: str) -> list:
    """Top-level commands, each with the comments, attributes and `set_option … in`
    that lead into it."""
    chunks: list = []
    for line in code.splitlines():
        if chunks and (not _TOP.match(line) or _chunk_kind(chunks[-1]) is None):
            chunks[-1] += "\n" + line
        else:
            chunks.append(line)
    return chunks


def _chunk_kind(chunk: str) -> str | None:
    text = re.sub(r"(?:set_option\s+\S+\s+\S+|open\b[^\n]*?)\s+in\b", " ", _uncommented(chunk))
    m = _KIND.search(text)
    return m.group(1) if m else None


def palomar_split(code: str, main: str, namespace: str) -> tuple[str, str, str]:
    """(Challenge.lean, Solution.lean, qualified main theorem) from one checked file.

    Solution is the file as Lean checked it, wrapped in `namespace`. Challenge
    keeps the definitions (made public) and states only `main` with `sorry`: the surface
    a Palomar reader audits, which Comparator matches against Solution.
    """
    header, challenge, solution = [], [], []
    for chunk in _lean_chunks(code.rstrip()):
        kind = _chunk_kind(chunk)
        # a private name is mangled with its module, so Challenge's and Solution's
        # copies of one definition would differ and Comparator would reject the pair
        chunk = re.sub(r"(?m)^((?:@\[[^\]]*\]\s*)*)private\s+", r"\1", chunk).rstrip()
        if kind is None or kind.startswith("#"):
            continue  # the axiom probe and stray #eval/#check are for the pipeline, not the reader
        if kind == "import":
            header.append(chunk)
            continue
        solution.append(chunk)
        if kind in ("theorem", "lemma", "example"):
            m = _DECL.match(next((ln for ln in _uncommented(chunk).splitlines() if _DECL.match(ln)), ""))
            if m and m.group(1).split(".")[-1] == main:
                decl = re.search(r"(?<![\w.])(?:theorem|lemma)\s+" + re.escape(m.group(1)), chunk).end()
                challenge.append(chunk[:chunk.index(":=", decl)].rstrip() + " := by\n  sorry")
        else:
            challenge.append(chunk)

    def wrap(body: list) -> str:
        return "\n".join(header) + f"\n\nnamespace {namespace}\n\n" + "\n\n".join(body) + f"\n\nend {namespace}\n"

    names = [n for n in _theorems(wrap(solution))[0] if n.split(".")[-1] == main]
    if len(names) != 1 or sum(bool(re.search(r"\bsorry\b", c)) for c in challenge) != 1:
        raise ValueError(f"could not isolate the main theorem `{main}` in the Lean file")
    return wrap(challenge), wrap(solution), names[0]


def _main_theorem(r: dict) -> str:
    if r["status"] in REFUTED:
        return "refutation"
    names = [n.split(".")[-1] for n in _theorems((r.get("lean") or {}).get("code", ""))[0]]
    return "main_theorem" if "main_theorem" in names else (names[-1] if names else "")


def _yaml(value, indent: int = 0) -> str:
    """Block YAML with JSON-quoted scalars: always valid, no dependency."""
    pad = "  " * indent
    if isinstance(value, dict):
        return "".join(f"{pad}{k}:" + (f"\n{_yaml(v, indent + 1)}" if isinstance(v, (dict, list)) and v
                                        else f" {_yaml(v)}\n") for k, v in value.items())
    if isinstance(value, list):
        if not value:
            return "[]"
        return "".join(f"{pad}-" + (f"\n{_yaml(v, indent + 1)}" if isinstance(v, (dict, list)) and v
                                     else f" {_yaml(v)}\n") for v in value)
    return json.dumps(value, ensure_ascii=False)


def formalization_yaml(r: dict, models: dict | None, maintainer: str, namespace: str, main: str,
                       classification: dict) -> str:
    refuted = r["status"] in REFUTED
    faith = (r.get("lean") or {}).get("faithfulness") or {}
    used = [m for m in models_used(r, models).split(", ") if m != "not recorded"]
    claim = " ".join(str(r.get("statement", "")).split())
    return ("# yaml-language-server: $schema=https://raw.githubusercontent.com/mathlib-initiative/"
            "formalization.yaml/main/schema/formalization.schema.json\n") + _yaml({
        "version": "v0.4",
        "project": {
            "name": headline(r),
            "description": (f"A counterexample to the claim: {claim}" if refuted else claim),
            "authors": [f"{maintainer} (operator of the mathforge pipeline)"],
            "license": "Apache-2.0",
            "responsible_maintainers": [maintainer],
        },
        "classification": classification,
        "sources": [{
            "title": ("Refutation of a conjecture proposed by a language model in an automated mathforge run"
                      if refuted else "Theorem conjectured and proved in an automated mathforge run"),
            "type": "original-proof",
            "relationship": "other",
            "note": "Novelty was screened only by an automated search of arXiv, Crossref and OpenAlex; "
                    "the result may be a rediscovery.",
        }],
        "related_formalizations": [],
        "status": {
            "scope": (f"Formalizes the negation of the claim above as `{main}`." if refuted
                      else f"Formalizes the statement above as `{main}`.")
                     + " The Lean statement was back-translated by a model and judged "
                     + f"{faith.get('verdict', 'n/a')}; it was not audited by a person.",
            "sorry_count": 0,
            "sorry_in_definitions": 0,
            "axioms": [],
        },
        "automation": {
            "methods": [{
                "method": "autonomous",
                "models": used,
                "framework": "mathforge",
                "tool_setup": "Models proposed the claim, wrote Python search and re-check scripts that were "
                              "executed, and wrote the Lean file, repaired against Lean's own errors; "
                              "the axiom report was read with #print axioms.",
                "cost": {"wall_time": "not tracked", "spend_usd": "not tracked",
                         "hardware": "local machine (Python, Lean) and model APIs"},
                "prompting_notes": "n/a",
            }],
            "spend_usd": "not tracked",
            "notes": "Fully automated, no human in the loop; the operator only chose to publish.",
        },
        "fidelity": {"divergences": "; ".join(map(str, faith.get("differences") or [])) or "none known"},
        "review": {
            "status": "unreviewed",
            "reviewers": ["none"],
            "notes": "Automated checks only: executed scripts, Lean 4 + Mathlib, and a model's "
                     "back-translation of the Lean statement.",
        },
        "alignment": {
            "namespace": namespace,
            "statements": [{
                "source": claim,
                "lean": f"{namespace}.{main}" if "." not in main else main,
                "module": "Solution",
                "status": "proved",
                "note": "the negation of the claim" if refuted else "the claim as stated",
            }],
        },
        "acknowledgements": "Lean 4 and Mathlib.",
    })


def _lakefile(package: str, lean_project: Path) -> str:
    rev = re.search(r'^rev\s*=\s*"([^"]+)"', (lean_project / "lakefile.toml").read_text(encoding="utf-8"), re.M)
    return (f'name = "{package}"\nversion = "0.1.0"\ndefaultTargets = ["Challenge", "Solution"]\n\n'
            f'[[require]]\nname = "mathlib"\nscope = "leanprover-community"\nrev = "{rev.group(1) if rev else "master"}"\n\n'
            '[[lean_lib]]\nname = "Challenge"\nroots = ["Challenge"]\n\n'
            '[[lean_lib]]\nname = "Solution"\nroots = ["Solution"]\n')


def _namespace(folder_name: str) -> str:
    """A valid Lean namespace from a result folder: `3-term-...` and `A240513-...` both broke the
    lowercase-only version (a leading digit, a dropped capital)."""
    name = "".join(w.capitalize() for w in re.findall(r"[A-Za-z0-9]+", folder_name))
    return "Mathforge." + ("R" + name if name[:1].isdigit() else name)


def palomar_bundle(forge: Forge, r: dict, folder: Path, classification: dict, maintainer: str) -> str:
    """Write a Palomar-ready Lake project into `folder`; returns "" or why not.

    Challenge and Solution are elaborated here first: a bundle that does not
    compile is worse than none.
    """
    project, lean = forge.lean_project, r.get("lean") or {}
    main = _main_theorem(r)
    if not (project and lean.get("code") and main):
        return "no Lean project or Lean file"
    namespace = _namespace(folder.name)
    try:
        challenge, solution, qualified = palomar_split(lean["code"], main, namespace)
    except ValueError as exc:
        return str(exc)
    scratch = forge.run.path / f"{r['id']}_palomar"
    scratch.mkdir(exist_ok=True)
    rc, out = run_lean(challenge, project, scratch, f"{r['id']}_Challenge")
    if rc != 0:
        return f"Challenge.lean does not elaborate: {_verdict_line(out)}"
    rc, out = _run_lean_probed(solution, project, scratch, f"{r['id']}_Solution")
    if not lean_verdict(solution, rc, out).get("sorry_free"):
        return f"Solution.lean is not sorry-free after wrapping: {_verdict_line(out)}"
    package = re.sub(r"[^A-Za-z0-9]", "", namespace.split(".")[-1])
    files = {
        "Challenge.lean": challenge,
        "Solution.lean": solution,
        "comparator.json": json.dumps({"challenge_module": "Challenge", "solution_module": "Solution",
                                       "theorem_names": [qualified], "definition_names": [],
                                       "permitted_axioms": sorted(STANDARD_AXIOMS)}, indent=2) + "\n",
        "formalization.yaml": formalization_yaml(r, forge.run.data.get("models"), maintainer, namespace,
                                                 qualified, classification),
        "lakefile.toml": _lakefile(package, project),
        "lean-toolchain": (project / "lean-toolchain").read_text(encoding="utf-8"),
    }
    manifest = project / "lake-manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["name"] = package
        files["lake-manifest.json"] = json.dumps(data, indent=1) + "\n"
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    return ""


def _results_checkout() -> tuple[str, Path]:
    """(owner/name, local checkout) of the public results repository, created on first use."""
    repo = RESULTS_REPO
    if "/" not in repo:
        login = _gh("gh", "api", "user", "-q", ".login").stdout.strip()
        if not login:
            raise RuntimeError("gh is not logged in (gh auth login)")
        repo = f"{login}/{repo}"
    if not (RESULTS_CHECKOUT / ".git").exists():
        if _gh("gh", "repo", "view", repo).returncode != 0:
            made = _gh("gh", "repo", "create", repo, "--public", "--description",
                       "Machine-checked results from mathforge, an automated conjecture-and-proof pipeline")
            if made.returncode != 0:
                raise RuntimeError(f"gh repo create {repo}: {(made.stdout + made.stderr).strip()[-300:]}")
        cloned = _gh("gh", "repo", "clone", repo, str(RESULTS_CHECKOUT))
        if cloned.returncode != 0:
            raise RuntimeError(f"gh repo clone {repo}: {(cloned.stdout + cloned.stderr).strip()[-300:]}")
        _gh("git", "checkout", "-B", "main", cwd=RESULTS_CHECKOUT)
    _gh("git", "pull", "--ff-only", "origin", "main", cwd=RESULTS_CHECKOUT)  # fails harmlessly on an empty repo
    return repo, RESULTS_CHECKOUT


def results_index(checkout: Path, repo: str) -> str:
    """The repository's front page: every published result, newest first."""
    rows = []
    for meta in checkout.glob("*/result.json"):
        rows.append(json.loads(meta.read_text(encoding="utf-8")))
    rows.sort(key=lambda m: (m.get("date", ""), m.get("folder", "")), reverse=True)
    lines = [
        "# mathforge results", "",
        f"Results from [mathforge]({MATHFORGE_URL}), a fully automated "
        "pipeline: language models propose conjectures, search for counterexamples, prove, and "
        "formalize in Lean 4 + Mathlib. Only results Lean checked sorry-free are published here. "
        "No person reviewed them; corrections are welcome as issues.", "",
        "Each folder holds the write-up (`README.md`), the scripts that were run, and, where the "
        "split succeeded, a Lake project ready for the [Palomar registry](https://palomar-registry.org/) "
        "(`Challenge.lean`, `Solution.lean`, `comparator.json`, `formalization.yaml`).", "",
        f"{len(rows)} result(s).", "",
        "| Date | Verdict | Claim | Models | Palomar |",
        "| --- | --- | --- | --- | --- |",
    ]
    for m in rows:
        claim = m.get("headline", "").split(": ", 1)[-1].replace("|", "\\|")
        lines.append(f"| {m.get('date', '')} | {'false' if m.get('status') in REFUTED else 'true'} | "
                     f"[{claim}]({m['folder']}/) | {m.get('models', '')} | {'bundle' if m.get('palomar') else '—'} |")
    return "\n".join(lines) + "\n"


def publish_result(forge: Forge, seed: str, r: dict) -> dict:
    """Publish one result as a folder of the public results repository. Never
    raises: publishing is a side effect of research, and a failed post must not
    lose the result."""
    if not shutil.which("gh") or not shutil.which("git"):
        return {"error": "gh and git must be on PATH to publish"}
    run, cid = forge.run, r["id"]
    try:
        with _REPO_LOCK:
            repo, checkout = _results_checkout()
            folder = checkout / f"{run.path.name[:48]}-{cid}"
            folder.mkdir(exist_ok=True)
            models = run.data.get("models")
            # the generated artifacts are the point: a reader can re-run them
            extras = ([f"{cid}_refute.lean", f"{cid}_falsify.py", f"{cid}_confirm.py"] if r["status"] in REFUTED
                      else [f"{cid}_lean.lean", f"{cid}_falsify.py", f"{cid}_check.py"])
            for name in extras:
                if (run.path / name).exists():
                    shutil.copy2(run.path / name, folder / name)
            maintainer = _gh("git", "config", "user.name", cwd=checkout).stdout.strip() or "mathforge operator"
            classification = run.stage(f"{cid}.classify", lambda: forge.classify(r))
            why_not = palomar_bundle(forge, r, folder, classification, maintainer)
            body = _publication(seed, r, models)
            body += ("\n## Palomar\n\n" + (
                f"This folder is a Lake project for the [Palomar registry](https://palomar-registry.org/). "
                f"To submit it, use the [form]({PALOMAR_FORM}) with this repository, the commit, and "
                f"`{folder.name}` as the project path. Read `Challenge.lean` against the statement first: "
                "Palomar asks for human review, and none has been done."
                if not why_not else f"No Palomar bundle: {why_not}.") + "\n")
            (folder / "README.md").write_text(body, encoding="utf-8")
            (folder / "result.json").write_text(json.dumps({
                "folder": folder.name, "status": r["status"], "headline": headline(r),
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "models": models_used(r, models),
                "palomar": not why_not, "run": run.path.name, "id": cid}, indent=2) + "\n", encoding="utf-8")
            (checkout / "README.md").write_text(results_index(checkout, repo), encoding="utf-8")
            # Palomar requires Apache-2.0, detected in the project directory; the
            # code repo's own CC0 LICENSE is not it
            for target in (checkout, folder):
                shutil.copy2(HERE / "LICENSE-results", target / "LICENSE")
            _gh("git", "add", "-A", cwd=checkout)
            _gh("git", "commit", "-m", f"{headline(r)} ({r['status']})", cwd=checkout)
            pushed = _gh("git", "push", "-u", "origin", "main", cwd=checkout)
            if pushed.returncode != 0:
                return {"error": f"git push: {(pushed.stdout + pushed.stderr).strip()[-300:]}"}
    except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if why_not:
        log(f"no Palomar bundle: {why_not}", cid)
    return {"url": f"https://github.com/{repo}/tree/main/{folder.name}", "status": r["status"],
            "title": r.get("title", cid), "palomar": not why_not}


def publish(forge: Forge, seed: str, results: list) -> list:
    """Publish every qualifying result once. Cached in state.json, so a --resume
    of a published run re-reads the URL instead of publishing it again."""
    run, published = forge.run, []
    for r in results:
        key = f"{r['id']}.published"
        if not publishable(r):
            for old in (key, f"{r['id']}.gist"):
                stale = (run.data.get(old) or {}).get("url")
                if stale:
                    log(f"PUBLIC {stale} no longer qualifies (now `{r['status']}`); retract it", r["id"])
            continue
        # a failed post is not a result: drop it so --resume retries instead of
        # caching the error forever
        if not (run.data.get(key) or {}).get("url"):
            run.data.pop(key, None)
        info = run.stage(key, lambda r=r: publish_result(forge, seed, r))
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
        return f"publish: {len(published)}/{qualifying} published, rest failed; --resume retries them"
    return f"publish: {len(published)} result(s) published"


# Submission packages: preprint sites take a PDF, and ProofForum also wants LaTeX
# source whose theorem-like environments it can extract. Nothing here uploads.
SUBMISSIONS = "_submissions"
EXPORT_ABORTED = 2  # export_latex could not refresh the folder at all, as against 1: some package is not `ok`
EXPORT_LOCK_STALE = 1800  # seconds after which another export's lock is taken for a crash
REFUTATIONS = "_refutations"  # run directories never start with an underscore, so this cannot collide
_GENERATED = "Text generated by language models in the mathforge pipeline; not reviewed by a person."
_KINDS = "Theorem|Lemma|Proposition|Corollary|Claim"
_HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_LEAD = re.compile(r"^(\*{1,2})([^*]+?)\1[.:]?\s*(.*)$")
# `Lemma 1. text` / `Proof. text` with no markup; same groups as _LEAD
_PLAIN_LEAD = re.compile(rf"^()((?:{_KINDS}) \d+|Proof(?:, verbatim[^:.]*)?)[.:](?:\s+(.*))?$")
# kind, label, parenthesised note, title. A label is a number, `c1`, or one capital: `Theorem list` is a heading
_STATEMENT = re.compile(rf"^(?:\d+(?:\.\d+)*\.?\s+)?({_KINDS})(?:\s+(\d+(?:\.\d+)*[a-z]?|[A-Za-z]\d+|[A-Z]))?'?"
                        r"\s*(?:\((.*)\))?\s*(?:(?:[:.]|\s[—–-])\s*(.*?))?\s*[.:]?$")
# the sections after the results, where a `Theorem 1` heading reports on a theorem instead of stating it
_AFTER_RESULTS = re.compile(r"(?i)^(?:\d+(?:\.\d+)*\.?\s+.*\b(?:verification|limitations|novelty)\b.*|novelty|models)$")
_PROOF = re.compile(r"^Proof(?:\s+(of\s+.*?|sketch|idea)|,?\s*\(?\s*verbatim.*?\)?|)\s*[.:]?$")
_STATEMENT_END = re.compile(r"(?i)^\W*(?:[\w ]{0,30}\bproof\b[^.]*\bverbatim\b|(?:for|in) this proof\b)")
_QED = re.compile(r"\s*(?:[∎□■]|\$\\(?:blacksquare|square|Box|qed)\$|\\\(\\(?:blacksquare|square|Box|qed)\\\)|\\qed)\s*$")
# stretches Markdown must not touch: code, display and inline math, LaTeX environments
_MATH = re.compile(r"```.*?```|\$\$.*?\$\$|\$(?:[^$\n]|\n(?!\n))+\$|\\\(.*?\\\)|\\\[.*?\\\]|`[^`\n]*`"
                   r"|\\begin\{(\w+\*?)\}.*?\\end\{\1\}", re.S)
_UNICODE_TEX = {
    "≥": r"\geq", "≤": r"\leq", "≠": r"\neq", "−": "-", "×": r"\times", "·": r"\cdot", "→": r"\to",
    "≡": r"\equiv", "∈": r"\in", "∉": r"\notin", "⊆": r"\subseteq", "∪": r"\cup", "∩": r"\cap",
    "∅": r"\emptyset", "∖": r"\setminus", "∞": r"\infty", "∑": r"\sum", "∏": r"\prod", "∎": r"\blacksquare", "∣": r"\mid",
    "∤": r"\nmid", "⌊": r"\lfloor", "⌋": r"\rfloor", "⌈": r"\lceil", "⌉": r"\rceil", "ℤ": r"\mathbb{Z}",
    "ℕ": r"\mathbb{N}", "𝔽": r"\mathbb{F}", "²": "^2", "³": "^3", "⁻": "^{-}", "ⱼ": "_j", "ₘ": "_m", "ₙ": "_n",
    "ᵢ": "_i", "₊": "_{+}", "ᵀ": "^T", "ᵃ": "^a", "ᵈ": "^d", "ᵉ": "^e", "′": "'","ℓ": r"\ell",
    "↦": r"\mapsto", "⋯": r"\cdots",
    **{chr(0x2080 + d): f"_{d}" for d in range(10)},
    **dict(zip("αβγδεηλμνπρστφχψω", ("\\" + name for name in "alpha beta gamma delta epsilon eta lambda mu nu pi rho "
                                     "sigma tau phi chi psi omega".split()))),
}
# Latin Modern Math has no glyph for unicode-math's \setminus; \notni is a macro a paper used undefined.
# unicode-math redefines ∑, ∏ and ′ at \begin{document} as math-only, so one in running text stopped
# the build with "Missing $ inserted". Their text form is made after that, wrapping the meaning
# unicode-math gave the character (its \sum expands to the character, so naming \sum would loop).
_LATEX_PREAMBLE = ("\\usepackage{amsthm}\n\\usepackage{newunicodechar}\n\\providecommand{\\notni}{\\nni}\n"
                   "\\AtBeginDocument{\\renewcommand{\\setminus}{\\mathbin{\\backslash}}}\n\\emergencystretch=3em\n"
                   # a witness is printed as code, and a name like conclusion_A_equals_B has no other place to break
                   "\\let\\mfunderscore\\_\n\\renewcommand{\\_}{\\mfunderscore\\allowbreak}\n") + "".join(
    f"\\newtheorem{{{k}}}{{{k.capitalize()}}}\n\\newtheorem*{{{k}*}}{{{k.capitalize()}}}\n"
    for k in _KINDS.lower().split("|")
) + "".join(f"\\newunicodechar{{{ch}}}{{\\ensuremath{{{tex}}}}}\n" for ch, tex in _UNICODE_TEX.items()) + "".join(
    f"\\AtBeginDocument{{\\let\\mf{name}={ch}\\newunicodechar{{{ch}}}{{\\ensuremath{{\\mf{name}}}}}}}\n"
    for ch, name in (("∑", "sum"), ("∏", "prod"), ("′", "prime")))


def _outside_math(text: str, fn) -> str:
    """Apply `fn` to the stretches of `text` that are neither math nor code."""
    out, at = [], 0
    for m in _MATH.finditer(text):
        out += [fn(text[at:m.start()]), m.group()]
        at = m.end()
    return "".join(out + [fn(text[at:])])


def _stars(text: str) -> str:
    """Escape `*` written as multiplication (`2*s`, `I(A)*I(B)`), which Markdown would pair up as emphasis."""
    return _outside_math(text, lambda s: re.sub(r"(?<=[\w)\]])\*(?=[\w(\[])", lambda _m: "\\*", s))


def _plain(text: str) -> str:
    """A plain-text statement as Markdown: besides `*`, a backslash that is spaced or stands before a
    one-letter set name (`A\\B`) is set difference, not an escape or a command, `F_[n](x,y)` is not a
    link, and `_` is a subscript, never emphasis."""
    def prose(s: str) -> str:
        s = re.sub(r"(?<= )\\(?= )|(?<=[\w)\]])\\(?=[A-Z](?![A-Za-z]))", "∖", s)
        return re.sub(r"(?<!\\)_", r"\\_", s).replace("](", "]&#40;").replace("<", "\\<")
    return _outside_math(_stars(text), prose)


def _tex_note(text: str) -> str:
    """A heading fragment as LaTeX: Markdown marks dropped and specials escaped, outside math."""
    def prose(s: str) -> str:
        s = re.sub(r"(?<!\\)[*`]", "", s).replace("\\*", "*")
        return re.sub(r"([&%#_])", r"\\\1", s)
    return _outside_math(text.replace("`", ""), prose)


def paper_meta(markdown: str) -> tuple[str, str, str]:
    """(title, abstract, body) of a generated paper, the body's own sections raised to top level."""
    lines = markdown.strip().splitlines()
    title, abstract, body, i, fenced = "", [], [], 0, False
    while i < len(lines):
        fenced ^= lines[i].lstrip().startswith("```")
        head = None if fenced else _HEAD.match(lines[i])
        text = re.sub(r"^\d+\.\s+", "", head.group(2)) if head else ""
        if head and text == "Title":  # the title is the line under a `Title` heading, sometimes a heading itself
            i = next((j for j in range(i + 1, len(lines)) if lines[j].strip()), i)
            title = re.sub(r"[*`]", "", lines[i].strip().lstrip("#").strip())
        elif head and not title:
            title = re.sub(r"[*`]", "", text)
        elif head and text == "Abstract":
            i += 1
            while i < len(lines) and not _HEAD.match(lines[i]):
                abstract.append(lines[i])
                i += 1
            continue
        else:
            body.append(lines[i])
        i += 1
    # the paper's numbered sections set the depth: an embedded proof brings headings of its own, at any level
    heads, fenced = [], False
    for n, line in enumerate(body):
        fenced ^= line.lstrip().startswith("```")
        head = None if fenced else _HEAD.match(line)
        if head and not (_STATEMENT.match(head.group(2)) or _PROOF.match(head.group(2))):
            heads.append((n, len(head.group(1)), head.group(2)))
    numbered = [level for _n, level, text in heads if re.match(r"\d+\.\s", text)]
    shift = min(numbered or [level for _n, level, _text in heads] or [1]) - 1
    for n, level, text in heads:
        level = max(level - shift, 1)
        if numbered and not re.match(r"\d+(?:\.\d+)*\.?\s|Models$", text):
            level = max(level, 2)  # an embedded proof's own heading is not one of the paper's sections
        body[n] = "#" * level + " " + text
    return title, "\n".join(abstract).strip(), "\n".join(body).strip() + "\n"


def theorem_envs(body: str) -> str:
    """Rewrite a Markdown paper's theorem, lemma and proof headings (or paragraph leads) as LaTeX
    environments, passed to pandoc as raw blocks so the text between them stays Markdown."""
    # ponytail: a statement runs to the next heading, lead or proof announcement, and a proof to its QED
    # mark or the next heading or lead; prose after an unmarked one lands inside it, and a theorem restated
    # by its embedded proof appears twice. Mark statement and proof ends in the paper prompt if that bites
    out, env, fenced, blank, owner, past_results = [], None, False, True, "", False

    def raw(tex: str) -> list:
        return ["", "```{=latex}", tex, "```"]

    def close(interrupted: bool = False):
        nonlocal env
        if env:
            name, text = env["name"], env["body"]
            empty = not "".join(text).strip()
            if name != "proof":
                label, note = env["label"], env["note"]
                first = next((ln.lstrip() for ln in text if ln.strip()), "")
                # the heading carried the statement itself: nothing under it, or only a list
                if note and (empty or re.match(r"(?:[-*+]|\d+[.)])\s", first)):
                    text, note = [note + ("" if note[-1] in ".:;!?" else "."), *text], ""
                note = ": ".join(filter(None, [label, note]))
                out.extend([*raw(env["start"] + (f"[{{{_tex_note(note)}}}]" if note else "")), *text, *raw(f"\\end{{{name}}}"), ""])
            elif interrupted and not empty:  # it resumes after the lemmas, so no QED box here
                out.extend(["", "*Proof.*", "", *text, ""])
            elif not empty:  # an empty one was a lead-in with no proof under it
                out.extend([*raw(env["start"]), *text, *raw("\\end{proof}"), ""])
        env = None

    def feed(text: str, starts_paragraph: bool):
        if env["name"] == "proof":
            stripped = _QED.sub("", text)
            env["body"].append(stripped)
            if stripped != text:
                close()
        elif starts_paragraph and _STATEMENT_END.search(text):
            close()
            out.append(text)
        else:
            env["body"].append(text)

    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        starts = blank and not fenced
        head = None if fenced else _HEAD.match(line)
        lead = (_LEAD.match(line) or _PLAIN_LEAD.match(line)) if starts and not head else None
        label = head.group(2) if head else lead.group(2).strip() if lead else ""
        if head and owner and _AFTER_RESULTS.match(label):  # `owner`: a statement has been seen, so results came first
            past_results = True  # from here `Theorem 1` labels a search log or a query list
        kind, proof = (None, None) if past_results else (_STATEMENT.match(label), _PROOF.match(label))
        rest = (lead.group(3) or "") if lead else ""
        blank = not line.strip()
        if (kind or proof) and _HEAD.match(rest):  # `*Proof.* # Title`: the heading is a heading
            close()
            out.extend(["", rest])
        elif kind:
            name, number = kind.group(1).lower(), kind.group(2) or ""
            close(interrupted=bool(env) and env["name"] == "proof" and env["of"] in ("theorem", "proposition", "corollary")
                  and name in ("lemma", "claim"))
            owner = name
            numbered = number.isdigit()
            env = {"name": name if numbered else name + "*", "label": "" if numbered else number,
                   "note": ": ".join(filter(None, [kind.group(3), kind.group(4)])), "body": [],
                   "start": f"\\setcounter{{{name}}}{{{int(number) - 1}}}\\begin{{{name}}}" if numbered else f"\\begin{{{name}*}}"}
            if rest:
                feed(rest, False)
        elif proof:
            close()
            of = proof.group(1) or ""
            target = re.search(rf"(?i)\b({_KINDS})\b", of)
            # doubled braces: amsthm puts the title inside \item[...], where a bare `]` ends it early
            env = {"name": "proof", "body": [], "of": target.group(1).lower() if target else owner,
                   "start": "\\begin{proof}" + (f"[{{{{Proof {_tex_note(of)}}}}}]" if of else "")}
            if rest:
                feed(rest, False)
        elif head:
            close()
            out.append(line)
        elif env:
            feed(line, starts)
        else:
            out.append(line)
    close()
    return "\n".join(out) + "\n"


def pandoc_input(markdown: str, author: str) -> str:
    """What pandoc is given for a paper: a metadata block, then the body with its environments."""
    # a row break written with three or more backslashes stops pandoc reading the environment as LaTeX
    # and pandoc drops a `<br>` in a table cell together with the space it stood for
    title, abstract, body = paper_meta(re.sub(r"<br\s*/?>", " ", re.sub(r"(?m)\\{3,}$", lambda _m: "\\\\", markdown)))
    meta = json.dumps({"title": _stars(title), "author": [author], "date": _GENERATED, "abstract": _stars(abstract)},
                      ensure_ascii=False)
    return f"---\n{meta}\n---\n\n{theorem_envs(_stars(body))}"


def latex_document(markdown: str, author: str) -> str:
    """A standalone LaTeX file from a Markdown paper, through pandoc."""
    if not shutil.which("pandoc"):
        raise RuntimeError("pandoc is not installed")
    # `^` and `~` stay literal: statements write exponents as x^{k}, which Markdown superscripts would swallow
    done = subprocess.run(["pandoc", "-f", "markdown+tex_math_single_backslash-superscript-subscript", "-t", "latex",
                           "-s", "-V", "geometry=margin=1in"], input=pandoc_input(markdown, author),
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    if done.returncode:
        raise RuntimeError(f"pandoc: {done.stderr.strip()[-400:]}")
    return done.stdout.replace("\\begin{document}", _LATEX_PREAMBLE + "\\begin{document}", 1)


def refutations_note(entries: list) -> str:
    """One Markdown paper for every Lean-checked refutation: `entries` is a list of
    (seed, result, models, public url). Each is a proposition whose proof is the witness."""
    n = len(entries)
    lines = [
        f"# {n} machine-checked refutation{'s' * (n != 1)} of machine-generated conjectures", "",
        "## Abstract", "",
        f"We record {n} false statement{'s' * (n != 1)}, each with an explicit counterexample. The statements "
        "were conjectured by a language model inside an automated pipeline (mathforge), not taken from the "
        "literature. For each, a search script found a witness, an independently written script re-checked it "
        "from the statement alone, and Lean 4 with Mathlib accepted a sorry-free proof of the negation, whose "
        "statement a separate model call judged faithful to the informal one. No person has reviewed the "
        "statements, the witnesses or the Lean files.", "",
        "## 1. Introduction", "",
        "A counterexample is a small true fact about the objects, and it is normally thrown away. The claims "
        "below are plausible-looking statements over infinite families that fail at a small instance. They are "
        "collected so that nobody has to rediscover that they fail. Each section gives the claim, the witness, "
        "how it was checked, and a link to the scripts and the Lean file.", "",
        "## 2. Refutations", "",
    ]
    for k, (seed, r, models, url) in enumerate(entries, 1):
        lean, witness = r.get("lean") or {}, _witness(r)
        lines += [f"### 2.{k} {_plain(str(r.get('headline') or r.get('title') or r['id']).strip().rstrip('.'))}", "",
                  f"Seed topic of the run: {_plain(str(seed))}", ""]
        if r.get("notation"):
            lines += [f"Notation. {_plain(r['notation'])}", ""]
        lines += [
            f"**Proposition {k}.** The following statement is false. {_plain(r.get('statement', ''))}", "",
            "**Proof.** Counterexample: " + (f"`{witness}`" if witness else "recorded in the scripts linked below")
            + ". An independent script, written by a separate model call that re-implemented the definitions "
            "from the statement, confirmed the witness. ∎", "",
            "Record of the checks:", "",
            *(map(_plain, _lean_lines(lean)) if lean else ["- Lean: not run."]),
            *([f"- Scripts, Lean file and Lake project: <{url}>"] if url else []),
            f"- Models: {models_used(r, models)}", "",
        ]
    return "\n".join(lines + [
        "## 3. Limitations", "",
        "Every statement, script and Lean file here was written by language models in an automated pipeline; no "
        "person has reviewed them. Lean certifies the Lean statement only. Whether that statement says what the "
        "informal one says was judged by a separate model call that back-translated it, which is a screening "
        "filter, not an oracle, so each Lean file should be read against its statement.", "",
        "The conjectures were posed by the pipeline itself, so their refutation carries no claim of significance. "
        "No literature search was run for these statements: any of them, or its failure, may already be known.", "",
    ])


def _latex_engine() -> str | None:
    return shutil.which("tectonic") or shutil.which("xelatex")


def build_pdf(folder: Path, stem: str) -> str:
    """Compile `<stem>.tex` in `folder`; returns a status starting `ok`, or what went wrong."""
    engine, pdf = _latex_engine(), folder / f"{stem}.pdf"
    if not engine:
        return "no LaTeX engine (install tectonic)"
    pdf.unlink(missing_ok=True)  # a failed build must not leave the previous PDF standing in
    built = subprocess.run([engine, *([] if Path(engine).stem.lower() == "tectonic" else ["-interaction=nonstopmode"]),
                            f"{stem}.tex"], cwd=folder, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=180)
    output = built.stdout + built.stderr
    if built.returncode or not pdf.exists():
        pdf.unlink(missing_ok=True)  # xelatex writes one even when it stops on an error
        return "PDF FAILED: " + " ".join(output.split())[-300:]
    missing = sorted(set(re.findall(r"Missing character: There is no \S+ \((U\+\w+)\)", output)))
    # a set: the engine reports each line once per pass
    wide = {float(w) for w in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", output) if float(w) > 20}
    return ("ok" + (f"; glyphs missing from the PDF: {' '.join(missing)}" if missing else "")
            + (f"; {len(wide)} line(s) run past the margin, worst by {max(wide):.0f}pt" if wide else ""))


def _byline() -> str:
    """The name on a paper: MATHFORGE_AUTHOR, else git's user.name."""
    try:
        return os.getenv("MATHFORGE_AUTHOR") or _gh("git", "config", "user.name").stdout.strip() or "mathforge operator"
    except (OSError, subprocess.SubprocessError):
        return "mathforge operator"


def export_latex(author: str, retry: bool = False) -> int:
    """Write a submission package under math_output/_submissions: `.md`, `.tex` and `.pdf` for the paper of
    every run whose kept results are all `verified` or `machine-verified`, and one note for the machine-refuted
    ones. A paper is compiled again only when its LaTeX changes, or, with `retry`, when its last build was not
    `ok`. Returns 1 if any package is not `ok`, EXPORT_ABORTED if the folder could not be refreshed."""
    out = OUTPUT_ROOT / SUBMISSIONS
    out.mkdir(parents=True, exist_ok=True)
    # a --forever session and a manual call share this folder; two exports at once would compile over each other
    lock = out / "export.lock"
    try:
        if time.time() - lock.stat().st_mtime > EXPORT_LOCK_STALE:
            lock.unlink()  # its owner died: an export takes seconds, a minute or two at most
    except OSError:
        pass
    try:
        lock.open("x").close()
    except OSError:
        vlog("export: another mathforge is refreshing the submission folder; skipped")
        return EXPORT_ABORTED
    try:
        return _export_latex(out, author, retry)
    finally:
        lock.unlink(missing_ok=True)


def _export_latex(out: Path, author: str, retry: bool) -> int:
    try:
        known = json.loads((out / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}
    if not (isinstance(known, dict) and all(isinstance(v, str) for v in known.values())):
        known = {}  # a damaged record costs one rebuild
    papers, refuted, held_back, statuses = [], [], [], {}
    for path in sorted(p.parent for p in OUTPUT_ROOT.glob("*/state.json") if not p.parent.name.startswith("_")):
        try:
            data = Run(path).data
        except (OSError, ValueError) as exc:
            log(f"export: skipped {path.name}: unreadable state.json ({exc})")
            if path.name in known:  # unreadable is not unqualified: its package stays
                statuses[path.name] = known[path.name]
            continue
        results = [r for r in data.get("results") or [] if isinstance(r, dict) and r.get("id")]
        models = data.get("models")
        refuted += [(data.get("seed", path.name), r, models, (data.get(f"{r['id']}.published") or {}).get("url"))
                    for r in results if r.get("status") == "machine-refuted"]
        kept = [r for r in results if r.get("status") in ("machine-verified", "verified", "provisional")]
        if not (path / "paper.md").exists() or all(r["status"] == "provisional" for r in kept):
            continue
        if any(r["status"] == "provisional" for r in kept):
            held_back.append(path.name)  # the paper typesets an unproved claim as a theorem with a proof
            continue
        papers.append((path.name, (path / "paper.md").read_text(encoding="utf-8"),
                       [(r.get("title", r["id"]), r["status"], models_used(r, models)) for r in kept]))
    if refuted:
        papers.append((REFUTATIONS, refutations_note(refuted),
                       [(r.get("title", r["id"]), r["status"], models_used(r, models)) for _s, r, models, _u in refuted]))
    index = [
        "# Submission packages", "",
        f"{len(papers)} paper(s). Nothing here has been uploaded or reviewed by a person.", "",
        "`verified` means a referee model accepted the proof and an independently written script re-derived the "
        "result on small cases; it is not a Lean proof. `machine-verified` and `machine-refuted` mean Lean 4 + "
        "Mathlib accepted a sorry-free proof of the statement or of its negation, and a model judged the Lean "
        "statement faithful to the informal one.", "",
    ]
    for stem, markdown, rows in papers:
        title, abstract, _body = paper_meta(markdown)
        tex_file, status = out / f"{stem}.tex", known.get(stem, "")
        try:
            tex = latex_document(markdown, author)
            if (not status or (retry and not status.startswith("ok"))
                    or not tex_file.exists() or tex_file.read_text(encoding="utf-8") != tex
                    or (status.startswith("ok") and not (out / f"{stem}.pdf").exists())
                    or (status.startswith("no LaTeX engine") and _latex_engine())):
                (out / f"{stem}.md").write_text(markdown, encoding="utf-8")
                tex_file.write_text(tex, encoding="utf-8")
                status = build_pdf(out, stem)
                log("export: " + _named(stem, _short(status, 100)))
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            if "pandoc is not installed" in str(exc):
                log("export: pandoc is not installed; no submission package written")
                return EXPORT_ABORTED
            status = f"FAILED: {type(exc).__name__}: {exc}"
            log("export: " + _named(stem, _short(status, 100)))
        statuses[stem] = status
        index += [f"## {title}", "", f"- Files: `{stem}.pdf`, `{stem}.tex`, `{stem}.md` ({status})",
                  *(f"- `{state}` {name} ({models})" for name, state, models in rows), "", abstract, ""]
    if held_back:
        index += ["## Not packaged", "", "These papers also state `provisional` results, which the referee or the "
                  "independent check did not accept, as theorems:", "", *(f"- {name}" for name in held_back), ""]
    for stem in set(known) - set(statuses):  # a run that no longer qualifies takes its package with it
        for suffix in (".md", ".tex", ".pdf"):
            (out / f"{stem}{suffix}").unlink(missing_ok=True)
    (out / "status.json").write_text(json.dumps(statuses, indent=1), encoding="utf-8")
    (out / "README.md").write_text("\n".join(index), encoding="utf-8")
    vlog(f"export: {len(papers)} paper(s) in {out}")
    return int(any(not s.startswith("ok") for s in statuses.values()))


# AiraXiv (airaxiv.com) takes AI-generated papers and gives agents an MCP endpoint, spoken here as
# plain JSON-RPC. Only --submit-airaxiv uploads: a submission is public once the site's moderation passes it.
AIRAXIV_MCP = os.getenv("AIRAXIV_MCP", "https://airaxiv.com/mcp/")
AIRAXIV_BATCH = 3  # per call; the site's terms forbid large-scale automated submission
AIRAXIV_MAX = 10  # per call, whatever was asked for; the site throttles well below this anyway
AIRAXIV_PAPER_TYPE = "ai_generated"
MATHFORGE_URL = "https://github.com/augusto-rehfeldt/mathforge"
_MODEL_LINE = re.compile(r"(?m)^- (?:Models|Proposed the claim|Counterexample search|Independent[^:\n]*|Wrote the proof|Referee"
                         r"|Lean formalization|Faithfulness judge|Revised against the review): (.+)$")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect would carry the API key to wherever it points; treat it as an error instead."""

    def redirect_request(self, *args, **kwargs):
        return None


def _airaxiv_http(url: str, method: str = "POST", body: bytes | None = None, headers: dict | None = None) -> tuple[dict, bytes]:
    """The AiraXiv client's one network call; the selftest replaces it."""
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    with urllib.request.build_opener(_NoRedirect).open(request, timeout=120) as response:
        return dict(response.headers), response.read()


def _airaxiv_busy(exc: Exception) -> bool:
    """The site is throttling (rate limit, daily quota): about the caller, not the paper, so the batch ends."""
    return getattr(exc, "code", None) == 429 or any(w in str(exc).lower() for w in ("too many requests", "quota"))


def _airaxiv_reason(exc: Exception) -> str:
    """An upload failure in one line; for an HTTP refusal, with the wait the site asks for and what it said."""
    reason = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, urllib.error.HTTPError):
        wait = exc.headers.get("Retry-After") if exc.headers else None
        try:
            said = " ".join(exc.read(400).decode("utf-8", "replace").split())
        except (OSError, ValueError):
            said = ""
        reason += (f"; retry after {wait}s" if wait else "") + (f"; {said}" if said else "")
    return reason


def airaxiv_tool(headers: dict, tool: str, arguments: dict) -> dict:
    """Call one AiraXiv MCP tool and return its JSON reply. RuntimeError means the site refused;
    ValueError means the reply could not be read, which says nothing about what the site did."""
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}
    body = json.loads(_airaxiv_http(AIRAXIV_MCP, body=json.dumps(message).encode(), headers=headers)[1])
    if isinstance(body, dict) and body.get("error"):
        raise RuntimeError(f"{tool}: {body['error'].get('message') if isinstance(body['error'], dict) else body['error']}")
    try:
        text = body["result"]["content"][0]["text"]
        refused = body["result"].get("isError")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError(f"{tool}: unreadable reply") from exc
    if refused:
        raise RuntimeError(f"{tool}: {text}")
    reply = json.loads(text)
    if not isinstance(reply, dict):
        raise ValueError(f"{tool}: unreadable reply")
    return reply


def _airaxiv_session(key: str) -> dict:
    """Open an MCP session; the headers every later call carries."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"}
    hello = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-03-26", "clientInfo": {"name": "mathforge", "version": "1.0"}, "capabilities": {}}}
    got, _body = _airaxiv_http(AIRAXIV_MCP, body=json.dumps(hello).encode(), headers=headers)
    session = next((value for name, value in got.items() if name.lower() == "mcp-session-id"), "")
    if session:
        headers["Mcp-Session-Id"] = session
    _airaxiv_http(AIRAXIV_MCP, body=json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}).encode(),
                  headers=headers)
    return headers


def _airaxiv_close(headers: dict) -> None:
    try:
        _airaxiv_http(AIRAXIV_MCP, method="DELETE", headers={k: v for k, v in headers.items() if k != "Content-Type"})
    except (OSError, ValueError):
        pass


def _airaxiv_upload(headers: dict, pdf_path: Path) -> tuple[str, str]:
    """Upload one PDF; (the one-time pdf_file_id that submit_paper or update_paper takes, the PDF's sha256)."""
    pdf = pdf_path.read_bytes()
    digest = hashlib.sha256(pdf).hexdigest()
    upload = airaxiv_tool(headers, "create_upload", {"filename": f"{pdf_path.stem.strip('_')}.pdf", "sha256": digest})
    target = urllib.parse.urlsplit(str(upload["upload_url"]))
    if target.scheme != "https":
        raise RuntimeError(f"upload address is not https: {str(upload['upload_url'])[:80]}")
    # the upload address may be another host's signed URL: the key goes only to AiraXiv itself
    own = target.username is None and target.hostname == urllib.parse.urlsplit(AIRAXIV_MCP).hostname
    _airaxiv_http(upload["upload_url"], method="PUT", body=pdf,
                  headers={"Content-Type": "application/pdf", **({"Authorization": headers["Authorization"]} if own else {})})
    return airaxiv_tool(headers, "complete_upload", {"upload_id": upload["upload_id"], "sha256": digest})["pdf_file_id"], digest


def _airaxiv_save(record_file: Path, record: dict) -> None:
    scratch = record_file.with_name("airaxiv.json.tmp")
    scratch.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(scratch, record_file)


def _airaxiv_authors(markdown: str) -> list:
    """The pipeline, naming every model in the paper's Models section, and the person who ran it."""
    models = sorted({m.strip() for line in _MODEL_LINE.findall(markdown) for m in line.split(",")} - {"not recorded"})
    return [{"name": "mathforge pipeline" + (f" ({', '.join(models)})" if models else ""), "type": "ai", "url": MATHFORGE_URL},
            {"name": _byline(), "type": "human"}]


def _airaxiv_submission(entry: dict) -> str:
    """The site's submission id in a record entry."""
    reply = entry.get("reply") if isinstance(entry.get("reply"), dict) else {}
    return str((reply["paper"] if isinstance(reply.get("paper"), dict) else reply).get("submission_id"))


def submit_airaxiv(limit: int = AIRAXIV_BATCH) -> int:
    """Upload packaged papers to AiraXiv's AI-generated track: each package whose build is `ok` with no
    missing glyphs, once, at most `limit` (never more than AIRAXIV_MAX) per call, the refutations note first.
    One upload runs at a time (`airaxiv.lock`). Returns 1 if anything failed or is unresolved."""
    return _airaxiv_locked(lambda out, key: _submit_airaxiv(out, key, max(0, min(limit, AIRAXIV_MAX))))


def revise_airaxiv(forge_for, limit: int = AIRAXIV_BATCH) -> int:
    """Answer AiraXiv's AI review, which arrives some time after a paper goes public: rewrite the run's paper
    against the report, rebuild its package and upload it as a new version. At most `limit` rewrites per call,
    each paper once. The rewrite is kept in state.json, so a refused upload is retried without it."""
    return _airaxiv_locked(lambda out, key: _revise_airaxiv(out, key, forge_for, max(0, min(limit, AIRAXIV_MAX))))


def _revise_airaxiv(out: Path, key: str, forge_for, limit: int) -> int:
    record_file = out / "airaxiv.json"
    try:
        record = json.loads(record_file.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        log(f"airaxiv: cannot read airaxiv.json ({exc}); nothing revised")
        return 1
    failed, reports = 0, []
    try:
        headers = _airaxiv_session(key)
    except (OSError, ValueError) as exc:
        log(f"airaxiv: could not open a session: {_airaxiv_reason(exc)[:300]}")
        return 1
    try:
        # the record holds submission ids; a paper has a public id, and a review, only once moderation passed it
        public, offset = {}, 0
        while True:
            page = airaxiv_tool(headers, "list_papers", {"scope": "user", "limit": 100, "offset": offset}).get("papers") or []
            public.update({str(p.get("submission_id")): str(p["paper_id"]) for p in page if isinstance(p, dict) and p.get("paper_id")})
            offset += len(page)
            if len(page) < 100:
                break
        for stem, entry in record.items():
            # ponytail: one revision per paper, so the review of a revised version goes unanswered;
            # record revisions per version if a second round is ever wanted
            if len(reports) >= limit or not isinstance(entry, dict) or entry.get("revision") or stem.startswith("_"):
                continue  # the refutations note has no run of its own to rewrite
            paper_id = public.get(_airaxiv_submission(entry))
            if not paper_id or not (OUTPUT_ROOT / stem / "state.json").exists():
                continue
            reviews = airaxiv_tool(headers, "get_paper_reviews", {"paper_id": paper_id}).get("reviews") or []
            report = "\n\n".join(str(r.get("content") or "") for r in reviews if isinstance(r, dict)).strip()
            if report:
                reports.append((stem, paper_id, report))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        failed += 1
        log(f"airaxiv: reading reviews stopped: {_short(_airaxiv_reason(exc), 300)}")
    finally:
        _airaxiv_close(headers)

    for stem, paper_id, report in reports:  # minutes each, so outside any session
        try:
            run = Run(OUTPUT_ROOT / stem)
            old = str(run.data["paper"])
            forge = forge_for(run)
            new = re.sub(r"\A```\w*\s*\n|\n```\s*\Z", "", forge.revise(old, report).strip())
            if not all(paper_meta(new)[:2]) or len(new) < len(old) // 2:
                raise ValueError("the rewrite lost the title, the abstract or half the paper")
            reviser = getattr(forge, "models", {}).get("writing") or "not recorded"
            run.data.update(paper=new, paper_v1=old, airaxiv_review=report, paper_revised_by=reviser)
            run.save()
            keepers = [r for r in run.data.get("results") or [] if r.get("status") in ("machine-verified", "verified", "provisional")]
            (run.path / "paper.md").write_text(new.rstrip() + "\n\n" + paper_models(keepers, run.data.get("models"), reviser), encoding="utf-8")
            record[stem]["revision"] = {"paper_id": paper_id, "state": "written"}
            _airaxiv_save(record_file, record)
            log("airaxiv: " + _named(stem, f"rewritten against the review of {paper_id}"))
        except Exception as exc:  # one paper's failure must not cost the others
            failed += 1
            log("airaxiv: " + _named(stem, f"NOT REVISED: {type(exc).__name__}: {_short(str(exc), 300)}"))

    written = [stem for stem, entry in record.items()
               if isinstance(entry, dict) and (entry.get("revision") or {}).get("state") == "written"]
    if not written:
        log("airaxiv: no review to answer")
        return int(bool(failed))
    if export_latex(_byline()) == EXPORT_ABORTED:
        log("airaxiv: no revision sent, the submission folder could not be refreshed")
        return 1
    try:
        statuses = json.loads((out / "status.json").read_text(encoding="utf-8"))
        headers = _airaxiv_session(key)
    except (OSError, ValueError) as exc:
        log(f"airaxiv: no revision sent: {_airaxiv_reason(exc)[:300]}")
        return 1
    try:
        for stem in written:
            status = str(statuses.get(stem, "no longer packaged"))
            try:
                if not status.startswith("ok") or "glyphs missing" in status:
                    raise RuntimeError(f"the revised package is not sendable: {status}")
                markdown = (out / f"{stem}.md").read_text(encoding="utf-8")
                title, abstract, _rest = paper_meta(markdown)
                file_id, digest = _airaxiv_upload(headers, out / f"{stem}.pdf")
                reply = airaxiv_tool(headers, "update_paper", {
                    "author_list": _airaxiv_authors(markdown),  # the reviser may be a model the first version never used
                    "paper_id": record[stem]["revision"]["paper_id"], "pdf_file_id": file_id, "title": title, "abstract": abstract,
                    "version_notes": "Revised in response to the AiraXiv AI review; the last section lists the changes."})
            except Exception as exc:
                failed += 1
                log("airaxiv: " + _named(stem, "REVISION NOT SENT: " + _short(_airaxiv_reason(exc), 300)))
                if _airaxiv_busy(exc):
                    log("airaxiv: rate limit or daily quota reached; written revisions are sent by a later call")
                    break
                continue
            record[stem]["revision"].update(state="sent", updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                                            sha256=digest, reply=reply)
            _airaxiv_save(record_file, record)
            log("airaxiv: " + _named(stem, f"revision sent for {record[stem]['revision']['paper_id']}"))
    finally:
        _airaxiv_close(headers)
    return int(bool(failed))


def _airaxiv_locked(work) -> int:
    """Run `work(submission folder, API key)` as the one AiraXiv job in flight (`airaxiv.lock`)."""
    key = os.getenv("AIRAXIV_API_KEY")
    if not key:
        log("airaxiv: set AIRAXIV_API_KEY (airaxiv.com > My API Keys) in the environment or in .env")
        return 1
    out = OUTPUT_ROOT / SUBMISSIONS
    lock = out / "airaxiv.lock"
    try:
        lock.open("x").close()
    except FileExistsError:
        log(f"airaxiv: another upload is running; if none is, one crashed: delete {lock}")
        return 1
    except OSError as exc:
        log(f"airaxiv: nothing packaged yet ({exc})")
        return 1
    try:
        return work(out, key)
    finally:
        lock.unlink(missing_ok=True)


def _submit_airaxiv(out: Path, key: str, limit: int) -> int:
    record_file = out / "airaxiv.json"
    try:
        statuses = json.loads((out / "status.json").read_text(encoding="utf-8"))
        record = json.loads(record_file.read_text(encoding="utf-8")) if record_file.exists() else {}
        if not isinstance(statuses, dict) or not isinstance(record, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        # an unreadable record must not read as "nothing sent yet": that would send everything again
        log(f"airaxiv: cannot read status.json or airaxiv.json ({exc}); nothing sent")
        return 1

    def save():
        _airaxiv_save(record_file, record)

    # written before submit_paper and replaced by the reply: one left behind means the reply never arrived
    unresolved = [stem for stem, entry in record.items() if isinstance(entry, dict) and entry.get("state") == "submitting"]
    for stem in unresolved:
        log(f"airaxiv: {stem[:50]} was sent but the reply was lost; check My Papers on the site, then replace or "
            f"delete its entry in {record_file.name}")
    def offered(stem: str) -> bool:
        """Not sent yet; a paper the site refused is offered again only once its PDF has changed."""
        entry, pdf = record.get(stem), out / f"{stem}.pdf"
        if not pdf.exists():
            return False
        if isinstance(entry, dict) and entry.get("state") == "refused":
            return entry.get("sha256") != hashlib.sha256(pdf.read_bytes()).hexdigest()
        return entry is None

    pending = [stem for stem in sorted(statuses, key=lambda stem: (stem != REFUTATIONS, stem))
               if str(statuses[stem]).startswith("ok") and "glyphs missing" not in str(statuses[stem])
               and offered(stem)][:limit]
    if not pending:
        log("airaxiv: nothing new to submit")
        return int(bool(unresolved))
    try:
        headers = _airaxiv_session(key)
    except (OSError, ValueError) as exc:
        log(f"airaxiv: could not open a session: {_airaxiv_reason(exc)[:300]}")
        return 1
    failed = len(unresolved)
    try:
        for stem in pending:
            try:
                markdown = (out / f"{stem}.md").read_text(encoding="utf-8")
                title, abstract, _rest = paper_meta(markdown)
                file_id, digest = _airaxiv_upload(headers, out / f"{stem}.pdf")
                stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                record[stem] = {"title": title, "submitted": stamp, "state": "submitting"}
                save()
                try:
                    reply = airaxiv_tool(headers, "submit_paper", {
                        "title": title, "abstract": abstract, "pdf_file_id": file_id,
                        "paper_type": AIRAXIV_PAPER_TYPE, "research_category": "theoretical",
                        "author_list": _airaxiv_authors(markdown)})
                except (RuntimeError, urllib.error.HTTPError) as refusal:
                    # the site said no, so nothing was accepted. A busy site may be asked again later; a paper it
                    # turned down is kept out of the queue until its PDF changes, or it would be refused every call
                    if isinstance(refusal, RuntimeError) and not _airaxiv_busy(refusal):
                        record[stem] = {"title": title, "refused": stamp, "state": "refused",
                                        "error": str(refusal)[:300], "sha256": digest}
                    else:
                        del record[stem]
                    save()
                    raise
            except Exception as exc:  # one paper's failure, whatever it is, must not skip the session clean-up
                failed += 1
                log("airaxiv: " + _named(stem, "FAILED: " + _short(_airaxiv_reason(exc), 300)))
                if _airaxiv_busy(exc):
                    log(f"airaxiv: rate limit or daily quota reached; {len(pending) - pending.index(stem)} paper(s) left for a later call")
                    break
                continue
            record[stem] = {"title": title, "submitted": stamp, "reply": reply}
            save()
            paper = reply.get("paper") if isinstance(reply.get("paper"), dict) else reply
            log("airaxiv: " + _named(stem, f"submitted as #{paper.get('submission_id', '?')}"))
    finally:
        _airaxiv_close(headers)
    return int(bool(failed))


def export_and_submit(limit: int | None, retry: bool = False) -> int:
    """Refresh the submission folder, then upload up to `limit` papers if a limit was given. The upload
    never reads a folder the export could not refresh: its statuses could be stale."""
    failed = export_latex(_byline(), retry=retry)
    if limit is None:
        return failed
    if failed == EXPORT_ABORTED:
        log("airaxiv: nothing sent, the submission folder could not be refreshed")
        return failed
    return submit_airaxiv(limit) or failed


def latest_run_dir(root: Path | None = None) -> Path | None:
    """The most recently touched run, for a bare `--resume`. Underscore-prefixed
    scratch directories (`_scout`, `_selftest`) are not runs."""
    root = OUTPUT_ROOT if root is None else root
    runs = [p.parent for p in root.glob("*/state.json") if not p.parent.name.startswith("_")]
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
    rule(_short(seed, 74))
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
            log(f"{bar}{r['status']:<16} {_short(c.get('title', ''), 60)}", c["id"])
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
        (run.path / "negative_results.md").write_text(negative_results(seed, results, run.data.get("models")), encoding="utf-8")
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
        # appended at write time, not cached: the paper model cannot misreport it,
        # and papers cached before this existed gain it on --resume
        (run.path / "paper.md").write_text(
            paper.rstrip() + "\n\n" + paper_models(keepers, run.data.get("models"), run.data.get("paper_revised_by")), encoding="utf-8")
        paper_path = run.path / "paper.md"

    published = publish(forge, seed, results) if getattr(args, "publish", False) else []

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
    if paper_path or any(r["status"] == "machine-refuted" for r in results):
        try:  # the submission folder is a by-product; it must not cost the run
            export_and_submit(getattr(args, "submit_airaxiv", None))
        except Exception as exc:
            log(f"export failed: {type(exc).__name__}: {exc}")
    if getattr(args, "revise_airaxiv", None) is not None:
        try:  # reviews arrive days after a submission, so every run looks
            revise_airaxiv(forge_for, args.revise_airaxiv)
        except Exception as exc:
            log(f"airaxiv revision failed: {type(exc).__name__}: {exc}")
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
        choices=EFFORTS,
        help="reasoning effort per call on the work model, overriding the menu's pick. An "
             "empty reply with finish_reason=length is the model thinking past its output "
             "allowance; lower this when that happens. `xhigh` only exists on some "
             "OpenAI-compatible providers and is rejected elsewhere (default: the menu's "
             f"pick, else {DEFAULT_EFFORT})",
    )
    ap.add_argument(
        "--review-effort",
        choices=EFFORTS,
        help="reasoning effort on the review model (proposing, refereeing, novelty, "
             f"independent check; default: the menu's pick, else {DEFAULT_REVIEW_EFFORT})",
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
        help="publish every machine-verified and machine-refuted result as a folder of the PUBLIC "
             "GitHub repository MATHFORGE_RESULTS_REPO (default mathforge-results; needs `gh` logged in)",
    )
    ap.add_argument(
        "--publish-existing",
        action="store_true",
        help="publish every qualifying result already under math_output (each once; see --publish) and exit",
    )
    ap.add_argument(
        "--export-latex",
        action="store_true",
        help="write a submission package (Markdown, LaTeX with theorem environments, PDF) for every paper "
             "whose kept results are all verified, plus one note for the machine-refuted ones, under math_output/_submissions, and exit. "
             "Needs pandoc; the PDF needs tectonic or xelatex. Uploads nothing; MATHFORGE_AUTHOR overrides the byline",
    )
    ap.add_argument(
        "--submit-airaxiv",
        nargs="?",
        type=int,
        const=AIRAXIV_BATCH,
        metavar="N",
        help="upload up to N (default %(const)s) packaged papers not yet sent to airaxiv.com's AI-generated track, where "
             "they become PUBLIC after the site's moderation. Alone it uploads and exits; with a seed, --resume or "
             "--forever it uploads after every run. Needs AIRAXIV_API_KEY (environment or .env)",
    )
    ap.add_argument(
        "--revise-airaxiv",
        nargs="?",
        type=int,
        const=AIRAXIV_BATCH,
        metavar="N",
        help="answer the AI review airaxiv.com posts on a public paper: rewrite up to N (default %(const)s) reviewed "
             "papers against their report and upload each as a new version, once per paper. Alone it revises and exits; "
             "with a seed, --resume or --forever it does so after every run. Needs AIRAXIV_API_KEY",
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

    researching = bool(args.seed or args.resume or args.forever or args.runs > 1 or args.publish_existing)
    revising = args.revise_airaxiv is not None and not researching  # needs the models, unlike an upload
    if args.export_latex or (args.submit_airaxiv is not None and not researching and not revising):
        load_local_env(HERE / ".env")  # AIRAXIV_API_KEY: this project's own .env, not the ai-suite checkout's
        return export_and_submit(args.submit_airaxiv, retry=args.export_latex)

    if not researching and not revising:
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
    load_local_env(HERE / ".env")  # this project's own .env (AIRAXIV_API_KEY), beside the ai-suite one
    interactive = False
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
    # The shared menu asked an effort per role, from the levels that role's model lists.
    args.effort, review_effort = resolve_efforts(args.effort, args.review_effort, interactive)
    effort_supported = set_reasoning_effort(ai, args.effort, review_effort)

    log(f"models      {args.model} (work) / {args.review_model} (review)")
    log(f"effort      {args.effort} (work) / {review_effort} (review)"
        f"{'' if effort_supported else ' -- unsupported by this provider'}, "
        f"{args.max_tokens} tokens/call requested")
    log(f"plan        {'forever' if args.forever else f'{args.runs} run(s)'}, "
        f"{args.conjectures} conjectures each, {args.workers} worker(s)")
    vlog(f"lean        {lean_project or ('disabled (--no-lean)' if args.no_lean else 'no Mathlib project found; run --setup-lean')}")
    vlog(f"novelty     {'arXiv + Crossref + OpenAlex + AiraXiv' + (' + Semantic Scholar' if os.getenv('S2_API_KEY') else '') if not args.no_search else 'disabled (--no-search)'}")
    vlog(f"publish     {'PUBLIC results repository ' + RESULTS_REPO if args.publish else 'off'}")
    vlog(f"library     {OUTPUT_ROOT}")

    def forge_for(run: Run) -> Forge:
        return Forge(ai, run, lean_project, search=not args.no_search)

    if revising:
        failed = revise_airaxiv(forge_for, args.revise_airaxiv)
        return (export_and_submit(args.submit_airaxiv) if args.submit_airaxiv is not None else 0) or failed

    if args.publish_existing:
        runs = sorted(p.parent for p in OUTPUT_ROOT.glob("*/state.json") if not p.parent.name.startswith("_"))
        count = 0
        for path in runs:
            run = Run(path)
            results = run.data.get("results") or []
            if any(publishable(r) for r in results):
                count += len(publish(forge_for(run), run.data.get("seed", path.name), results))
        log(f"publish: {count} result(s) published from {len(runs)} run(s)")
        return 0

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
    # A flag wins; an asked menu's per-model pick is next and its "default" is the provider's
    # own; unattended, a remembered pick or else the built-in defaults.
    picks = {"AI_WRITING_EFFORT": "max"}
    assert resolve_efforts(None, None, True, picks) == ("max", "provider-default")
    assert resolve_efforts("low", None, True, picks) == ("low", "provider-default")
    assert resolve_efforts(None, None, False, picks) == ("max", DEFAULT_REVIEW_EFFORT)
    assert resolve_efforts(None, "xhigh", False, {}) == (DEFAULT_EFFORT, "xhigh")

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
    real_arxiv_delay, real_airaxiv_delay = globals()["ARXIV_DELAY"], globals()["AIRAXIV_DELAY"]
    globals()["ARXIV_DELAY"] = globals()["AIRAXIV_DELAY"] = 0.0  # keep the selftest off the real request spacing
    _LIMITERS.clear()
    try:
        found = literature(["binomial sums", "binomial sums"], rows=1)  # repeated on purpose
    finally:
        globals()["_http_get"] = real_get
        globals()["ARXIV_DELAY"], globals()["AIRAXIV_DELAY"] = real_arxiv_delay, real_airaxiv_delay
        _LIMITERS.clear()
    # AiraXiv's search matches a phrase, so a query with no hit is asked again by its most distinctive word
    card = ('<li class="paper-item"> <span class="paper-card-id">2610.0001</span> <div class="paper-title">Sidon &amp; sums</div>'
            ' <div class="paper-card-abstract">An <b>abstract</b>.</div> <a href="/papers/view/2610.0001/">View</a> </li>')
    asked = []
    globals()["_http_get"] = lambda url, timeout=30: (asked.append(url), card if url.endswith("q=unitriangular") else "<ul></ul>")[1]
    try:
        aira = search_airaxiv("class counts in unitriangular groups")
        nothing = search_airaxiv("a b c")
    finally:
        globals()["_http_get"] = real_get
    assert aira == [{"source": "AiraXiv", "title": "Sidon & sums", "abstract": "An abstract.",
                     "url": "https://airaxiv.com/papers/view/2610.0001/"}], aira
    assert len(asked) == 3 and asked[0].endswith("q=class+counts+in+unitriangular+groups") and nothing == [], asked
    assert search_airaxiv in [backend for backend, _gap in _backends()]
    assert _short("abc", 5) == "abc" and _short("abcdef", 5) == "abcd…"
    # a package's log line: its name, cut if long, never padded out to a column
    assert _named("_refutations", "ok") == "_refutations: ok" and _named("a" * 60, "ok") == "a" * 43 + "…: ok"
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
    real_run_one, real_root, real_export = globals()["run_one"], globals()["OUTPUT_ROOT"], globals()["export_latex"]
    exports = []
    globals()["run_one"] = lambda forge, c: {**c, "status": outcomes[c["id"]]}
    globals()["OUTPUT_ROOT"] = tmp
    export_result = [0]
    globals()["export_latex"] = lambda author, retry=False: exports.append(author) or export_result[0]
    try:
        opts = argparse.Namespace(conjectures=2, workers=1, publish=False)
        research_run(_PaperForge, "seed", opts, run=Run(tmp / "p"))
        outcomes["c2"] = "verified"
        research_run(_PaperForge, "seed", opts, run=Run(tmp / "p"))
        research_run(_PaperForge, "seed", opts, run=Run(tmp / "p"))  # unchanged: cached
        outcomes.update(c1="known", c2="known")  # nothing to submit: the package is left alone
        research_run(_PaperForge, "seed", opts, run=Run(tmp / "q"))
        real_submit, uploads = globals()["submit_airaxiv"], []
        globals()["submit_airaxiv"] = lambda limit: uploads.append(limit) or 0
        try:  # uploading is asked for, never implied
            outcomes.update(c1="verified", c2="verified")
            research_run(_PaperForge, "seed", argparse.Namespace(conjectures=2, workers=1, publish=False, submit_airaxiv=2),
                         run=Run(tmp / "p"))
            # the upload reads the folder the export just refreshed: a failed PDF elsewhere does not stop it,
            # an export that could not run at all does
            export_result[0] = 1
            assert export_and_submit(4) == 1 and uploads == [2, 4], uploads
            export_result[0] = EXPORT_ABORTED
            assert export_and_submit(5) == EXPORT_ABORTED and uploads == [2, 4], uploads
            assert export_and_submit(None) == EXPORT_ABORTED and uploads == [2, 4], "no flag, no upload"
            export_result[0] = 0
        finally:
            globals()["submit_airaxiv"] = real_submit
    finally:
        globals()["run_one"], globals()["OUTPUT_ROOT"], globals()["export_latex"] = real_run_one, real_root, real_export
    assert papers == [["c1"], ["c1", "c2"]], papers
    assert len(exports) == 7 and uploads == [2, 4], (exports, uploads)  # every run with a paper refreshes the folder, no flag needed
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
    assert "0/1 published" in publish_verdict(True, mixed, [])
    assert publish_verdict(True, mixed, [{"url": "u"}]) == "publish: 1 result(s) published"
    assert _counterexample_line("checks ok\nNO COUNTEREXAMPLE here\nCOUNTEREXAMPLE: n=7") == "COUNTEREXAMPLE: n=7"
    assert _counterexample_line("NO COUNTEREXAMPLE up to 10^6") == ""

    # Palomar split: Challenge keeps every definition (made public: a private name
    # is mangled per module) and states only the main theorem, with one sorry
    lean_src = ("import Mathlib\n\nset_option maxRecDepth 2048\n\nprivate def f (n : ℕ) : ℕ := n + 1\n\n"
                "lemma helper : f 1 = 2 := by decide\n\n-- FAITHFULNESS: f is the successor.\n"
                "theorem refutation :\n    ¬ (∀ n, f n = n) := by\n  intro h\n  have := h 0\n  simp [f] at this\n"
                "set_option pp.fullNames true in\n#print axioms refutation\n")
    challenge, solution, qualified = palomar_split(lean_src, "refutation", "Mathforge.T")
    assert qualified == "Mathforge.T.refutation", qualified
    # a Lean identifier cannot start with a digit, and seqforge folders start with an uppercase A-number
    assert _namespace("A240513-c1234567") == "Mathforge.A240513C1234567", _namespace("A240513-c1234567")
    assert _namespace("3-term-ap-free-c2") == "Mathforge.R3TermApFreeC2", _namespace("3-term-ap-free-c2")
    assert "def f" in challenge and "private" not in challenge + solution and "helper" not in challenge
    assert "-- FAITHFULNESS" in challenge and challenge.count("sorry") == 1 and "¬ (∀ n, f n = n) := by\n  sorry" in challenge
    assert "lemma helper" in solution and "#print" not in solution and "set_option maxRecDepth" in solution
    assert solution.startswith("import Mathlib\n\nnamespace Mathforge.T\n") and solution.endswith("end Mathforge.T\n")
    assert _main_theorem({"status": "machine-verified", "lean": {"code": "theorem a : True := trivial\n"
                                                                  "theorem main_theorem : True := trivial\n"
                                                                  "theorem b : True := trivial"}}) == "main_theorem"
    yml = formalization_yaml({"id": "c1", "status": "machine-refuted", "statement": "Every n is odd.", "title": "T"},
                             {"work": "w", "review": "r"}, "Op", "Mathforge.T", "Mathforge.T.refutation",
                             {"arxiv": ["math.CO"], "msc2020": ["05A15"]})
    assert 'type: "original-proof"' in yml and 'models:\n        - "r"\n        - "w"' in yml, yml
    assert "sorry_count: 0" in yml and "axioms: []" in yml and 'lean: "Mathforge.T.refutation"' in yml
    assert "TEMPLATE" not in yml

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
    # the credit line sits under the verdict, distinct models in pipeline order
    assert "**Models:** review-z, work-y, checker-x" in credited, credited[:300]
    assert "**Models:** not recorded" in proved
    paper_credit = paper_models([{"id": "c2", "title": "Kept", "status": "verified"}], {"work": "w", "review": "r"})
    assert paper_credit.startswith("## Models") and "- Wrote the proof: w" in paper_credit
    assert "- Referee: r" in paper_credit

    pub_run = Run(OUTPUT_ROOT / "_publishtest")
    pub_forge = Forge(None, pub_run)
    posts = []
    real_publish_result = globals()["publish_result"]
    globals()["publish_result"] = lambda forge, seed, r: (
        posts.append(r["id"]),
        {"error": "offline"} if len(posts) == 1 else {"url": "https://github.com/u/mathforge-results/tree/main/x",
                                                      "title": r["id"]},
    )[1]
    try:
        one = {"id": "c1", "status": "machine-refuted", "statement": "s"}
        assert publish(pub_forge, "seed", [one, {"id": "c2", "status": "known"}]) == []
        assert publish(pub_forge, "seed", [one])[0]["url"].endswith("/tree/main/x")
        assert publish(pub_forge, "seed", [one])  # cached, no third post
        assert posts == ["c1", "c1"], posts
    finally:
        globals()["publish_result"] = real_publish_result
        shutil.rmtree(pub_run.path, ignore_errors=True)
    listing = tmp / "_results"
    (listing / "a-c1").mkdir(parents=True, exist_ok=True)
    (listing / "a-c1" / "result.json").write_text(json.dumps({
        "folder": "a-c1", "status": "machine-refuted", "headline": "Refuted: x | y", "date": "2026-09-27",
        "models": "m", "palomar": True}), encoding="utf-8")
    front = results_index(listing, "u/r")
    assert "1 result(s)" in front and "| 2026-09-27 | false | [x \\| y](a-c1/) | m | bundle |" in front, front

    # LaTeX export: title and abstract lifted out, statements and proofs become environments
    title, abstract, body = paper_meta(
        "# Title\n\nCounting **things**\n\n## Abstract\n\nWe count.\n\nTwice.\n\n## 1. Results\n\ntext\n")
    assert (title, abstract) == ("Counting things", "We count.\n\nTwice."), (title, abstract)
    assert body.strip() == "# 1. Results\n\ntext", body  # shallowest heading becomes a section
    assert paper_meta("# A $2\\times n$ law\n\nbody\n")[:2] == ("A $2\\times n$ law", "")
    # a `Title` heading wins over an earlier heading, and the title under it may itself be a heading
    late = paper_meta("# the seed topic\n\n## Title\n\n# Real title\n\n## 1. Intro\n\nx\n")
    assert late[0] == "Real title" and "Title" not in late[2] and "seed topic" not in late[2], late
    # numbered sections set the depth; an embedded proof's headings and code comments do not
    shifted = paper_meta("# T\n\n## 1. Results\n\n# Theorem\n\nX.\n\n# Parity of words\n\n```\n# comment\n```\n\n### 1.1 Sub\n")[2].splitlines()
    assert all(ln in shifted for ln in ("# 1. Results", "## 1.1 Sub", "# comment", "# Theorem", "## Parity of words")), shifted
    envs = theorem_envs(
        "### 3.1 Theorem c1: Even & odd\n\nFor all $n_1$, X.\n\n**Proof, verbatim from the verified proof:**\n\n"
        "## Lemma 2 (Rectangles)\n\nA holds.\n\n### Proof of Lemma 2\n\nBecause. ∎\n\n"
        "**Lemma 3.** B holds.\n\n**Proof.** Clear. $\\blacksquare$\n\n"
        "## 4. Checks\n\n**Theorem 1 queries:** q\n\n## Lemmas\n\n```\n**Lemma 9.** not prose\n```\n")
    assert "\\begin{theorem*}[{c1: Even \\& odd}]\n```\n\nFor all $n_1$, X." in envs, envs
    assert "\\setcounter{lemma}{1}\\begin{lemma}[{Rectangles}]" in envs and "\\begin{proof}[{{Proof of Lemma 2}}]" in envs, envs
    assert "\\setcounter{lemma}{2}\\begin{lemma}\n```\nB holds." in envs and "Because.\n" in envs and "Clear.\n" in envs, envs
    assert envs.count("\\begin{proof}") == envs.count("\\end{proof}") == 2, envs  # the empty lead-in proof is dropped
    assert envs.count("\\begin{") == envs.count("\\end{") == 5, envs
    assert "## 4. Checks" in envs and "**Theorem 1 queries:** q" in envs and "## Lemmas" in envs, envs
    assert "**Lemma 9.** not prose" in envs, envs  # code fences are left alone
    assert envs.index("\\end{proof}", envs.index("Clear.")) < envs.index("## 4. Checks"), envs
    # a theorem's proof that stops for a lemma gets no QED box; a QED mark ends a proof, and what follows is prose
    nested = theorem_envs("## Theorem 1\n\nS.\n\n## Proof\n\nSetup text.\n\n## Lemma 1\n\nL.\n\n### Proof\n\n"
                          "P. \\(\\square\\)\n\nConclusion, as required. ∎\n")
    assert nested.count("\\begin{proof}") == 1 and "*Proof.*" in nested and "\\square" not in nested, nested
    assert nested.index("*Proof.*") < nested.index("Setup text.") < nested.index("\\begin{lemma}"), nested
    assert nested.index("P.\n") < nested.index("\\end{proof}") < nested.index("Conclusion, as required. ∎"), nested
    # a lemma's proof that runs into the next lemma is complete, not interrupted
    assert theorem_envs("**Lemma 1.** A.\n\n**Proof.** x\n\n**Lemma 2.** B.\n").count("\\begin{proof}") == 1
    # a heading that carries the whole statement becomes the body, not an empty environment with a long title
    carried = theorem_envs("## Lemma 1. Every orientation of \\(F_n\\) is acyclic iff X\n\n### Proof\n\nok\n")
    assert "\\setcounter{lemma}{0}\\begin{lemma}\n```\nEvery orientation of \\(F_n\\) is acyclic iff X." in carried, carried
    # a long title stays a title when a statement follows it, unless that statement opens with a list
    long_title = "Torus scaling divisibility for joint Jordan-support class counts in the group"
    assert f"\\begin{{theorem}}[{{{long_title}}}]" in theorem_envs(f"## Theorem 2: {long_title}\n\nFor every n, X.\n")
    assert f"\\begin{{theorem}}\n```\n{long_title}." in theorem_envs(f"## Theorem 2: {long_title}\n\n- a\n- b\n")
    # past the results, `Theorem 1` labels a search log or a query list, not a statement
    zoned = theorem_envs("## 3. Results\n\n## Theorem 1\n\nS.\n\n## 4. Computational verification\n\n### 4.1 Theorem 1\n\n"
                         "The search ran.\n\nTheorem 1. The search found nothing.\n\n### 5.1 Novelty\n\n**Theorem 1**\n\nqueries\n")
    assert zoned.count("\\begin{theorem}") == 1 and "### 4.1 Theorem 1" in zoned and "**Theorem 1**" in zoned, zoned
    assert "Theorem 1. The search found nothing." in zoned, zoned
    assert "\\begin{theorem}" in theorem_envs("## 1.2 Novelty of the approach\n\nx\n\n## Theorem 1\n\nS.\n")  # only after a statement
    # a lead followed by a heading on the same line: the heading is a heading
    inline = theorem_envs("*Proof (verbatim from the verified proof).* # Displacement\n\ntext\n")
    assert "\n# Displacement\n" in inline and "\\begin{proof}" not in inline, inline
    assert not _STATEMENT.match("Theorem-like results") and _STATEMENT.match("Theorem 1 — Parity").group(4) == "Parity"
    assert _plain("a<b and c>d") == "a\\<b and c>d", _plain("a<b and c>d")
    # unmarked leads, and prose that announces the proof, end a statement
    bare = theorem_envs("Lemma 1. A holds.\n\nProof. Easy. ∎\n\n## Theorem 2\n\nS holds.\n\n"
                        "Full proof, verbatim from the verified proof:\n\ntext\n")
    assert "\\setcounter{lemma}{0}\\begin{lemma}\n```\nA holds." in bare and "\\begin{proof}\n```\nEasy.\n" in bare, bare
    assert bare.index("\\end{theorem}") < bare.index("Full proof, verbatim"), bare
    labelled = _STATEMENT.match("Theorem 1 (c1). Mod-four law")
    assert labelled and labelled.group(2, 3, 4) == ("1", "c1", "Mod-four law"), labelled
    for not_a_statement in ("Theorem list", "Lemma used", "Claim that", "Theorem data", "Theorem 1 queries:", "Lemmas"):
        assert not _STATEMENT.match(not_a_statement), not_a_statement
    for a_proof in ("Proof", "Proof.", "Proof of Lemma 2", "Proof sketch", "Proof, verbatim from the verified proof:",
                    "Proof (verbatim verified proof)."):
        assert _PROOF.match(a_proof), a_proof
    assert not _PROOF.match("Proof-of-concept code") and not _PROOF.match("Proof strategy and overview")
    assert _tex_note("a_b & $x_1$ 50%") == "a\\_b \\& $x_1$ 50\\%"
    assert _tex_note("of \\(F_n\\) and $a*b$, *it*") == "of \\(F_n\\) and $a*b$, it", _tex_note("of \\(F_n\\) and $a*b$, *it*")
    # `*` as multiplication is not emphasis; math, code and real emphasis are left alone
    assert _stars("4*I(A)*I(B), *it*, **b**, $a*b$, \\(c*d\\), `e*f`") == "4\\*I(A)\\*I(B), *it*, **b**, $a*b$, \\(c*d\\), `e*f`"
    # `_` is escaped too: `∑_{i=0}^{n−1}a_i … ∑_{j}` paired two underscores up as emphasis
    assert _plain("2*s = k*(m+1), $a \\ b$, [n] \\ (A), F_[n](x)") == "2\\*s = k\\*(m+1), $a \\ b$, [n] ∖ (A), F\\_[n]&#40;x)"
    assert _plain("A(x)=∑_{i=0}a_i, $x_1$, `a_b`, done\\_") == "A(x)=∑\\_{i=0}a\\_i, $x_1$, `a_b`, done\\_"
    # unicode-math claims ∑ and ∏ at \begin{document}, so their text form is made after it
    assert all(ch in _UNICODE_TEX for ch in "ηᵀᵃᵈᵉᵢ′₊ℓ↦⋯"), "glyphs the refutations note printed as missing"
    assert "\\AtBeginDocument{\\let\\mfsum=∑\\newunicodechar{∑}{\\ensuremath{\\mfsum}}}" in _LATEX_PREAMBLE
    fed = pandoc_input("# T\n\n## Abstract\n\na*b\n\n## 1. R\n\n" + "".join(f"x{n}" + "\\" * n + "\n" for n in range(2, 7)), "me")
    assert all(f"x{n}\\\\\n" in fed for n in range(2, 7)), fed  # a row break is two backslashes, however many were written
    assert '"author": ["me"]' in fed and "not reviewed by a person" in fed and '"abstract": "a\\\\*b"' in fed, fed
    note = refutations_note([("seed A", {
        "id": "c1", "status": "machine-refuted", "title": "T", "headline": "Every X is even", "statement": "All X are even.",
        "notation": "X is a thing.", "confirmation": {"output": "REFUTATION CONFIRMED: n=3"},
        "lean": {"sorries": 0, "faithfulness": {"verdict": "FAITHFUL", "backtranslation": "2*s"}}},
        {"work": "wm", "review": "rm"}, "https://example.org/a-c1")])
    assert note.startswith("# 1 machine-checked refutation of") and "## Abstract" in note, note[:200]
    assert "**Proposition 1.** The following statement is false. All X are even." in note, note
    assert "**Proof.** Counterexample: `n=3`" in note and "https://example.org/a-c1" in note and "wm" in note, note
    assert "2\\*s" in note and "No literature search" in note and "GitHub issues" not in note, note
    note_envs = theorem_envs(paper_meta(note)[2])
    assert "\\begin{proposition}" in note_envs, note_envs
    assert note_envs.index("\\end{proof}") < note_envs.index("Lean 4 + Mathlib"), note_envs  # the record is not part of the proof
    assert paper_meta(note)[0].startswith("1 machine-checked"), paper_meta(note)[0]  # a leading count is not a section number
    minus = refutations_note([("s", {"id": "c1", "status": "machine-refuted",
                                     "statement": "R = [n] \\ (A ∪ B) and $a\\ b$, F_[n](x,y), A\\B, x\\geq y"}, {}, "")])
    # plain-text statements: a backslash that is spaced, or sits between two sets, is set difference
    # (`A\B` reached LaTeX as the undefined command \B), and `[n](x,y)` is not a link
    assert "R = [n] ∖ (A ∪ B) and $a\\ b$, F\\_[n]&#40;x,y), A∖B, x\\geq y" in minus and "recorded in the scripts" in minus, minus
    real_gh = globals()["_gh"]

    def _no_git(*a, **k):
        raise FileNotFoundError("git")

    globals()["_gh"] = _no_git
    try:
        assert _byline() == (os.getenv("MATHFORGE_AUTHOR") or "mathforge operator"), _byline()  # git is optional
    finally:
        globals()["_gh"] = real_gh

    # every run refreshes the submission folder; an unchanged paper is not compiled again
    real_root, real_doc, real_pdf = globals()["OUTPUT_ROOT"], globals()["latex_document"], globals()["build_pdf"]
    built, verdicts = [], {"run-a": "ok; glyphs missing from the PDF: U+1", "run-f": "PDF FAILED: boom"}
    globals()["OUTPUT_ROOT"] = tmp / "_export"
    globals()["latex_document"] = lambda markdown, author: f"TEX {author} {markdown}"
    globals()["build_pdf"] = lambda folder, stem: (built.append(stem), (folder / f"{stem}.pdf").write_text("pdf"),
                                                   verdicts.get(stem, "ok"))[2]
    try:
        def _run(name, statuses, paper="# T\n\n## Abstract\n\nab\n"):
            folder = tmp / "_export" / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "state.json").write_text(json.dumps({"seed": "s", "results": [
                {"id": f"c{i}", "status": s, "title": f"R{i}"} for i, s in enumerate(statuses, 1)]}), encoding="utf-8")
            (folder / "paper.md").write_text(paper, encoding="utf-8")
            return folder

        one = _run("run-a", ["verified"])
        _run("run-b", ["provisional"], "# U\n")  # nothing verified: no package
        _run("run-p", ["verified", "provisional"], "# P\n")  # its paper states an unproved claim as a theorem
        (tmp / "_export" / "run-x").mkdir()
        (tmp / "_export" / "run-x" / "state.json").write_text("{not json", encoding="utf-8")  # must not stop the others
        assert export_latex("me") == 0 and export_latex("me") == 0 and built == ["run-a"], built
        pack = tmp / "_export" / SUBMISSIONS
        listing = (pack / "README.md").read_text(encoding="utf-8")
        assert "## T" in listing and "`run-a.pdf`" in listing and "run-b" not in listing and "1 paper(s)" in listing, listing
        assert "glyphs missing from the PDF: U+1" in listing, listing  # a warning outlives the run that compiled
        assert "`verified` R1" in listing and "referee model" in listing and "run-p" in listing.split("## Not packaged")[1], listing
        (one / "paper.md").write_text("# T2\n\n## Abstract\n\nab\n", encoding="utf-8")
        assert export_latex("me") == 0 and built == ["run-a", "run-a"], built
        assert "## T2" in (pack / "README.md").read_text(encoding="utf-8")
        _run("run-f", ["verified"])
        assert export_latex("me") == 1 and export_latex("me") == 1, "a failed PDF is an error"
        assert built == ["run-a", "run-a", "run-f"], built  # and is not compiled again until its source changes
        _run("run-a", ["known"])  # no longer qualifies: its package goes
        (tmp / "_export" / "run-f" / "state.json").write_text("{not json", encoding="utf-8")  # unreadable is not unqualified
        export_latex("me")
        assert not list(pack.glob("run-a.*")) and (pack / "run-f.tex").exists(), list(pack.iterdir())
        assert json.loads((pack / "status.json").read_text(encoding="utf-8")) == {"run-f": "PDF FAILED: boom"}
        _run("run-f", ["verified"])
        verdicts.pop("run-f")
        assert export_latex("me") == 1 and export_latex("me", retry=True) == 0, "an explicit export retries a failed build"
        (pack / "status.json").write_text("[1]", encoding="utf-8")  # a damaged record means rebuild, not a crash
        assert export_latex("me") == 0
        # a second mathforge may be exporting: stand aside, unless its lock is so old that it must have died
        (pack / "export.lock").write_text("", encoding="utf-8")
        before = len(built)
        assert export_latex("me", retry=True) == EXPORT_ABORTED and len(built) == before, built
        os.utime(pack / "export.lock", (time.time() - 3600, time.time() - 3600))
        assert export_latex("me") == 0 and not (pack / "export.lock").exists()
    finally:
        globals()["OUTPUT_ROOT"], globals()["latex_document"], globals()["build_pdf"] = real_root, real_doc, real_pdf

    # AiraXiv upload: asked for, a few papers per call, each package once, the refutations note first
    real_root, real_http, real_key = globals()["OUTPUT_ROOT"], globals()["_airaxiv_http"], os.environ.pop("AIRAXIV_API_KEY", None)
    real_export = export_latex
    sent = []

    def _fake_airaxiv(url, method="POST", body=None, headers=None):
        sent.append((method, url, body, dict(headers or {})))
        if method != "POST":
            return {}, b""
        message = json.loads(body)
        if message["method"] == "initialize":
            return {"mcp-session-id": "sid"}, b'{"jsonrpc": "2.0", "id": 1, "result": {}}'
        if message["method"] != "tools/call":
            return {}, b""
        tool, arguments = message["params"]["name"], message["params"]["arguments"]
        if tool == "submit_paper" and arguments["title"] == "Bad":
            result = {"isError": True, "content": [{"text": "rejected"}]}
        else:
            reply = {"create_upload": {"upload_id": "u1", "upload_url": "https://files.example/u1"},
                     "complete_upload": {"pdf_file_id": "f1"}, "submit_paper": {"submission_id": "s1"},
                     "list_papers": {"papers": [{"submission_id": "s1", "paper_id": "2610.1"}, {"submission_id": "s9", "paper_id": None}]},
                     "get_paper_reviews": {"reviews": [{"content": "Unclear."}]}, "update_paper": {"version": 2}}[tool]
            result = {"content": [{"text": json.dumps(reply)}]}
        return {}, json.dumps({"jsonrpc": "2.0", "id": 1, "result": result}).encode()

    globals()["OUTPUT_ROOT"], globals()["_airaxiv_http"] = tmp / "_air", _fake_airaxiv
    try:
        pack = tmp / "_air" / SUBMISSIONS
        pack.mkdir(parents=True)
        titles = {REFUTATIONS: "Note", "a": "Ta", "b": "Tb", "bad": "Bad", "c": "Tc", "g": "Tg", "z": "Tz"}
        for stem, name in titles.items():
            (pack / f"{stem}.md").write_text(f"# {name}\n\n## Abstract\n\nab {stem}\n\n## Models\n\n- Wrote the proof: m1\n"
                                             "- Referee: m2\n- Faithfulness judge: not recorded\n", encoding="utf-8")
            (pack / f"{stem}.pdf").write_bytes(b"%PDF " + stem.encode())
        (pack / "status.json").write_text(json.dumps({**{s: "ok" for s in titles}, "b": "ok; 1 line(s) run past the margin, worst by 22pt", "g": "ok; glyphs missing from the PDF: U+1",
                                                      "c": "PDF FAILED: x", "gone": "ok"}), encoding="utf-8")
        assert submit_airaxiv(2) == 1 and not sent, "no key, no request"
        os.environ["AIRAXIV_API_KEY"] = "k"
        assert submit_airaxiv(2) == 0
        record = json.loads((pack / "airaxiv.json").read_text(encoding="utf-8"))
        assert list(record) == [REFUTATIONS, "a"] and record["a"]["reply"] == {"submission_id": "s1"}, record
        puts = [c for c in sent if c[0] == "PUT"]
        assert [c[2] for c in puts] == [b"%PDF _refutations", b"%PDF a"], puts
        assert all("Authorization" not in c[3] for c in puts), "the key is not sent to another host"
        posts = [(json.loads(c[2]), c[3]) for c in sent if c[0] == "POST"]
        assert all(h["Authorization"] == "Bearer k" for _m, h in posts), posts
        assert all(h.get("Mcp-Session-Id") == "sid" for m, h in posts if m["method"] == "tools/call"), posts
        paper = next(m["params"]["arguments"] for m, _h in posts if m.get("params", {}).get("name") == "submit_paper")
        assert paper["title"] == "Note" and paper["abstract"] == "ab _refutations" and paper["pdf_file_id"] == "f1", paper
        assert paper["paper_type"] == AIRAXIV_PAPER_TYPE and paper["research_category"] == "theoretical", paper
        assert [a["type"] for a in paper["author_list"]] == ["ai", "human"] and "m1, m2" in paper["author_list"][0]["name"], paper
        assert "not recorded" not in paper["author_list"][0]["name"], paper
        sent.clear()
        assert submit_airaxiv(9) == 1, "a refused paper is an error"  # b, bad (refused), z; never c, g (glyphs missing) or gone
        record = json.loads((pack / "airaxiv.json").read_text(encoding="utf-8"))
        assert sorted(record) == sorted([REFUTATIONS, "a", "b", "bad", "z"]), record
        assert record["bad"]["state"] == "refused" and "rejected" in record["bad"]["error"] and "reply" not in record["bad"], record["bad"]
        sent.clear()  # a refused paper is not offered again as it is: it would be refused again, ahead of the others
        assert submit_airaxiv(9) == 0 and not [c for c in sent if c[0] == "PUT"], sent
        (pack / "bad.pdf").write_bytes(b"%PDF bad, rebuilt")  # a rebuilt package is a new offer
        assert submit_airaxiv(9) == 1 and [c[2] for c in sent if c[0] == "PUT"] == [b"%PDF bad, rebuilt"], sent
        # the site's rate limit ends the batch: the rest wait for the next call instead of knocking again
        (pack / "airaxiv.json").write_text("{}", encoding="utf-8")
        titles_seen = []

        def _limited(url, method="POST", body=None, headers=None):
            message = json.loads(body) if method == "POST" and body else {}
            if message.get("params", {}).get("name") == "create_upload":
                titles_seen.append(1)
                refusal = {"isError": True, "content": [{"text": "Too many requests. Please try again later."}]}
                return {}, json.dumps({"jsonrpc": "2.0", "id": 1, "result": refusal}).encode()
            return _fake_airaxiv(url, method, body, headers)

        globals()["_airaxiv_http"] = _limited
        assert submit_airaxiv(9) == 1 and titles_seen == [1], titles_seen
        assert json.loads((pack / "airaxiv.json").read_text(encoding="utf-8")) == {}, "a rate limit is not a refusal of the paper"
        # nor is the daily quota, which the site reports at submit_paper: stop, and offer the paper again tomorrow
        quota_hits = []

        def _quota(url, method="POST", body=None, headers=None):
            message = json.loads(body) if method == "POST" and body else {}
            if message.get("params", {}).get("name") == "submit_paper":
                quota_hits.append(1)
                refusal = {"isError": True, "content": [{"text": "Daily quota exceeded. Please try again tomorrow."}]}
                return {}, json.dumps({"jsonrpc": "2.0", "id": 1, "result": refusal}).encode()
            return _fake_airaxiv(url, method, body, headers)

        globals()["_airaxiv_http"] = _quota
        assert submit_airaxiv(9) == 1 and quota_hits == [1], quota_hits
        assert json.loads((pack / "airaxiv.json").read_text(encoding="utf-8")) == {}, "a quota is not a refusal of the paper"
        assert _airaxiv_busy(RuntimeError("submit_paper: Daily quota exceeded")) and not _airaxiv_busy(RuntimeError("bad PDF"))

        # a review is answered once: the run's paper is rewritten, and the new PDF goes up as a version of the public paper
        globals()["_airaxiv_http"], globals()["export_latex"] = _fake_airaxiv, lambda author, retry=False: 0
        (pack / "airaxiv.json").write_text(json.dumps({"a": {"reply": {"submission_id": "s1"}}, "z": {"reply": {"submission_id": "s9"}},
                                                       REFUTATIONS: {"reply": {"submission_id": "s1"}}}), encoding="utf-8")
        reviewed = Run(tmp / "_air" / "a")
        reviewed.data.update(paper="# Ta\n\n## Abstract\n\nab a\n", results=[])
        reviewed.save()
        rewriter = lambda run: type("Rewriter", (), {"revise": staticmethod(
            lambda paper, report: f"```markdown\n{paper}\n## 6. Changes\n\n{report}\n```")})
        sent.clear()
        assert revise_airaxiv(rewriter, 5) == 0
        updates = [json.loads(c[2])["params"]["arguments"] for c in sent if c[0] == "POST" and b"update_paper" in c[2]]
        assert len(updates) == 1 and updates[0]["paper_id"] == "2610.1" and updates[0]["pdf_file_id"] == "f1", updates
        assert "Revised against the review: not recorded" in (tmp / "_air" / "a" / "paper.md").read_text(encoding="utf-8")
        assert "- Revised against the review: m9" in paper_models([], None, "m9") and "Revised" not in paper_models([], None)
        record = json.loads((pack / "airaxiv.json").read_text(encoding="utf-8"))
        assert record["a"]["revision"]["state"] == "sent" and "revision" not in record["z"] and "revision" not in record[REFUTATIONS], record
        state = Run(tmp / "_air" / "a").data
        assert state["paper"].startswith("# Ta") and "Unclear." in state["paper"] and state["paper_v1"] == "# Ta\n\n## Abstract\n\nab a\n", state
        sent.clear()
        assert revise_airaxiv(rewriter, 5) == 0 and not [c for c in sent if c[0] == "PUT"], "one revision per paper"
    finally:
        globals()["export_latex"] = real_export
        globals()["OUTPUT_ROOT"], globals()["_airaxiv_http"] = real_root, real_http
        os.environ.pop("AIRAXIV_API_KEY", None)
        if real_key is not None:
            os.environ["AIRAXIV_API_KEY"] = real_key

    # a refusal over HTTP says how long to wait and why, when the site does
    import io
    throttled = urllib.error.HTTPError("https://airaxiv.com/mcp/", 429, "Too Many Requests", {"Retry-After": "3600"},
                                       io.BytesIO(b'{"error": {"message": "slow down"}}'))
    reason = _airaxiv_reason(throttled)  # once: the reply body can only be read once
    assert "429" in reason and "retry after 3600" in reason and "slow down" in reason, reason
    assert _airaxiv_reason(ValueError("x")) == "ValueError: x"
    # AiraXiv upload: the ways it could send a paper twice, send too many, or send the key elsewhere
    real_root, real_http, real_key = globals()["OUTPUT_ROOT"], globals()["_airaxiv_http"], os.environ.pop("AIRAXIV_API_KEY", None)
    traffic = []

    def _site(upload_url="https://airaxiv.com/upload/u1", submit=None):
        def http(url, method="POST", body=None, headers=None):
            traffic.append((method, url, dict(headers or {})))
            if method != "POST":
                return {}, b""
            message = json.loads(body)
            if message["method"] != "tools/call":
                return {"Mcp-Session-Id": "sid"}, b'{"jsonrpc": "2.0", "id": 1, "result": {}}'
            tool = message["params"]["name"]
            if tool == "submit_paper" and submit is not None:
                return submit()
            reply = {"create_upload": {"upload_id": "u1", "upload_url": upload_url},
                     "complete_upload": {"pdf_file_id": "f1"}, "submit_paper": {"submission_id": "s1"}}[tool]
            return {}, json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": [{"text": json.dumps(reply)}]}}).encode()
        return http

    globals()["OUTPUT_ROOT"] = tmp / "_air2"
    os.environ["AIRAXIV_API_KEY"] = "k"
    try:
        pack = tmp / "_air2" / SUBMISSIONS
        pack.mkdir(parents=True)
        stems = "abcdefghijklm"
        for stem in stems:
            (pack / f"{stem}.md").write_text(f"# T{stem}\n\n## Abstract\n\nab\n", encoding="utf-8")
            (pack / f"{stem}.pdf").write_bytes(b"%PDF")
        (pack / "status.json").write_text(json.dumps({stem: "ok" for stem in stems}), encoding="utf-8")
        sent_record = pack / "airaxiv.json"
        puts = lambda: [c for c in traffic if c[0] == "PUT"]
        for url, keyed in (("https://airaxiv.com/upload/u1", True), ("https://airaxiv.com.evil.example/u1", False),
                           ("https://airaxiv.com@evil.example/u1", False), ("https://files.example/u1", False)):
            traffic.clear()
            sent_record.unlink(missing_ok=True)
            globals()["_airaxiv_http"] = _site(url)
            assert submit_airaxiv(1) == 0 and ("Authorization" in puts()[0][2]) == keyed, url  # the key goes to AiraXiv's host only
        traffic.clear()
        sent_record.unlink()
        globals()["_airaxiv_http"] = _site("http://airaxiv.com/u1")
        assert submit_airaxiv(1) == 1 and not puts() and not sent_record.exists(), "a PDF is not sent over plain http"
        globals()["_airaxiv_http"] = _site()
        for damaged in ('{"a": ', "", "[]"):  # an unreadable record is not "nothing sent yet"
            sent_record.write_text(damaged, encoding="utf-8")
            traffic.clear()
            assert submit_airaxiv(3) == 1 and not traffic, damaged
        sent_record.unlink()
        traffic.clear()
        assert submit_airaxiv(-1) == 0 and not puts(), "a negative count sends nothing"
        assert submit_airaxiv(100) == 0 and len(puts()) == AIRAXIV_MAX, len(puts())
        (pack / "airaxiv.lock").write_text("", encoding="utf-8")  # another upload is running
        traffic.clear()
        assert submit_airaxiv(3) == 1 and not traffic
        (pack / "airaxiv.lock").unlink()

        def _lost():
            raise TimeoutError("timed out")

        for lost in (_lost, lambda: ({}, b'{"jsonrpc": "2.0", "id": 1, "result": {"content": []}}'),
                     lambda: ({}, b'{"jsonrpc": "2.0", "id": 1, "result": {"content": [{"text": "not json"}]}}')):
            sent_record.unlink(missing_ok=True)
            globals()["_airaxiv_http"] = _site(submit=lost)
            assert submit_airaxiv(1) == 1
            assert json.loads(sent_record.read_text(encoding="utf-8"))["a"]["state"] == "submitting"
            traffic.clear()
            globals()["_airaxiv_http"] = _site()
            assert submit_airaxiv(1) == 1, "an unresolved submission keeps the call failing until someone looks"
            record = json.loads(sent_record.read_text(encoding="utf-8"))
            assert record["a"]["state"] == "submitting" and record["b"]["reply"] == {"submission_id": "s1"}, record
            assert len(puts()) == 1, "the paper whose reply was lost is not sent again"
            assert not list(pack.glob("airaxiv.lock")) and not list(pack.glob("*.tmp")), list(pack.iterdir())
    finally:
        globals()["OUTPUT_ROOT"], globals()["_airaxiv_http"] = real_root, real_http
        os.environ.pop("AIRAXIV_API_KEY", None)
        if real_key is not None:
            os.environ["AIRAXIV_API_KEY"] = real_key

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
        assert latest_run_dir(tmp / "missing") is None
        assert latest_run_dir(tmp) == newer
        scratch = tmp / "_scout"  # scratch dirs are not runs
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / "state.json").write_text("{}", encoding="utf-8")
        os.utime(scratch / "state.json", (time.time() + 120, time.time() + 120))
        assert latest_run_dir() == newer, latest_run_dir()
        assert latest_run_dir(tmp / "missing") is None
        assert latest_run_dir(tmp) == newer
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
