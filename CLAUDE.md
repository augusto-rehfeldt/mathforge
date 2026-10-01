# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python mathforge.py --selftest              # the entire test suite; offline, no API calls
python mathforge.py "seed topic"            # one run
python mathforge.py --forever --workers 4   # continuous mode, self-chosen topics
python mathforge.py --resume math_output/<slug>   # bare --resume takes the newest run
python mathforge.py --setup-lean            # elan + Mathlib Lake project (multi-GB); runs also do this
```

`--selftest` is intercepted in `__main__` before `argparse` runs, so it is not a
declared flag. It is the only test entry point: `_selftest()` is one function of
`assert`s covering JSON/code-block salvage, `classify_search`, script execution
and timeouts, `write_and_run` repair loops (via a stub AI), literature parsing
(canned payloads, `_http_get` monkeypatched), publishing, run-directory
bookkeeping, and `Run.stage` caching. There is no pytest, no framework, no
per-function suite — add new checks inline in `_selftest`, next to the ones for
the same area. There is no linter config; match the existing style.

Running a single check means editing `_selftest` locally; the whole thing takes
seconds.

## Architecture

One file, `mathforge.py`, ~1700 lines. The layering is by section, top to
bottom: parsing helpers → logging → `Run` (state) → subprocess execution →
literature backends → Lean → `Forge` (the agents) → `pipeline` → outputs →
publishing → CLI.

**The design premise**: a model's claim to have proved something is worth
nothing; its code is worth something because code has an exit status. Every gate
is an executed program, a compiler, or a model reading evidence it did not
produce. Changes that replace an executable gate with a model's opinion break
the premise of the project.

**`Run.stage(key, produce)`** is the spine. Every expensive step goes through
it: the result is cached in `state.json` under `key` and never recomputed, which
is what makes `--resume` work, and it is also the single place start/finish/
heartbeat logging lives. Adding a stage means adding a `run.stage(...)` call in
`pipeline`, not a print at the call site.

**`Forge`** holds one `AIService` and one `Run`. Each method is one agent. Two
models are used deliberately: `model_type="writing"` (the work model) proves,
falsifies and formalizes; `model_type="review"` (the review model) proposes,
referees, judges novelty, writes the independent check, and judges
faithfulness. The split exists so the two *executable* verdicts (adversarial
search, independent check) do not come from the same model. Do not consolidate
them. Proposing sits on the review model so the prover faces a statement it did
not author; the cost is that the model judging novelty is the one that made the
conjecture, which is why novelty rests on retrieved literature rather than on
that judgement alone.

**`Forge.ask`** is the only call into the AI client. Provider usage limits are
waited out inside book writer's `AIService.generate_content` (see its CLAUDE.md),
so a limit notice never reaches the JSON or code-block parsers here.
`propose` asks proposers that produced nothing once more before giving up.

**`Forge.write_and_run`** is the code-writing loop: ask for code, run it, hand
failures back up to `MAX_CODE_REPAIRS` times. It repairs crashes and
*silent* runs (exit 0 with no marker from `markers`) — never the verdict itself.
A clean run that refutes the conjecture is a result, not a bug to be repaired.

**`pipeline(forge, c)`** is one conjecture end to end: falsify (→ confirm →
Lean refutation, when the search reports a witness) → novelty →
prove → referee (+ one repair round if not VALID) → independent check → Lean →
faithfulness. A reported witness is only a claim: `cN.confirm` (review model,
`refutation_confirmed`) re-checks it from the statement, and a rejection makes
the result `inconclusive` — live falsifiers compared against the wrong quantity,
mis-computed a witness, and used one outside the hypotheses. A confirmed one goes
to `cN.lean_refute` (`theorem refutation : ¬ (claim)`) and
`cN.refute_faithfulness` (`faithfulness(..., negated=True)`); sorry-free and
faithful gives `machine-refuted`, otherwise `refuted`. Lean is required: a run
with no ready project calls `setup_lean` itself, which installs elan unattended
if `lake` is missing (`install_elan`), creates `~/mathforge-lean` on the
toolchain Mathlib pins (`lake +leanprover-community/mathlib4:lean-toolchain new`),
fetches the cache and builds. Every step is repeatable and only a finished build
writes `LEAN_READY`, so an interrupted setup resumes on the next run. `_lake()`
also looks in `~/.elan/bin`, where the VS Code Lean extension installs elan
without updating an open shell's PATH. Only `--no-lean` skips all of this.
Early exits produce `machine-refuted`/`refuted`/`inconclusive`/`known`; a crashed
conjecture is recorded as `error` by `run_one` (which `research_run` calls for
every conjecture) without killing the rest of the run, and nothing is cached
for it so `--resume` retries. The final
status ladder: `machine-verified` requires sorry-free Lean *and* a FAITHFUL,
non-`trivialized` back-translation (an `axiom` declaration, with or without
modifiers, counts as a sorry, via `lean_verdict`); `verified` requires referee
VALID *and* `check_passed` (exit 0, `ALL CHECKS PASSED`, no `CHECK FAILED`);
otherwise `provisional`. Faithfulness is its own `cN.faithfulness` stage, so a
garbled judge reply does not throw away the Lean run; older runs cached it
inside `cN.lean` and are read as-is.

`RULES` requires every conjecture to quantify over an infinite family, and
`next_seed` asks for an area rather than a computation: seeds phrased "for n <= 10,
enumerate ..." produced "theorems" confined to the enumerated range, which the
search decides outright (the 2026-08-15 Motzkin run's three `verified` results
were of that kind).

`sorry_free` is decided by Lean itself: `_axiom_probe` appends `#print axioms`
for every theorem (qualified by its `namespace`), and `lean_verdict` requires at
least one report and nothing outside `propext`/`Classical.choice`/`Quot.sound` —
`sorryAx`, a hidden `axiom`, `native_decide`'s `Lean.ofReduceBool` all fail it.
`_theorems` parses names from the source with comments stripped (`«…»` names,
`.{u}`, `nonrec`, `set_option … in`) and also counts every `theorem`/`lemma`
keyword; if the two disagree, or fewer reports come back than theorems, it is not
`sorry_free`. An error only on the probe's appended lines is not sent for repair
(`_run_lean_probed`). The expected `#print axioms` output format has not yet been
checked against a real Lean install (none on this machine); a mismatch fails safe
— nothing is ever `sorry_free` — so check one run after `--setup-lean`.
The source regex stays as a backstop. Every verdict is read from `_clip`ped
output, and `_clip` keeps each `CLIP_KEEP` marker line from the middle, so a
`CHECK FAILED` or `declaration uses 'sorry'` cannot fall out of a long log.
`BOUNDED_SEED` marks bounded-computation seeds in the history `next_seed` reads,
so their old `verified` tallies do not teach it that phrasing.

Resume invariants: an unusable proposer reply is dropped from `state.json` — which
only matters when *every* proposer failed, since a partial `conjectures` list is
cached and a resume does not ask the missing proposers again — the
`paper` stage is redone when the set of keepers changes (`paper_keepers`; a paper
cached before that key existed is redone once), and
`append_index` replaces a resumed run's entry instead of appending a duplicate.
An unreadable `index.json` is renamed to `index.damaged-<time>.json`, never reset.
Agents read JSON through `Forge.ask_json(prompt, key)`: `_json_object(reply, key)`
picks the first object carrying the expected key out of whatever `_json_block`
salvaged, and a reply with none is asked again (up to `MAX_JSON_RETRIES` = 3
times), told it has no tools — live
review models answered "I'll brute-force the formula" and stopped there.

**Parallelism** is `ThreadPoolExecutor` at two levels — one call per proposer in
`Forge.propose`, and one thread per conjecture in `research_run`. Proposals are
fanned out one-per-call on purpose: a whole-batch reply is the longest
completion in the pipeline and is exactly where a reasoning model exhausts its
output budget and returns nothing, losing the run. `Run` has a lock; `log` has a
print lock; every line carries a `cN` tag because stages are minutes apart and
interleaved. Backend request spacing (arXiv's 3 seconds) is enforced globally
across threads by `_RateLimiter`, not by a per-thread sleep.

**Credentials and the AI client are not implemented here.** They are imported
from the shared `ai_suite` package (`AIService`, `choose_ai`, `load_local_env`): the
sibling `ai-suite` checkout (`AI_SUITE_DIR` overrides it), else the vendored `ai_suite/`
copy in this repo, synced by ai-suite's `sync.py` -- never edit it here. AIService
reads the opencode CLI's `auth.json`. Models and token budgets are passed by
setting `AI_WRITING_MODEL` / `AI_REVIEW_MODEL` / `AI_*_COMPLETION_TOKENS` env
vars before constructing `AIService`, so the shared config file stays untouched.
`set_reasoning_effort` delegates to the public AIService method; SDK methods are never wrapped.

Without `--config`, provider and models come from the shared menu
(`ai_suite.choose_ai`, roles work + review, live model list). It asks
on a terminal and reuses the last pick otherwise; picks are remembered in
`math_output/provider_state.json`, first defaults opencode-go with
deepseek-v4-pro / deepseek-v4.1-flash. `--provider` skips the provider question;
`--model` / `--review-model` still win over the menu.

Any ai-suite provider config works, so OpenAI models are reachable two ways:
`--config <ai-suite>/ai_suite/config/ai_config_openai.local.json` (API
key) or `ai_config_openai_oauth.json` (ChatGPT sign-in — `AIService.__init__`
starts `npx openai-oauth` on `127.0.0.1:10531` and opens a browser the first
time). `--model` / `--review-model` override whatever model ids the config
names. The oauth proxy's catalogue is live (checked 2026-09-26: `gpt-6-sol`,
`gpt-6-luna`, `gpt-6-astra`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`,
`gpt-5.5`), and ai-suite's oauth default is `gpt-6-sol`, so a Sol+Luna pairing
no longer needs the API-key route. On `ai_config_openai.local.json`,
`openai_daily_token_limits` buckets every model outside `openai_big_models` as
`mini` (the 2.5M/day budget).

**Provider quirk worth knowing**: the opencode proxy ignores the requested
completion-token cap, so `--max-tokens` is advisory there. Reasoning effort is
the knob that bites — an empty reply with `finish_reason=length` means the model
spent its whole allowance thinking, and the fix is a *lower* `--effort`.
Effort comes from the shared menu, which asks one per role from the levels that role's
model lists (`resolve_efforts`): a `--effort` / `--review-effort` flag wins, then the
menu's pick (`AI_WRITING_EFFORT` / `AI_REVIEW_EFFORT`); with no pick an attended run
sends the provider's default and an unattended or `--config` run falls back to
`DEFAULT_EFFORT` / `DEFAULT_REVIEW_EFFORT`. The roles keep separate efforts even
when both use the same model. AIService applies the request shape appropriate
to Responses or Chat Completions. Run the workspace book-writer/music-writer/mathforge
checks together after changing that public contract.

### Parsing model output

Model replies are unreliable, and the salvage logic in `_json_block` /
`_code_block` encodes shapes live runs actually produced: fenced or bare JSON,
concatenated objects with no enclosing array, NDJSON, and replies truncated
mid-object (complete objects are kept, the partial one dropped). `classify_search`
strips `NO COUNTEREXAMPLE` before looking for `COUNTEREXAMPLE`, because the
negative marker contains the positive one — a trap a live run walked into.
`canon_negatives` first rewrites `NO-COUNTEREXAMPLE`, `NO_COUNTEREXAMPLES` and
lowercase spellings to the canonical one; `pipeline` applies it to the cached
search output too, so old runs re-classify on `--resume`. Both
markers present means the script ignored its brief: `inconclusive`, not refuted.

### Output layout

`math_output/<slug>/` holds `state.json` (every stage, resumable), the generated
`.py`/`.lean` scripts, `paper.md` and `negative_results.md`. Runs append to
`math_output/index.md` and `index.json`; `index.json` doubles as the history fed
to `Forge.next_seed` in continuous mode so it avoids repeating topics.
`_scout` and `_selftest` are scratch dirs and are excluded from `latest_run_dir`.

### Publishing

`--publish` publishes each result whose status rests on Lean -- `machine-verified`
and `machine-refuted` only (`PUBLISH_STATUSES`) -- as a folder of one public GitHub
repository (`RESULTS_REPO`, default `<gh user>/mathforge-results`, checkout
`RESULTS_CHECKOUT`; `_results_checkout` creates and clones it on first use).
`publish_result` writes the folder (`<run dir>-<cN>`: README from `_publication`,
the generated scripts, `result.json`), regenerates the root README from every
`result.json` (`results_index`), commits and pushes, under `_REPO_LOCK`. URLs are
cached in `state.json` under `cN.published`, so a resumed run reuses rather than
republishes, and a failed publish retries on the next resume. A cached URL
(`cN.published`, or an old `cN.gist`) whose result no longer qualifies is logged
loudly, never deleted automatically. `--publish-existing` runs `publish` over every
run under `math_output` and exits.

Credit: `_publication` puts a `**Models:**` line (`models_used`) under the verdict
and ends with a `## Models` section (`_model_rows`); `paper.md` gets the same
section (`paper_models`), appended at write time rather than cached, and each
refutation in `negative_results.md` gets a `**Models.**` line. Code artifacts
record their own `model` in `write_and_run`; the rest fall back to the run's
`models` pair that `Forge.__init__` stores in `state.json`; runs from before that
record say `not recorded`.

Palomar (palomar-registry.org, a registry of Lean-verified mathematics) is prepared
for, never submitted to: it asks for human review and a research-interest floor.
`palomar_bundle` writes a Lake project into the folder: `palomar_split` cuts the
checked Lean file into top-level chunks (`_lean_chunks`) and wraps both halves in
`namespace Mathforge.<Folder>`; Challenge keeps every definition and states only the
main theorem (`refutation`, or `main_theorem`, which the Lean prompt now asks for,
else the last theorem) with `sorry`; Solution is the whole file minus `#` commands.
`private` is stripped from both, because a private name is mangled with its module
and Comparator would see two different definitions. Both halves are elaborated
first; a failure means no bundle, and the README says why. `formalization.yaml`
(v0.4, `formalization_yaml`) is written with JSON-quoted scalars; `Forge.classify`
(review model, cached as `cN.classify`) supplies the arXiv/MSC codes, falling back
to math.CO / 05A99. `lake comparator` itself has not been run locally.

### Submission packages

`export_latex` (run by `research_run` after every run with a paper or a
`machine-refuted` result, and by `--export-latex`) fills `math_output/_submissions/`:
one `.md`/`.tex`/`.pdf` per run paper whose keepers are all `verified` or
`machine-verified` (a paper with a `provisional` keeper is listed under "Not packaged":
it typesets that claim as a theorem), `_refutations.*` from `refutations_note` (all
`machine-refuted` results, each a `**Proposition k.**` / `**Proof.**` pair), a
`README.md` index with every result's status, and `status.json`. It uploads nothing
and is separate from `--publish`, whose Lean-only rule (`PUBLISH_STATUSES`) is unchanged.

`pandoc_input` builds what pandoc reads. `paper_meta` lifts title and abstract into the
metadata block and raises the paper's numbered sections to top level (an embedded proof
brings its own headings at any depth). `theorem_envs` turns theorem/lemma/proof headings,
bold leads and unmarked `Lemma 1.` / `Proof.` leads into LaTeX environments, emitted as
raw `{=latex}` blocks so the text between stays Markdown. Its rules, each from a live
paper: a statement ends at the next heading, lead, or proof announcement
(`_STATEMENT_END`); a proof ends at its QED mark (`_QED`) or the next heading or lead; a
theorem's proof cut short by a lemma is written as plain `*Proof.*` text with no QED box;
a heading that carries the whole statement (nothing under it, or only a list) becomes the
body; an empty lead-in proof is dropped; after a numbered verification, limitations or
novelty section (`_AFTER_RESULTS`) no environment opens, because `Theorem 1` there heads
a search log or a query list. `_stars` escapes `*` used as multiplication and `_plain` also handles a spaced
backslash and `[n](x)` in plain-text statements, both through `_outside_math` so math and
code are untouched. `latex_document` calls pandoc with `tex_math_single_backslash` on and
`superscript`/`subscript` off (papers write `\(..\)`, statements write `x^{k}`), then
splices `_LATEX_PREAMBLE` (amsthm environments, `newunicodechar` for `_UNICODE_TEX`).

`build_pdf` runs tectonic or xelatex and reports missing glyphs (add the character to
`_UNICODE_TEX`) and lines more than 20pt past the margin. `status.json` maps each package
to its last build status: a paper is compiled again only when its `.tex` changes, so a
failing one is not retried every run (`--export-latex` passes `retry=True` and does retry
it), and the package of a run that stops qualifying is deleted; a run whose `state.json`
is unreadable keeps its package. The selftest stubs `latex_document` and `build_pdf`, so it needs neither tool,
and `_byline` falls back when git is missing. Two mathforge processes exporting at once
are not coordinated.

`submit_airaxiv` (`--submit-airaxiv [N]`, never implied by anything else) uploads `ok`
packages to airaxiv.com through its MCP endpoint, spoken as plain JSON-RPC over
`_airaxiv_http` (the one network call; the selftest replaces it): `initialize`, then per
paper `create_upload` → `PUT` the PDF → `complete_upload` → `submit_paper`. Guards against
sending twice, each from a review finding: `airaxiv.lock` allows one upload at a time;
`airaxiv.json` is written through a temp file, and an unreadable one aborts the call
(only a missing file means nothing was sent); an entry `{"state": "submitting"}` is
written before `submit_paper` and replaced by the reply, so a lost or unreadable reply
(`ValueError`, a timeout) leaves a marker that is reported and never retried, while a
refusal (`RuntimeError`, an HTTP error) removes it and is tried again next time. N is
clamped to 0..`AIRAXIV_MAX`, and a rate-limit reply ends the batch. The API key goes
only to AiraXiv's own host over https (parsed hostname, not a string prefix), never to
a signed upload URL elsewhere, and `_NoRedirect` refuses redirects. Packages whose
status mentions missing glyphs are held back. Every caller goes through
`export_and_submit`, which refreshes the folder first and sends nothing when
`export_latex` returns `EXPORT_ABORTED` (pandoc missing), so the upload never reads a
stale `status.json`. A `submit_paper` refusal other than the rate limit is recorded as
`{"state": "refused", "sha256": ...}` and the paper is offered again only when its PDF's
hash differs; otherwise a refused paper at the head of the queue would be retried, and
would block the others, on every call. The request shape matches the endpoint's own `tools/list` schema and was run live on
2026-10-01 (submission 1104). `AIRAXIV_API_KEY` is read from this project's `.env`
(`load_local_env(HERE / ".env")`); a bare `load_local_env()` reads the ai-suite
checkout's `.env` instead.

The novelty search also asks AiraXiv (`search_airaxiv`, no key): its public search page
is scraped for result cards, so a result this pipeline already uploaded there is seen as
known. The site matches the query as one phrase; a query with no hit is asked again by
its longest word.

## Security

Every agent-written script is executed with `subprocess`, with `cwd` set to the
run directory and a wall-clock timeout, but with **no sandbox**. That is
arbitrary code execution on the host, and it is a documented, accepted property
of the design — do not silently "fix" it, and do not add anything that widens it
(e.g. running generated code outside the run directory, or removing a timeout).
