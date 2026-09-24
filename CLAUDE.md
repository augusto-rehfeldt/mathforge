# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python mathforge.py --selftest              # the entire test suite; offline, no API calls
python mathforge.py "seed topic"            # one run
python mathforge.py --forever --workers 4   # continuous mode, self-chosen topics
python mathforge.py --resume math_output/<slug>   # bare --resume takes the newest run
python mathforge.py --setup-lean            # one-off Mathlib Lake project (multi-GB)
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

**`pipeline(forge, c)`** is one conjecture end to end: falsify → novelty →
prove → referee (+ one repair round if not VALID) → independent check → Lean →
faithfulness. Early exits produce `refuted`/`inconclusive`/`known`; a crashed
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
Agents read JSON through `_json_object(reply, key)`, which picks the first object
carrying the expected key out of whatever `_json_block` salvaged.

**Parallelism** is `ThreadPoolExecutor` at two levels — one call per proposer in
`Forge.propose`, and one thread per conjecture in `research_run`. Proposals are
fanned out one-per-call on purpose: a whole-batch reply is the longest
completion in the pipeline and is exactly where a reasoning model exhausts its
output budget and returns nothing, losing the run. `Run` has a lock; `log` has a
print lock; every line carries a `cN` tag because stages are minutes apart and
interleaved. Backend request spacing (arXiv's 3 seconds) is enforced globally
across threads by `_RateLimiter`, not by a per-thread sleep.

**Credentials and the AI client are not implemented here.** They are imported
from the sibling `book writer` project (`ai_book_creator.services.ai_service.AIService`,
`ai_book_creator.env.load_local_env`), located via `MATHFORGE_BOOK_WRITER`, which
reads the opencode CLI's `auth.json`. Models and token budgets are passed by
setting `AI_WRITING_MODEL` / `AI_REVIEW_MODEL` / `AI_*_COMPLETION_TOKENS` env
vars before constructing `AIService`, so the shared config file stays untouched.
`set_reasoning_effort` delegates to the public AIService method; SDK methods are never wrapped.

Without `--config`, provider and models come from book writer's shared menu
(`ai_book_creator.cli.choose_ai`, roles work + review, live model list). It asks
on a terminal and reuses the last pick otherwise; picks are remembered in
`math_output/provider_state.json`, first defaults opencode-go with
deepseek-v4-pro / deepseek-v4-flash. `--provider` skips the provider question;
`--model` / `--review-model` still win over the menu.

Any book-writer provider config works, so OpenAI models are reachable two ways:
`--config <book writer>/ai_book_creator/config/ai_config_openai.local.json` (API
key) or `ai_config_openai_oauth.json` (ChatGPT sign-in — `AIService.__init__`
starts `npx openai-oauth` on `127.0.0.1:10531` and opens a browser the first
time). `--model` / `--review-model` override whatever model ids the config
names. The two routes do not serve the same catalogue: the oauth proxy lists
`gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4-mini`; `gpt-5.6-sol` exists
only on the API-key route, so a Sol+Luna pairing has to go through
`ai_config_openai.local.json`. That config's `openai_daily_token_limits` bucket
every model outside `openai_big_models` as `mini`, so the gpt-5.6 pair currently
draws on the 2.5M/day mini budget.

**Provider quirk worth knowing**: the opencode proxy ignores the requested
completion-token cap, so `--max-tokens` is advisory there. Reasoning effort is
the knob that bites — an empty reply with `finish_reason=length` means the model
spent its whole allowance thinking, and the fix is a *lower* `--effort`.
`--effort` sets the writing role and `--review-effort` sets the review role, including
when both roles use the same model. AIService applies the request shape appropriate
to Responses or Chat Completions. Run the workspace book-writer/music-writer/mathforge
checks together after changing that public contract.

### Parsing model output

Model replies are unreliable, and the salvage logic in `_json_block` /
`_code_block` encodes shapes live runs actually produced: fenced or bare JSON,
concatenated objects with no enclosing array, NDJSON, and replies truncated
mid-object (complete objects are kept, the partial one dropped). `classify_search`
strips `NO COUNTEREXAMPLE` before looking for `COUNTEREXAMPLE`, because the
negative marker contains the positive one — a trap a live run walked into. Both
markers present means the script ignored its brief: `inconclusive`, not refuted.

### Output layout

`math_output/<slug>/` holds `state.json` (every stage, resumable), the generated
`.py`/`.lean` scripts, `paper.md` and `negative_results.md`. Runs append to
`math_output/index.md` and `index.json`; `index.json` doubles as the history fed
to `Forge.next_seed` in continuous mode so it avoids repeating topics.
`_scout` and `_selftest` are scratch dirs and are excluded from `latest_run_dir`.

### Publishing

`--publish` posts a **public** gist through the `gh` CLI for each result whose
status rests on a machine verdict — `machine-verified` and `refuted` only
(`PUBLISH_STATUSES`). Gist URLs are cached in `state.json` so a resumed run
reuses rather than reposts, and a failed post retries on the next resume. A gist
is public the instant it is created.

## Security

Every agent-written script is executed with `subprocess`, with `cwd` set to the
run directory and a wall-clock timeout, but with **no sandbox**. That is
arbitrary code execution on the host, and it is a documented, accepted property
of the design — do not silently "fix" it, and do not add anything that widens it
(e.g. running generated code outside the run directory, or removing a timeout).
