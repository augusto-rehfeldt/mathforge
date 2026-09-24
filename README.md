# mathforge

A multi-agent pipeline that looks for small, original mathematical results and
then tries very hard to destroy them. Whatever survives becomes a paper.

The design assumption is that a language model's *claim* to have proved
something is worth nothing, and its *code* is worth something, because code has
an exit status. Every gate in the pipeline is therefore either an executed
program, a compiler, or a model reading evidence it did not produce.

## Pipeline

| Stage | Agent | What it sees |
|---|---|---|
| Propose | work model | the seed topic only |
| Falsify | work model, writes a search script | the claim, not the proposer's reasoning — it is rewarded for breaking it |
| Novelty | review model + arXiv, Crossref, OpenAlex | the statement, plus retrieved abstracts |
| Prove | work model | the claim and the search evidence |
| Referee | review model | the proof, told to find the error |
| Independent check | review model, writes a second script | the proof; re-implements every definition from scratch |
| Lean | work model, `lake env lean` | Mathlib; a `sorry` is a failure, not a shortcut — and so is an `axiom` |
| Faithfulness | review model | the Lean file, back-translated and compared to the informal claim |
| Paper | work model | only what survived |

The work model and the review model are deliberately different: the prover and
the falsifier run on one, the referee and the independent checker on the other,
so the two executable verdicts do not share a single model's blind spots.

### Outcomes

| Status | Meaning |
|---|---|
| `refuted` | the adversarial search produced a counterexample |
| `inconclusive` | the search crashed, timed out, or never reported |
| `known` | the novelty referee found it in the literature or recalled it |
| `provisional` | proved, but the referee or the independent check objected |
| `verified` | referee accepted and an independently written script re-derived it |
| `machine-verified` | Lean 4 + Mathlib accepted it with no `sorry`, and the Lean statement back-translates faithfully |
| `error` | the pipeline itself crashed on this conjecture (API or parsing failure); nothing is cached, so `--resume` retries it |

## Usage

```bash
python mathforge.py "additive structure of squarefree numbers"
python mathforge.py "seed topic" --conjectures 5 --workers 3
python mathforge.py --forever --workers 4      # picks its own topics, paper after paper
python mathforge.py --runs 10 --pause 300      # ten papers, five minutes apart
python mathforge.py --resume math_output/<slug>
python mathforge.py --forever --publish        # public gist per machine-checked result
python mathforge.py --selftest                 # offline, no API calls
```

Each run writes `math_output/<slug>/`: `state.json` (every stage, resumable),
the generated scripts, `paper.md`, and `negative_results.md`. Runs are appended
to `math_output/index.md`.

`negative_results.md` is deliberate. A counterexample is a small true fact, and
a *failed* counterexample search is the evidence that supports a conjecture —
both are normally thrown away, and neither is expensive to keep.

### Publishing

`--publish` posts a **public** GitHub gist for each result whose status comes
from a machine verdict rather than a model's opinion: `machine-verified` (Lean
compiled it sorry-free and the back-translation was faithful) and `refuted` (an
adversarial script printed a counterexample). Nothing else is published, and
nothing at all without the flag.

Each gist carries the statement, the proof or the counterexample, the
verification detail, and the generated `.lean` / `.py` artifacts so a reader can
re-run them — plus a header stating that this is an unreviewed automated
artifact and may be a rediscovery. Gist URLs are recorded in `state.json` and in
`math_output/index.md`; a resumed run reuses the URL instead of posting twice,
and a failed post is retried on the next resume.

Requires the GitHub CLI logged in (`gh auth login`). Note that a gist is public
the moment it is created, before anyone has read it.

### Lean

Formalization is skipped unless a Mathlib project is present. To enable it:

```bash
python mathforge.py --setup-lean     # creates ~/mathforge-lean, downloads several GB
```

Or point `--lean-project` / `MATHFORGE_LEAN_PROJECT` at an existing Lake project.

### Models and credentials

The AI client, opencode credential loading and usage accounting come from the
sibling `book writer` project; set `MATHFORGE_BOOK_WRITER` if it lives elsewhere.
Credentials are read from the opencode CLI's own `auth.json`, so
`opencode auth login` is the only setup.

```bash
python mathforge.py "topic" --model deepseek-v4-pro --review-model deepseek-v4-flash
```

Optional environment: `S2_API_KEY` adds Semantic Scholar to the novelty search,
`OPENALEX_MAIL_ADDRESS` is OpenAlex API etiquette, `MATHFORGE_OUTPUT` relocates
the library.

## Security

Every script the agents write is executed with `subprocess`. That is arbitrary
code execution on the host. Scripts run inside the run directory with a
wall-clock timeout, but there is no sandbox — run this where that is acceptable,
or point `MATHFORGE_OUTPUT` at a container.

## Honest limitations

- **Novelty is a bounded automated search**, not a literature review: three
  databases, a handful of model-written queries, one refinement round. It is
  blind to books, to journals outside those indexes, and to anything phrased
  differently. Papers say so in their own limitations section.
- **A sorry-free Lean proof certifies the Lean statement**, and the
  back-translation check that the Lean statement matches the informal one is a
  screening filter, not an oracle. There is no mechanical oracle for this.
- **A clean brute-force search is evidence, not proof.** It bounds where a
  counterexample is not.
- **Rediscovery looks like discovery.** Small results that no one bothered to
  publish will pass every gate here.

## License

Released into the public domain under [CC0 1.0](LICENSE).
