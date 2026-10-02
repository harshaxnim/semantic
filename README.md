# semantic
Infinite visibilities

A harness that solves long, multi-step problems by decomposing them along many
"lenses" (divide and conquer, abstraction/refinement, constraint propagation,
change of basis, ...) and letting a swarm of cheap models solve, verify and
compose the pieces.

- [`docs/plan.html`](docs/plan.html): thesis, architecture, benchmark selection and game plan.
- [`docs/index.html`](docs/index.html): progress dashboard, deployed to GitHub Pages by
  `.github/workflows/pages.yml`. Edit `docs/progress.json` to update phases and the log.

## Quick start

```bash
pip install -e '.[dev]'
python -m pytest -q

# run a sweep (models: anthropic:<id>, openai:<id>, vllm:<id>, oracle, noisy-oracle:<p>)
python -m semantic.run --task blocksworld --model anthropic:claude-haiku-4-5 --complexity 2-12 --n 20

# aggregate results/ and verifier self-checks into docs/data/ for the site
python -m semantic.report
```

Tasks: `hanoi`, `blocksworld`, `blocksworld-mystery`, `multiplication`.
