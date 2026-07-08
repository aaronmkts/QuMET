# JOSS Submission Checklist

Last updated: 2026-07-08.

This checklist tracks QuMET against the current JOSS submission requirements and reviewer checklist:

- <https://joss.readthedocs.io/en/latest/submitting.html>
- <https://joss.readthedocs.io/en/latest/review_checklist.html>

Status keys:

- `[x]` Done in the `joss/integration` branch.
- `[ ]` Still required before submission.
- `[~]` Partly present, but needs more evidence, polish, or validation.

## Completed in `joss/integration`

- [x] Align package metadata with the repository license and current project version.
- [x] Keep CLI version output consistent with package metadata.
- [x] Expose a package console script through `pyproject.toml`.
- [x] Fix the Bars-and-Stripes helper syntax/import blocker caught by compilation.
- [x] Make source-checkout pytest runs import `qumet` without ad hoc `PYTHONPATH` handling.
- [x] Add install and CLI smoke tests for editable installs and version reporting.
- [x] Add a GitHub Actions CI workflow for install, compile, and pytest checks.
- [x] Rewrite README content around the research use case, target audience, install, quickstart, supported models, datasets, and tasks.
- [x] Add user-facing quickstart, configuration, model/dataset, and testing docs.
- [x] Add documentation tests that reject stale CLI flags and invalid quickstart/config examples.
- [x] Add `CONTRIBUTING.md`, `SUPPORT.md`, `CHANGELOG.md`, and `CITATION.cff`.
- [x] Add a JOSS paper scaffold in `paper/paper.md` with companion `paper/paper.bib`.
- [x] Archive the five implementation workstream reports under `docs/joss/workstream-reports/`.

## Submission Gates

- [x] Open source license is present and package metadata is Apache-2.0 aligned.
- [x] Code is in a Git repository with source, docs, tests, and paper files together.
- [x] Repository appears to support standard GitHub browsing, issues, and pull requests once the branch is merged.
- [~] Research application is clear in README and paper scaffold.
- [ ] Demonstrate research impact with concrete evidence: publications, preprints, benchmark artifacts, external adoption, or documented use in a research workflow.
- [ ] Confirm public development history is sufficient for JOSS's pre-review screen: more than six months public, active development over that period, and not a recent repo dump.
- [~] Open-source practice signals are present through tests, CI, docs, changelog, contribution, and support files.
- [ ] Add tagged release history and release notes before submission.
- [ ] Ensure public issues, PRs, and discussion history are visible enough to satisfy the open development gate.

## Installation and Packaging

- [x] Editable install metadata resolves for the current package.
- [x] Development extra exists for test tooling.
- [x] Console entry point is defined.
- [x] Installation smoke test covers package metadata and CLI availability.
- [ ] Test installation in a clean environment with dependencies resolved from scratch.
- [ ] Decide whether to publish to PyPI before or after JOSS review.
- [ ] Confirm supported Python versions against actual CI coverage.

## Functionality and Tests

- [x] Full local pytest suite passes on the integration branch.
- [x] Source package compiles with `compileall`.
- [x] Docs content tests cover user-facing command/config examples.
- [~] Model, dataset, and task compatibility is documented.
- [ ] Add minimal end-to-end smoke training tests for a representative QCBM workflow.
- [ ] Add minimal end-to-end smoke training tests for a representative QGAN or hybrid image-generation workflow.
- [ ] Add regression tests for config parsing, CLI overrides, metric selection, and output artifact creation.
- [ ] Add reviewer-friendly commands that finish quickly on CPU-only machines.
- [ ] Avoid unverified performance claims unless benchmark scripts and expected outputs are included.

## Documentation

- [x] README includes statement of need, target audience, install, quickstart, supported models, supported datasets, and supported tasks.
- [x] Documentation includes quickstart, configuration, model/dataset, and testing pages.
- [x] Community guidelines cover contribution, issue reporting, and support routes.
- [~] Core functionality is documented at a user level.
- [ ] Add API reference or equivalent developer-facing documentation for the main package modules.
- [ ] Add literature-to-implementation mapping for each QGAN/QCBM model claimed as supported.
- [ ] Add reproducible example outputs for at least one QCBM and one QGAN workflow.
- [ ] Ask a colleague to install and run the quickstart from a clean checkout, then record fixes.

## JOSS Paper

- [x] `paper/paper.md` and `paper/paper.bib` are present in the repository.
- [~] Paper scaffold has the expected JOSS shape.
- [ ] Complete `Summary` for a non-specialist scientific software audience.
- [ ] Complete `Statement of need` with problem, audience, and relation to other work.
- [ ] Complete `State of the field` comparing QuMET to related quantum ML/generative modelling packages.
- [ ] Complete `Software design` with concrete architecture and trade-off decisions.
- [ ] Complete `Research impact statement` with evidence rather than future-facing intent.
- [ ] Complete `AI usage disclosure` covering code, docs, tests, and paper assistance.
- [ ] Finalize authors, affiliations, acknowledgements, funding, and conflict-of-interest statements.
- [ ] Ensure every citation in the paper resolves in `paper/paper.bib`.

## Release and Submission

- [ ] Create a tagged release candidate after the branch is merged.
- [ ] Archive the release on Zenodo or another accepted archive and obtain a DOI when JOSS requests the final version.
- [ ] Update `CITATION.cff` and paper metadata with the release DOI/version when available.
- [ ] Fill out the JOSS submission form only after the gates above are satisfied.
- [ ] Be ready to respond to JOSS reviewer questions within the expected review windows.

## Five Next Most Impactful Changes

1. Finish the JOSS paper with strong `State of the field`, `Software design`, `Research impact statement`, and `AI usage disclosure` sections.
2. Add fast CPU-only end-to-end smoke workflows for one QCBM and one QGAN/hybrid model, including expected artifacts and reviewer commands.
3. Produce concrete research-impact evidence: benchmark outputs, a reproducible comparison notebook/script, links to papers/preprints, or documented internal/external usage.
4. Validate install and tests from a fresh clone in CI and in a colleague-style clean environment, then tighten the docs around any missing dependency or runtime assumptions.
5. Prepare release discipline: tagged release candidate, changelog entry, citation metadata, and a Zenodo-ready archive plan.
