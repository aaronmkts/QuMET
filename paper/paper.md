# Summary

QuMET is a Python toolkit for developing and evaluating quantum generative modelling workflows. The repository combines a command-line interface, dataset abstractions, model registries, training actions, and analysis utilities so that experiments can be configured and reproduced from a common software surface. QuMET is built on top of PennyLane for differentiable quantum circuit execution and PyTorch and Lightning for model definition, optimization, and training orchestration [@bergholm2018pennylane; @paszke2019pytorch].

# Statement of need

Research code for quantum machine learning often grows around one model family, one dataset, or one experiment script. That makes it harder to compare approaches, rerun experiments with consistent interfaces, and preserve working training pipelines as projects evolve. QuMET addresses that gap by providing a shared toolkit for quantum generative modelling experiments with reusable dataset loaders, model lookup utilities, CLI-driven training flows, checkpoint handling, and plotting wrappers.

The current repository is especially oriented toward quantum generative modelling use cases represented in the codebase, including quantum generative adversarial network variants, quantum circuit Born machine workflows, and supporting classical baselines. A common toolkit reduces the amount of per-project glue code needed to move between toy-distribution experiments and image-generation pipelines.

# State of the field

Hybrid quantum-classical software stacks such as PennyLane make it practical to integrate variational quantum circuits into familiar Python machine-learning workflows [@bergholm2018pennylane]. PyTorch remains a standard foundation for tensor programs and automatic differentiation in research software [@paszke2019pytorch]. Within quantum generative modelling, representative directions include adversarial training with parameterized quantum circuits [@dallaire2018qgan] and quantum circuit Born machines trained as implicit generative models [@liu2018qcbm].

QuMET sits at the layer above those frameworks and model papers. Rather than introducing a new learning algorithm, it packages experiment-facing software needed to configure datasets, select supported models, launch training, resume checkpoints, and inspect outputs with a consistent interface.

# Software design

QuMET is organized as a Python package under `src/qumet`. The CLI entrypoint coordinates configuration loading, argument validation, model selection, and dataset initialization. Model registries separate supported model families from task-specific constructors, while dataset modules provide unified access to manual benchmark datasets and vision datasets. Training and validation actions use Lightning-based orchestration, and the plotting wrappers expose downstream evaluation and visualization helpers.

This structure is intended to keep experiment code close to the reusable software boundary: configuration files describe runs, package modules provide stable interfaces, and tests cover high-risk public surfaces such as CLI behavior, registry mappings, dataset utilities, and checkpoint loading.

# Research impact statement

QuMET's research contribution is infrastructural. The package is intended to reduce software friction in quantum generative modelling projects by giving researchers a shared environment for implementing, comparing, and maintaining experiment workflows. In practice, that means less duplication across one-off scripts and a clearer path from exploratory model code to documented, testable package behavior.

This manuscript scaffold does not claim a benchmark or algorithmic advance on its own. Instead, it documents the software foundation used to support reproducible experimentation in the repository's current model families and analysis pipelines.

# AI usage disclosure

This manuscript scaffold and repository submission metadata were drafted with assistance from OpenAI Codex/GPT-5 and then reviewed and edited in-repository by the maintainer workflow for accuracy, scope, and citation conservatism.

# Acknowledgements

QuMET builds on the open-source ecosystems around PennyLane, PyTorch, and Lightning, and benefits from the broader quantum machine learning and scientific Python communities.

# References
