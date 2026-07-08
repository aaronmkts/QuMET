# QGAN and GAN Models

This page describes the QGAN-family models that are production-facing for the first QuMET public release. The goal is to make the supported surface clear for users and reviewers: every model listed under production-facing models must be registered, constructible, documented, and covered by smoke tests.

## Production-Facing Models

### `patchgan`

`patchgan` is a hybrid quantum-classical patch GAN based on the PennyLane quantum GAN tutorial pattern. It uses a classical discriminator and a patch-based quantum generator. Each sub-generator returns a probability patch that is concatenated into an image-shaped output.

- Registry type: `qgan`
- Main class: `PatchGAN`
- Constructor: `get_patchgan`
- Primary task: `image_generation`
- Current smoke coverage: registry lookup and constructor instantiation

### `mosaiq`

`mosaiq` follows the MosaiQ design for quantum image generation on NISQ computers. The reference paper describes a hybrid quantum-classical GAN that learns compressed image representations using PCA, distributes 40 PCA features across eight five-qubit quantum sub-generators, and uses a classical discriminator over the generated PCA feature vectors. The upstream reference implementation uses `RY` and `RX` input embedding, trainable `RY` layers, `CZ` entanglers, and Pauli-X expectation outputs.

The QuMET constructor preserves those core structural choices: 40 PCA dimensions, eight sub-generators, five qubits per sub-generator, depth six circuits, `RY`/`RX` input embedding, `CZ` entanglement, Pauli-X expectations, and the published feature redistribution ordering.

- Registry type: `qgan`
- Main class: `MosaiQ`
- Constructor: `get_mosaiq`
- Primary task: `image_generation`
- Reference paper: <https://arxiv.org/abs/2308.11096>
- Reference code: <https://github.com/SilverEngineered/MosaiQ/blob/main/mosaiq.py>
- Current smoke coverage: registry lookup and constructor instantiation

### `pqwgan_qc`

`pqwgan_qc` is a partially quantum GAN for image generation. It uses a quantum patch generator to produce image patches and a classical discriminator over image tensors. The generator uses parameterized quantum layers through PennyLane/Torch integration and reshapes the generated patches into image outputs.

- Registry type: `qgan`
- Main class: `PQWGAN_QC`
- Constructor: `get_pqwgan_qc`
- Primary task: `image_generation`
- Current smoke coverage: registry lookup and constructor instantiation

### `qinr`

`qinr` is a quantum implicit neural representation style GAN. It combines classical linear layers with quantum layers inside the generator, then maps generated features back to image tensors. The discriminator is a classical image discriminator.

- Registry type: `qgan`
- Main class: `QINR`
- Constructor: `get_qinr_qc`
- Primary task: `image_generation`
- Current smoke coverage: registry lookup and constructor instantiation

### `gan`

`gan` is a classical GAN baseline used for comparison against QGAN-family models. It shares the QGAN training and wrapper path but is registered with model type `gan`.

- Registry type: `gan`
- Main class: `GAN`
- Constructor: `get_gan`
- Primary task: `image_generation`
- Current smoke coverage: registry lookup and constructor instantiation

## Deferred Models

`vaeqwgan` is intentionally deferred for the first production-readiness pass. The registry still exposes it, but the implementation currently uses the class name `APQGAN` and overlaps with older `apqgan` configuration references. That naming and support boundary should be resolved in a separate branch before it is documented as production-ready.

Older or incomplete QGAN-related modules such as `qgcd`, `copula`, and stale `apqgan` configs are not production-facing in this release pass unless they are explicitly registered, constructible, tested, and documented.

## Testing Expectations

Production-facing QGAN-family models must satisfy these minimum checks before release:

- the model name is present in `QGAN_MODELS`;
- `get_model(name, "image_generation", get_dataset_info("mnist"))` constructs successfully;
- public model modules/classes/constructors have Google-style docstrings;
- the model is described on this page with its registry name, main class, constructor, task, and current support boundary;
- model-specific literature or upstream implementation claims are cited when the implementation is adapted from published or external code.
