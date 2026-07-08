# QuMET Models and Datasets

This page summarizes the model, dataset, and task names that are currently wired into the repository.

## Supported Models

The current model registry exposes:

- `qcbm`
- `patchgan`
- `mosaiq`
- `vaeqwgan`
- `pqwgan_qc`
- `qinr`
- `gan`
- `vae`

At a high level:

- `qcbm` is the discrete-generation QCBM entry point.
- `patchgan`, `mosaiq`, `vaeqwgan`, `pqwgan_qc`, `qinr`, and `gan` are handled through the QGAN or GAN model path.
- `vae` is available for image-generation workflows.

## Supported Datasets

Vision datasets:

- `mnist`
- `fashion_mnist`
- `cifar10`

Manual continuous datasets:

- `2d_gaussian`
- `2d_grid_gaussian`
- `2d_ring_gaussian`

Manual bitstring datasets:

- `bars_and_stripes`
- `2d_gaussian_b`
- `2d_grid_gaussian_b`
- `2d_ring_gaussian_b`

## Supported Tasks

The CLI currently advertises three task names:

- `discrete_generation`
- `continuous_generation`
- `image_generation`

## Practical Pairings

Examples already present in `src/configs/` give the most reliable starting points:

- `qcbm` with `2d_gaussian` for `discrete_generation`
- `qcbm` with `bars_and_stripes` for `discrete_generation`
- `pqwgan_qc` with `mnist` for `image_generation`

Some older config files still mention names that are not part of the current model registry. For JOSS-facing workflows, prefer the model and dataset names listed on this page and validated by the code under `src/qumet/models/` and `src/qumet/dataset/`.

