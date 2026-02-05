from .discrete_gen import QGANDiscreteGenModelWrapper
from .generation import QGANGenerationModelWrapper
from .image_generation import (
    APQGANWrapper,
    GANWrapper,
    MosaiQGANWrapper,
    PatchGANWrapper,
    PQWGANWrapper,
    ProbsQGANWrapper,
    QINRWrapper,
)
from .probs_gen import QGANProbsGenModelWrapper


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
