from .discrete_gen import QGANDiscreteGenModelWrapper
from .probs_gen import QGANProbsGenModelWrapper
from .image_generation import PatchGANWrapper, MosaiQGANWrapper, APQGANWrapper, PQWGANWrapper, ProbsQGANWrapper, QINRWrapper, GANWrapper
from .generation import QGANGenerationModelWrapper

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)