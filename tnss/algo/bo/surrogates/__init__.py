from tnss.algo.bo.surrogates.base import Surrogate
from tnss.algo.bo.surrogates.classification_gp import ClassificationGP, feasibility_prob
from tnss.algo.bo.surrogates.regression_gp import RegressionGP, objective_target

__all__ = [
    "Surrogate",
    "RegressionGP",
    "objective_target",
    "ClassificationGP",
    "feasibility_prob",
]
