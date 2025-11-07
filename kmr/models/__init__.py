"""Models module for Keras Model Registry."""

from kmr.models.SFNEBlock import SFNEBlock
from kmr.models.TerminatorModel import TerminatorModel
from kmr.models.feed_forward import BaseFeedForwardModel
from kmr.models.autoencoder import Autoencoder
from kmr.models.TimeMixer import TimeMixer
from kmr.models.TSMixer import TSMixer
from kmr.models.GeospatialClusteringModel import GeospatialClusteringModel
from kmr.models.MatrixFactorizationModel import MatrixFactorizationModel
from kmr.models.TwoTowerModel import TwoTowerModel
from kmr.models.DeepRankingModel import DeepRankingModel
from kmr.models.ExplainableRecommendationModel import ExplainableRecommendationModel
from kmr.models.UnifiedRecommendationModel import UnifiedRecommendationModel
from kmr.models.ExplainableUnifiedRecommendationModel import (
    ExplainableUnifiedRecommendationModel,
)

__all__ = [
    "SFNEBlock",
    "TerminatorModel",
    "BaseFeedForwardModel",
    "Autoencoder",
    "TimeMixer",
    "TSMixer",
    "GeospatialClusteringModel",
    "MatrixFactorizationModel",
    "TwoTowerModel",
    "DeepRankingModel",
    "ExplainableRecommendationModel",
    "UnifiedRecommendationModel",
    "ExplainableUnifiedRecommendationModel",
]
