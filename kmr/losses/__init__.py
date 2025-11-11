"""Losses for recommendation systems."""

from kmr.losses.max_min_margin_loss import MaxMinMarginLoss
from kmr.losses.average_margin_loss import AverageMarginLoss
from kmr.losses.improved_margin_ranking_loss import ImprovedMarginRankingLoss
from kmr.losses.geospatial_margin_loss import GeospatialMarginLoss

__all__ = [
    "MaxMinMarginLoss",
    "AverageMarginLoss",
    "ImprovedMarginRankingLoss",
    "GeospatialMarginLoss",
]
