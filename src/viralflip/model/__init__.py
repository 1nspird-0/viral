"""Model components for ViralFlip and ViralFlip-X."""

from viralflip.model.drift_score import DriftScoreModule
from viralflip.model.lag_lattice import LagLatticeHazardModel
from viralflip.model.interactions import InteractionModule
from viralflip.model.personalization import PersonalizationLayer
from viralflip.model.viralflip_model import ViralFlipModel, ViralFlipOutput, VirusClassifier
from viralflip.model.viralflip_x import ViralFlipX, ViralFlipXOutput, EncoderBackedDriftScore
from viralflip.model.virus_types import (
    VirusType, NUM_VIRUS_CLASSES, VIRUS_NAMES, VIRUS_SHORT_NAMES,
    get_virus_name, get_virus_from_name, list_virus_types,
)

__all__ = [
    # Core components
    "DriftScoreModule",
    "LagLatticeHazardModel",
    "InteractionModule",
    "PersonalizationLayer",
    # Models
    "ViralFlipModel",
    "ViralFlipOutput",
    "ViralFlipX",
    "ViralFlipXOutput",
    # Encoder-backed
    "EncoderBackedDriftScore",
    # Virus classification
    "VirusClassifier",
    "VirusType",
    "NUM_VIRUS_CLASSES",
    "VIRUS_NAMES",
    "VIRUS_SHORT_NAMES",
    "get_virus_name",
    "get_virus_from_name",
    "list_virus_types",
]

