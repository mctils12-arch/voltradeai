"""AlphaDesk — explainable equity & options research engine (clean-room)."""
from .engine import analyze, DEFAULT_WEIGHTS
from .tax import TaxProfile
from .providers import SampleProvider, LiveProvider
from .models import ResearchReport

__all__ = ["analyze", "DEFAULT_WEIGHTS", "TaxProfile",
           "SampleProvider", "LiveProvider", "ResearchReport"]
__version__ = "0.1.0"
