# Ensures app.ml.data is recognized as a package for imports like
# from app.ml.data.data_extractor import data_extractor

from .data_extractor import data_extractor  # re-export for convenience
