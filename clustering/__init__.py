"""

Clustering module that performs cluster cleaning and uses new clustering algorithms, such as 
spectral clustering and fuzzy c-means.

"""

__version__ = "0.1.0"

from .cluster_alts import spectral, fkmeans, dbscan, gaussianmix

__all__ = [
    "spectral", 
    "fkmeans", 
    "dbscan", 
    "gaussianmix"
]