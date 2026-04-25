SCALEDOWN_WINDOW = 60 # seconds
MODAL_TIMEOUT = 3600 # seconds
MIN_CONTAINERS = 0 # 0 to make sure we don't pay 24/7
COLLECTION_HANDLER_QDRANT_CLIENT_TIMEOUT = 600 # seconds

COLLECTION_HNSW_CONFIG = {
    # https://qdrant.tech/documentation/manage-data/indexing/
    # Number of edges per node in the index graph.
    # Larger the value - more accurate the search, more space required.
    "m": 16,
    # Number of neighbours to consider during the index building.
    # Larger the value - more accurate the search, more time required to build index.
    "ef_construct": 100,
    # Minimal size threshold (in KiloBytes) below which full-scan is preferred over HNSW search.
    # This measures the total size of vectors being queried against.
    # When the maximum estimated amount of points that a condition satisfies is smaller than
    # `full_scan_threshold_kb`, the query planner will use full-scan search instead of HNSW index
    # traversal for better performance.
    # Note: 1Kb = 1 vector of size 256
    "full_scan_threshold": 10000,
    # https://qdrant.tech/documentation/ops-configuration/configuration/
    # Store HNSW index on disk. If set to false, index will be stored in RAM. Default: false
    "on_disk": False,
}
