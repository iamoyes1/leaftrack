"""REST API (stub, Phase 5).

Will expose the pipeline over HTTP with FastAPI:

- ``POST /segment``: one image in, leaf instances (polygons, areas) out as JSON.
- ``POST /track``: an ordered image sequence in, tracked leaves and growth
  analytics out.
- ``GET /health``: liveness check including the loaded model version.
"""
