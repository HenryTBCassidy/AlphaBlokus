"""Local play server: the download tier of the web-play plan.

``PlayService`` (service.py) answers engine-interface requests with the real
torch + MCTS stack; ``server.py`` wraps it in a FastAPI app that also serves
the built frontend. Requires the ``play`` extra (``uv sync --extra play``)
for the HTTP layer; the service itself has no extra dependencies.
"""
