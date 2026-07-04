"""Self-play generation: the episode loop and the backend dispatcher."""

from alphablokus.selfplay.episode import GameExamples, ProcessedExample, play_self_play_episode
from alphablokus.selfplay.generate import generate_games

__all__ = ["GameExamples", "ProcessedExample", "generate_games", "play_self_play_episode"]
