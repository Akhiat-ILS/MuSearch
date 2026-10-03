"""Compatibility entry point; implementation lives in musearch.attack."""

from musearch.attack import AttackEnvironment, Controller, Node, run_episode, search, train

__all__ = ["AttackEnvironment", "Controller", "Node", "run_episode", "search", "train"]
