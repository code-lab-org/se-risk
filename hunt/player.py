#!/usr/bin/env python

from .game import Decision, Player

import numpy as np


class RandomPlayer(Player):
    """
    A player who always makes random decisions.

    Attributes:
        name (str): Name of this player class.
        game (:obj:`hunt.Game`): Game to be played by this player.
        rng (:obj:`numpy.random.Generator`): Random number generator.
    """

    def __init__(self, game, name="random", seed=None):
        super().__init__(game, name)
        # initialize a random number stream
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def get_decision(self):
        """
        Get a random decision (strategy: 0 or 1, design: random possible).

        Returns:
            :obj:`hunt.game.Decision`: The selected decision for the next game.
        """
        num_designs = len(self.game.designs)
        return Decision(
            strategy=self.rng.integers(0, 1, endpoint=True),
            design=self.rng.integers(0, num_designs) if num_designs > 1 else 0,
        )


class MirrorPlayer(Player):
    """
    A player who always mirrors the opponent's previous decision.

    Attributes:
        name (str): Name of this player.
        game (:obj:`hunt.game.Game`): Game to be played by this player.
        rng (:obj:`numpy.random.Generator`): Random number generator.
    """

    def __init__(self, game, name="mirror", seed=None):
        super().__init__(game, name)
        # initialize a random number stream
        self.rng = np.random.Generator(np.random.PCG64(seed))
        # random initial decision
        num_designs = len(self.game.designs)
        self.next_decision = Decision(
            strategy=self.rng.integers(0, 1, endpoint=True),
            design=self.rng.integers(0, num_designs) if num_designs > 1 else 0,
        )

    def get_decision(self):
        """
        Return the opponent's previous decision.

        Returns:
            :obj:`hunt.game.Decision`: The selected decision for the next game.
        """
        return self.next_decision

    def report_result(self, result):
        """
        Function called by the tournament to report the result of a game.

        Args:
            result (:obj:`hunt.game.Result`): The result of the most recent game.
        """
        # set the next decision
        self.next_decision = result.their_decision
