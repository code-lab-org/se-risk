#!/usr/bin/env python

import itertools
import numpy as np
import copy

from .game import DesignGame, InvalidDecisionError


class Tournament(object):
    """
    Defines a tournament with a list of players, a list of games, and the number
    of replications of each game for each pair of players. Performs a round-
    robin tournament including mirror matches. Returns a dictionary of final
    scores for each player.

    Attributes:
        players (:obj:`list` of :obj:`hunt.game.Player`): List of players.
        games (:obj:`list` of :obj:`hunt.game.Game`): List of games.
        num_reps (int): Number of replications for each game.
        p_rep (float): Probability of ending a game after each replication.
        rng (:obj:`numpy.random.Generator`): Random number generator.
        error_results (:obj:`list` of :obj:`dict`): List of any erroneous results.
        rep_results (:obj:`list` of :obj:`dict`): List of results for each replication.
        game_results (:obj:`list` of :obj:`dict`): List of results for each game.
        match_results (:obj:`list` of :obj:`dict`): List of results for each match.
        results (:obj:`dict`): Overall results.
    """

    def __init__(self, players=[], games=[], num_reps=None, p_rep=None, p_seed=None):
        self.players = players
        self.games = games
        self.num_reps = num_reps
        self.p_rep = p_rep
        self.rng = np.random.Generator(np.random.PCG64(p_seed))
        self.error_results = []
        self.rep_results = []
        self.game_results = []
        self.match_results = []
        self.results = {}

    def run(self):
        """
        Runs the tournament.

        Returns:
            :obj:`dict`: Total score for each player.
        """
        # initialize results
        self.error_results = []
        self.rep_results = []
        self.game_results = []
        self.match_results = []
        self.results = {}
        # generate list of matches (round-robin tournament)
        matches = list(itertools.combinations_with_replacement(self.players, 2))
        # run each match
        for match in matches:
            # initialize match scores
            p1_match_score = 0
            p2_match_score = 0
            # enumerate each game in the match
            for g, game in enumerate(self.games):
                # initialize game scores
                p1_game_score = 0
                p2_game_score = 0
                # initialize player objects for this game
                player_1 = match[0](DesignGame(copy.deepcopy(game.designs)))
                player_2 = match[1](DesignGame(copy.deepcopy(game.designs)))
                if self.num_reps is None:
                    num_reps = self.rng.geometric(self.p_rep)
                else:
                    num_reps = self.num_reps
                # generate the number of game replicates
                for rep in range(num_reps):
                    # play the game and record results
                    try:
                        results = game.play(player_1, player_2)
                        # log results of this replicate
                        self.rep_results.append(
                            {
                                "game": g,
                                "rep": rep,
                                "player_1": {
                                    "name": player_1.name,
                                    "design": results.my_decision.design,
                                    "strategy": results.my_decision.strategy,
                                    "payoff": results.my_payoff,
                                },
                                "player_2": {
                                    "name": player_2.name,
                                    "design": results.their_decision.design,
                                    "strategy": results.their_decision.strategy,
                                    "payoff": results.their_payoff,
                                },
                            }
                        )
                        # append results of this replicate to the game results
                        p1_game_score += results.my_payoff / num_reps
                        p2_game_score += results.their_payoff / num_reps
                    except InvalidDecisionError as e:
                        self.error_results.append(
                            {
                                "game": g,
                                "rep": rep,
                                "player": e.player.name,
                                "design": e.decision.design,
                                "strategy": e.decision.strategy,
                                "error": e,
                            }
                        )
                    except Exception as e:
                        self.error_results.append(
                            {
                                "game": g,
                                "rep": rep,
                                "player_1": player_1.name,
                                "player_2": player_2.name,
                                "error": e,
                            }
                        )
                # log results of this game
                self.game_results.append(
                    {
                        "game": g,
                        "reps": num_reps,
                        "players": [
                            {"name": player_1.name, "score": p1_game_score},
                            {"name": player_2.name, "score": p2_game_score},
                        ],
                    }
                )
                # append results of this game to the match results
                p1_match_score += p1_game_score / len(self.games)
                p2_match_score += p2_game_score / len(self.games)
            # log results of this match
            self.match_results.append(
                {
                    "players": [
                        {"name": player_1.name, "score": p1_match_score},
                        {"name": player_2.name, "score": p2_match_score},
                    ]
                }
            )
            # append results of this match to the overall results
            self.results[player_1.name] = self.results.get(
                player_1.name, 0
            ) + p1_match_score * (0.5 if player_1.name == player_2.name else 1) / len(
                self.players
            )
            self.results[player_2.name] = self.results.get(
                player_2.name, 0
            ) + p2_match_score * (0.5 if player_1.name == player_2.name else 1) / len(
                self.players
            )
        return self.results
