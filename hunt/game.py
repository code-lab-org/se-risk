#!/usr/bin/env python


class Decision(object):
    """
    Composition of an upper-level strategy and lower-level design decision.

    Attributes:
        strategy (int): Selected strategy (zero-based index).
        design (int): Selected design (zero-based index).
    """

    def __init__(self, strategy=0, design=0):
        self.strategy = strategy
        self.design = design


class Result(object):
    """
    Composition of players' decisions and resulting payoffs.

    Attributes:
        my_decision (:obj:`Decision`): Selected decision for the focal player.
        my_payoff (float): Resulting payoff for the focal player.
        their_decision (:obj:`Decision`): Selected decision for the other player.
        their_payoff (float): Resulting payoff for the other player.
    """

    def __init__(self, my_decision, my_payoff, their_decision, their_payoff):
        self.my_decision = my_decision
        self.my_payoff = my_payoff
        self.their_decision = their_decision
        self.their_payoff = their_payoff


class Player(object):
    """
    A basic player who always plays strategy 0.

    Attributes:
        game (:obj:`Game`): Game to be played by this player.
        name (str): Name of this player class.
    """

    def __init__(self, game, name="null"):
        self.name = name
        self.game = game

    def get_decision(self):
        """
        Function called by the tournament to get a player's next decision.

        Returns:
            :obj:`Decision`: The selected decision for the next game.
        """
        return Decision()

    def report_result(self, result):
        """
        Function called by the tournament to report the result of a game.

        Args:
            result (:obj:`Result`): The result of the most recent game.
        """
        pass


class InvalidDecisionError(Exception):
    """
    Raised when a player supplies an invalid decision.

    Attributes:
        player (:obj:`Player`): The player making an invalid decision.
        decision (:obj:`Decision`): The invalid decision.
    """

    def __init__(self, player, decision):
        self.player = player
        self.decision = decision


class DesignGame(object):
    """
    A symmetric game for two players that defines payoffs for a
    lower-level design and upper-level strategy decisions. Payoffs do not
    depend on the opponent's design decision.

    Attributes:
        designs (:obj:`list` of :obj:`list` of :obj:`list` of float): List of
            alternative lower-level designs. Each lower-level design is structured
            as a two-dimensional matrix of payoff values, indexed by player i,j
            strategy decisions.
    """

    def __init__(self, designs=[[[0, 0], [0, 0]]]):
        self.designs = designs

    def get_payoff(self, my_decision, their_decision):
        """
        Gets the payoff for a pair of decisions.

        Args:
            my_decision (:obj:`Decision`): The focal player's decision.
            their_decision (:obj:`Decision`): The other player's decision.

        Returns:
            float: The focal player's resulting payoff.
        """
        return self.designs[my_decision.design][my_decision.strategy][
            their_decision.strategy
        ]

    def play(self, player_1, player_2):
        """
        Players one iteration of a game. Gets decisions from both players,
        determines their payoffs, and results results to each player.

        Args:
            player_1 (:obj:`Player`): The first player.
            player_2 (:obj:`Player`): The second player.
        """
        decision_1 = player_1.get_decision()
        if decision_1.strategy not in [0, 1] or decision_1.design not in range(
            len(self.designs)
        ):
            raise InvalidDecisionError(player_1, decision_1)
        decision_2 = player_2.get_decision()
        if decision_2.strategy not in [0, 1] or decision_2.design not in range(
            len(self.designs)
        ):
            raise InvalidDecisionError(player_2, decision_2)
        payoff_1 = self.get_payoff(decision_1, decision_2)
        payoff_2 = self.get_payoff(decision_2, decision_1)
        player_1.report_result(Result(decision_1, payoff_1, decision_2, payoff_2))
        player_2.report_result(Result(decision_2, payoff_2, decision_1, payoff_1))
        return Result(decision_1, payoff_1, decision_2, payoff_2)
