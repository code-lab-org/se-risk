import numpy as np
import scipy.stats as stats

from hunt.game import Player, Decision


class RiskAwareRDplayer(Player):
    def __init__(self, game, name, risk_aversion):
        super().__init__(game, name)
        self.risk_aversion = risk_aversion
        self.their_prior_decision = None

    def get_utility(self, value):
        if self.risk_aversion == 0:
            return value
        else:
            return (1 - np.exp(-self.risk_aversion * value)) / self.risk_aversion

    def report_result(self, result):
        self.their_prior_decision = result.their_decision

    def get_risk_dominance(self, design, independent_design):
        if False:
            # old method assuming complete symmetry
            return np.log(
                (
                    self.get_utility(
                        self.game.get_payoff(
                            Decision(0, independent_design), Decision(0)
                        )
                    )
                    - self.get_utility(
                        self.game.get_payoff(Decision(1, design), Decision(0))
                    )
                )
                / (
                    self.get_utility(
                        self.game.get_payoff(Decision(1, design), Decision(1))
                    )
                    - self.get_utility(
                        self.game.get_payoff(
                            Decision(0, independent_design), Decision(1)
                        )
                    )
                )
            )
        else:
            # new method assuming partner's design is same as prior round
            return 0.5 * np.log(
                (
                    self.get_utility(
                        self.game.get_payoff(
                            Decision(0, independent_design), Decision(0)
                        )
                    )
                    - self.get_utility(
                        self.game.get_payoff(Decision(1, design), Decision(0))
                    )
                )
                / (
                    self.get_utility(
                        self.game.get_payoff(Decision(1, design), Decision(1))
                    )
                    - self.get_utility(
                        self.game.get_payoff(
                            Decision(0, independent_design), Decision(1)
                        )
                    )
                )
            ) + 0.5 * np.log(
                (
                    self.get_utility(
                        self.game.get_payoff(
                            Decision(0, independent_design), Decision(0)
                        )
                    )
                    - self.get_utility(
                        self.game.get_payoff(
                            Decision(
                                1,
                                self.their_prior_decision.design
                                if self.their_prior_decision is not None
                                else design,
                            ),
                            Decision(0),
                        )
                    )
                )
                / (
                    self.get_utility(
                        self.game.get_payoff(
                            Decision(
                                1,
                                self.their_prior_decision.design
                                if self.their_prior_decision is not None
                                else design,
                            ),
                            Decision(1),
                        )
                    )
                    - self.get_utility(
                        self.game.get_payoff(
                            Decision(0, independent_design), Decision(1)
                        )
                    )
                )
            )

    def get_decision(self):
        with np.errstate(invalid="ignore"):
            independent_value = [
                self.get_utility(self.game.get_payoff(Decision(0, design), Decision(0)))
                for design in range(len(self.game.designs))
            ]
            independent_design = np.nanargmax(independent_value)

            risk_dominance = np.array(
                [
                    self.get_risk_dominance(design, independent_design)
                    for design in range(len(self.game.designs))
                ]
            )

            collaborative_value = [
                self.get_utility(self.game.get_payoff(Decision(1, design), Decision(1)))
                for design in range(len(self.game.designs))
            ]
            collaborative_design = np.argmax(collaborative_value * (risk_dominance < 0))
            if np.nanmin(risk_dominance) < 0:
                return Decision(1, collaborative_design)
            else:
                return Decision(0, independent_design)


class RiskAwareRDplayer_averse1(RiskAwareRDplayer):
    def __init__(self, game):
        super().__init__(game, "RD_averse1", 0.1)


class RiskAwareRDplayer_averse2(RiskAwareRDplayer):
    def __init__(self, game):
        super().__init__(game, "RD_averse2", 0.2)


class RiskAwareRDplayer_seeker1(RiskAwareRDplayer):
    def __init__(self, game):
        super().__init__(game, "RD_seeker1", -0.03)


class RiskAwareRDplayer_seeker2(RiskAwareRDplayer):
    def __init__(self, game):
        super().__init__(game, "RD_seeker2", -0.065)


class RiskAwareRDplayer_neutral(RiskAwareRDplayer):
    def __init__(self, game):
        super().__init__(game, "RD_neutral", 0)


class RiskAwareEUplayer(Player):
    def __init__(self, game, name, risk_aversion, prior_collab=1, prior_no_collab=1):
        super().__init__(game, name)
        self.strategy_prior = np.array([prior_no_collab, prior_collab])
        self.risk_aversion = risk_aversion

    def get_utility(self, value):
        if self.risk_aversion == 0:
            return value
        else:
            return (1 - np.exp(-self.risk_aversion * value)) / self.risk_aversion

    def report_result(self, result):
        self.strategy_prior[result.their_decision.strategy] += 1

    def get_decision(self):
        p_collab = stats.beta(self.strategy_prior[1], self.strategy_prior[0]).mean()

        independent_expected_value = np.array(
            [
                self.get_utility(self.game.get_payoff(Decision(0, design), Decision(1)))
                * p_collab
                + self.get_utility(
                    self.game.get_payoff(Decision(0, design), Decision(0))
                )
                * (1 - p_collab)
                for design in range(len(self.game.designs))
            ]
        )
        independent_design = np.nanargmax(independent_expected_value)

        collab_expected_value = np.array(
            [
                self.get_utility(self.game.get_payoff(Decision(1, design), Decision(1)))
                * p_collab
                + self.get_utility(
                    self.game.get_payoff(Decision(1, design), Decision(0))
                )
                * (1 - p_collab)
                for design in range(len(self.game.designs))
            ]
        )
        collaborative_design = np.argmax(collab_expected_value)

        if np.all(
            collab_expected_value < independent_expected_value[independent_design]
        ):
            return Decision(0, independent_design)
        else:
            return Decision(1, collaborative_design)


class RiskAwareEUplayer_neutral(RiskAwareEUplayer):
    def __init__(self, game):
        super().__init__(game, "EU_neutral", 0)


class RiskAwareEUplayer_averse1(RiskAwareEUplayer):
    def __init__(self, game):
        super().__init__(game, "EU_averse1", 0.2)


class RiskAwareEUplayer_averse2(RiskAwareEUplayer):
    def __init__(self, game):
        super().__init__(game, "EU_averse2", 0.1)


class RiskAwareEUplayer_seeker1(RiskAwareEUplayer):
    def __init__(self, game):
        super().__init__(game, "EU_seeker1", -0.2)


class RiskAwareEUplayer_seeker2(RiskAwareEUplayer):
    def __init__(self, game):
        super().__init__(game, "EU_seeker2", -0.1)
