class Game:
    rewards: list[list]
    name: str
    action_names: list[str]


class PrisonersDilemma(Game):
    rewards = [
        [(3, 3), (0, 5)],
        [(5, 0), (1, 1)],
    ]
    action_names = ["Cooperate", "Defect"]
    name = "Prisoner's Dilemma"


class StagHunt(Game):
    rewards = [
        [(1, 1), (0, 2 / 3)],
        [(2 / 3, 0), (2 / 3, 2 / 3)],
    ]
    action_names = ["Stag", "Hare"]
    name = "Stag Hunt"


class MatchingPennies(Game):
    rewards = [
        [(0, 1), (1, 0)],
        [(1, 0), (0, 1)],
    ]
    action_names = ["Heads", "Tails"]
    name = "Matching Pennies"


class SubsidyGame(Game):
    rewards = [
        [(12, 12), (0, 11)],
        [(11, 0), (10, 10)],
    ]
    action_names = ["S1", "S2"]
    name = "Subsidy Game"
