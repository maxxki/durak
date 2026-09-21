"""Tests für ismcts_agent.py (Stufe 2a).

Die fünf geforderten Tests sind als ``test_01_...`` bis ``test_05_...``
nummeriert. ``test_05`` (Gewinnrate gegen Zufalls-Gegner) ist als
``@pytest.mark.slow`` markiert.
"""

from __future__ import annotations

import random
import time

import pytest

from durak_env import (
    FULL_DECK,
    State,
    initial_state,
    legal_actions,
    step,
    winner,
)
from ismcts_agent import ISMCTSAgent
from test_durak_env import c, cs, make_state  # bewährte Test-Helfer wiederverwenden


# --------------------------------------------------------------------------
# Hilfen
# --------------------------------------------------------------------------


def _actor(state: State) -> int:
    """Index des gerade handelnden Spielers."""
    return state.attacker_idx if state.phase == "ATTACK" else 1 - state.attacker_idx


def _random_policy(state: State, rng: random.Random):
    legal = legal_actions(state)
    return legal[rng.randrange(len(legal))]


def _play_game(seed: int, agent_idx: int, time_budget: float, max_plies: int = 2000):
    """Spielt eine volle Partie: Agent gegen Zufalls-Gegner. Liefert (winner, plies)."""
    state = initial_state(seed=seed)
    agent = ISMCTSAgent(player_idx=agent_idx, time_budget=time_budget, seed=seed)
    opp_rng = random.Random(10_000 + seed)
    plies = 0
    while True:
        legal = legal_actions(state)
        if not legal:
            break
        if _actor(state) == agent_idx:
            action = agent.choose_action(state)
        else:
            action = _random_policy(state, opp_rng)
        state, _reward, done = step(state, action)
        plies += 1
        if done:
            break
        assert plies <= max_plies, "Partie terminiert nicht (Endlosschleife?)."
    return winner(state), plies


# --------------------------------------------------------------------------
# 1. legal_actions des Agenten sind stets Teilmenge der Umgebungs-Aktionen
# --------------------------------------------------------------------------


def test_01_agent_actions_are_always_legal():
    for seed in range(20):
        state = initial_state(seed=seed)
        agent = ISMCTSAgent(player_idx=0, time_budget=0.02, seed=seed)
        opp_rng = random.Random(5_000 + seed)
        plies = 0
        while True:
            legal = legal_actions(state)
            if not legal:
                break
            if _actor(state) == 0:
                action = agent.choose_action(state)
                assert action in legal, (
                    f"seed={seed}: Agent-Aktion {action!r} ist nicht in "
                    f"legal_actions {legal!r}."
                )
            else:
                action = _random_policy(state, opp_rng)
            state, _reward, done = step(state, action)
            plies += 1
            if done:
                break
            assert plies <= 2000


# --------------------------------------------------------------------------
# 2. Determinisierung respektiert unknown_pool und Constraint-Check
# --------------------------------------------------------------------------


def test_02_determinization_respects_unknown_pool():
    agent = ISMCTSAgent(player_idx=0, time_budget=0.05, seed=3)

    # (a) Frischer Startzustand: keine Constraints, nur Größen-/Disjunktheits-Check.
    for seed in range(10):
        state = initial_state(seed=seed)
        det = agent.determinize(state)

        own_hand = det.hand_attacker  # player_idx=0 ist zu Beginn immer Angreifer
        opp_hand = det.hand_defender
        all_cards = list(own_hand) + list(opp_hand) + list(det.deck)
        if det.trump_card is not None:
            all_cards.append(det.trump_card)
        assert len(all_cards) == len(set(all_cards)), "Karte(n) doppelt vergeben."
        assert set(all_cards) <= set(FULL_DECK)
        assert len(det.hand_defender) == len(state.hand_defender)
        assert len(det.deck) == len(state.deck)
        # Eigene Hand bleibt unverändert (Agent kennt sie exakt).
        assert det.hand_attacker == state.hand_attacker

    # (b) Konstruierte Situation: Gegner hat zuvor aufgenommen (TAKE) und hält
    # damit nachweislich zwei bestimmte, öffentlich gesehene Karten. Diese
    # dürfen NIE Teil des Sampling-Pools sein bzw. müssen als bekannt in
    # seiner Hand landen (kein Widerspruch zum Constraint-Check).
    known_taken = cs("7S 8S")  # diese Karten hat der Verteidiger (Spieler 1) aufgenommen
    state = make_state(
        hand_attacker=cs("6H 9C QD KD AC 10H"),  # Spieler 0 (Agent), own_hand
        hand_defender=tuple([known_taken[0], known_taken[1]]) + cs("6C 7C 8C 9H"),
        table_attack=(),
        table_defense=(),
        deck=cs("JD QH KH AH"),
        trump_card=c("6D"),
        trump_suit="D",
        seen_cards=known_taken + (c("6D"),),
        attacker_idx=0,
        phase="ATTACK",
    )
    for _ in range(30):
        det = agent.determinize(state)
        assert known_taken[0] in det.hand_defender
        assert known_taken[1] in det.hand_defender
        # Keine der bekannten Karten taucht je im simulierten Deck auf.
        assert known_taken[0] not in det.deck
        assert known_taken[1] not in det.deck
        combined = list(det.hand_attacker) + list(det.hand_defender) + list(det.deck)
        assert len(combined) == len(set(combined))
        assert det.hand_attacker == state.hand_attacker


# --------------------------------------------------------------------------
# 3. Zeitbudget wird eingehalten
# --------------------------------------------------------------------------


def test_03_time_budget_is_respected():
    state = initial_state(seed=11)
    agent = ISMCTSAgent(player_idx=0, time_budget=0.5, seed=11)
    opp_rng = random.Random(99)
    moves_checked = 0
    while moves_checked < 10:
        legal = legal_actions(state)
        if not legal:
            break
        if _actor(state) == 0:
            t0 = time.monotonic()
            action = agent.choose_action(state)
            elapsed = time.monotonic() - t0
            assert elapsed <= 1.0, f"choose_action dauerte {elapsed:.3f}s (> 1.0s)."
            moves_checked += 1
        else:
            action = _random_policy(state, opp_rng)
        state, _reward, done = step(state, action)
        if done:
            break


# --------------------------------------------------------------------------
# 4. Reproduzierbarkeit
# --------------------------------------------------------------------------


def test_04_reproducible_with_same_seed():
    # Kleiner, fast entschiedener Zustand: Deck leer, nur zwei ATTACK-Optionen
    # mit klar unterschiedlichem p_beat -- der Suchbaum sättigt schnell, das
    # Ergebnis ist unabhängig von kleinen Schwankungen der Iterationszahl.
    state = make_state(
        hand_attacker=cs("6H AC"),  # 6H (Nicht-Trumpf) vs AC (Trumpf-Ass)
        hand_defender=cs("7S 8S 9S"),
        table_attack=(),
        table_defense=(),
        deck=(),
        trump_card=None,
        trump_suit="C",
        seen_cards=cs("6D 7D 8D 9D 10D JD QD KD AD 6S 6C 7C"),
        attacker_idx=0,
        phase="ATTACK",
    )

    def pick():
        agent = ISMCTSAgent(player_idx=0, time_budget=0.2, seed=123)
        return agent.choose_action(state)

    first = pick()
    for _ in range(3):
        assert pick() == first, "Gleicher Seed und gleiche Zustandsfolge liefern unterschiedliche Züge."


# --------------------------------------------------------------------------
# 5. Gewinnrate gegen Zufalls-Gegner (>= 60 % über 50 Partien)
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_05_wins_at_least_60_percent_vs_random_opponent():
    n_games = 50
    wins = 0
    for seed in range(n_games):
        agent_idx = seed % 2  # abwechselnd Angreifer/Verteidiger zu Beginn
        result, _plies = _play_game(seed, agent_idx=agent_idx, time_budget=0.15)
        if result == agent_idx:
            wins += 1
    win_rate = wins / n_games
    assert win_rate >= 0.60, f"Gewinnrate nur {win_rate:.0%} ({wins}/{n_games})."
