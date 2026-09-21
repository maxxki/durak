"""Durak-KI, Stufe 2a: ISMCTS-Agent mit heuristischer Simulations-Policy.

Informationssicht
------------------
``ISMCTSAgent`` erhält bei ``choose_action`` den echten, vollinformierten
``State`` aus ``durak_env`` (er enthält beide Hände und das Deck), verwendet
davon aber absichtlich nur das, was der Agent in einer echten Partie sehen
dürfte:

* die eigene Hand (``own_hand``), exakt bekannt,
* ``state.seen_cards`` (alles, was je offen lag, plus die Trumpfkarte),
* öffentliche Größen (Handgröße des Gegners, Deckgröße),
* und alles, was sich daraus *ableiten* lässt, ohne die Gegnerhand direkt zu
  lesen -- siehe ``discarded_cards`` in ``durak_env``: sie berechnet, welche
  gesehenen Karten endgültig aus dem Spiel sind (abgelegt). Jede gesehene
  Karte, die weder in der eigenen Hand liegt, noch abgelegt ist, noch die
  (immer bekannte) offene Trumpfkarte ist, MUSS in der Gegnerhand liegen --
  das ist exakt das, was ein aufmerksamer menschlicher Mitspieler am Tisch
  auch sehen würde (z. B. wenn der Gegner aufnimmt: die aufgenommenen Karten
  waren öffentlich sichtbar, ihr Verbleib in seiner Hand ist also bekannt,
  keine Information wird "erschlichen").

Determinisierung (Option B: Constraint-Check)
----------------------------------------------
Aus dieser Beobachtung folgt eine einfache und exakte Umsetzung des in der
Aufgabenstellung beschriebenen Constraint-Checks, ganz ohne einen separaten
Aktionshistorien-Log:

* ``known_opponent_cards`` = gesehene Karten, die nicht in der eigenen Hand,
  nicht abgelegt und nicht die (noch ungezogene) offene Trumpfkarte sind.
  Diese Karten sind zu 100 % in der Gegnerhand -- sie werden fest zugeteilt,
  nie in den Sampling-Pool gelegt.
* ``unknown_pool`` = alle 36 Karten minus eigene Hand minus gesehene Karten.
  Das sind genau die noch unbekannten Original-/gezogenen Karten des Gegners
  plus der unbekannte Teil des Decks. Daraus werden die fehlenden
  ``opponent_hand_size - len(known_opponent_cards)`` Handkarten des Gegners
  zufällig gezogen; der Rest bildet das simulierte Deck.

Diese Konstruktion erfüllt die drei genannten Constraints automatisch:
- Eine Karte, die der Gegner nachweislich nicht schlagen konnte (TAKE), wird
  Teil seiner *bekannten* Hand, nie einer widersprüchlichen Zuteilung.
- Eine Karte, mit der er nachweislich abgewehrt hat, steht bereits sichtbar
  auf ``table_defense`` bzw. ist über ``known_opponent_cards``/Ablage erfasst.
- Nachwurf-Ränge sind über ``table_attack`` ohnehin direkt sichtbar.

Reproduzierbarkeit
-------------------
Ein einziger ``random.Random(seed)`` wird im Konstruktor erzeugt und für
*alle* Zufallsentscheidungen (Determinisierung, Baumexpansion, Tie-Breaks,
Rollout-Policy) weiterverwendet -- nie das globale ``random``-Modul. Bei
gleicher Zustandsfolge und gleichem Seed liefert der Agent damit denselben
Zug (siehe Tests).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import replace

from durak_env import (
    Action,
    Card,
    FULL_DECK,
    RANKS,
    State,
    beats,
    discarded_cards,
    legal_actions,
    sort_hand,
    step,
    terminal_reward,
)

__all__ = ["ISMCTSAgent"]

_RANK_INDEX = {rank: i for i, rank in enumerate(RANKS)}
_UCB_C = math.sqrt(2.0)
_MAX_ROLLOUT_PLIES = 500  # Sicherheitsnetz; reale Durak-Partien enden weit früher.


# --------------------------------------------------------------------------
# Baum-Knoten
# --------------------------------------------------------------------------


class _Node:
    """Ein Informationsmengen-Knoten im ISMCTS-Baum.

    Wird über mehrere Determinisierungen hinweg geteilt. ``avail`` zählt,
    wie oft eine Aktion an diesem Knoten überhaupt verfügbar war (nicht jede
    Determinisierung eröffnet dieselben Gegneraktionen) -- Grundlage für die
    UCB1-Variante mit ungleicher Verfügbarkeit (Cowling et al., ISMCTS).
    """

    __slots__ = ("children", "visits", "value", "avail")

    def __init__(self) -> None:
        self.children: dict[Action, "_Node"] = {}
        self.visits: dict[Action, int] = {}
        self.value: dict[Action, float] = {}
        self.avail: dict[Action, int] = {}


def _ucb_select(node: _Node, legal: list[Action], rng: random.Random) -> Action:
    best_score = None
    best_actions: list[Action] = []
    for action in legal:
        n_a = node.visits[action]
        avail_a = node.avail[action]
        exploit = node.value[action] / n_a
        explore = _UCB_C * math.sqrt(math.log(avail_a) / n_a)
        score = exploit + explore
        if best_score is None or score > best_score:
            best_score = score
            best_actions = [action]
        elif score == best_score:
            best_actions.append(action)
    if len(best_actions) == 1:
        return best_actions[0]
    return best_actions[rng.randrange(len(best_actions))]


# --------------------------------------------------------------------------
# Heuristische Simulations-Policy (für Rollouts)
# --------------------------------------------------------------------------


def _own_unknown_pool(hand: tuple[Card, ...], seen: tuple[Card, ...]) -> list[Card]:
    seen_set = set(seen)
    hand_set = set(hand)
    return [c for c in FULL_DECK if c not in hand_set and c not in seen_set]


def _defend_choice(state: State, rng: random.Random) -> Action | None:
    defend_actions = [a for a in legal_actions(state) if a[0] == "DEFEND"]
    if not defend_actions:
        return None

    def key(action: Action) -> tuple[int, bool]:
        card = action[1]
        return (_RANK_INDEX[card[0]], card[1] == state.trump_suit)

    best_key = min(key(a) for a in defend_actions)
    candidates = [a for a in defend_actions if key(a) == best_key]
    if len(candidates) == 1:
        return candidates[0]
    return candidates[rng.randrange(len(candidates))]


def _endgame_attack_choice(
    state: State, attack_actions: list[Action], rng: random.Random
) -> Action:
    own_hand = state.hand_attacker
    unknown_pool = _own_unknown_pool(own_hand, state.seen_cards)

    def p_beat(card: Card) -> float:
        if not unknown_pool:
            return 0.0
        beating = sum(1 for u in unknown_pool if beats(u, card, state.trump_suit))
        return beating / len(unknown_pool)

    ace_trump: Card = (RANKS[-1], state.trump_suit)
    non_ace = [a for a in attack_actions if a[1] != ace_trump]
    ace_action = next((a for a in attack_actions if a[1] == ace_trump), None)

    if not non_ace:
        # Trumpf-Ass ist die einzige Option ("keine Alternative").
        return attack_actions[0]

    scored = [(p_beat(a[1]), _RANK_INDEX[a[1][0]], a) for a in non_ace]
    best_alt_p = min(s[0] for s in scored)

    if ace_action is not None and p_beat(ace_action[1]) < best_alt_p:
        # Trumpf-Ass ist strikt besser als jede Alternative.
        return ace_action

    candidates = [s for s in scored if s[0] == best_alt_p]
    min_rank = min(s[1] for s in candidates)
    candidates = [s for s in candidates if s[1] == min_rank]
    if len(candidates) == 1:
        return candidates[0][2]
    return candidates[rng.randrange(len(candidates))][2]


def _attack_choice(state: State, rng: random.Random) -> Action:
    attack_actions = [a for a in legal_actions(state) if a[0] == "ATTACK"]
    if not attack_actions:
        return ("PASS", None)

    endgame = not state.deck and state.trump_card is None
    if endgame:
        return _endgame_attack_choice(state, attack_actions, rng)

    best_rank = min(_RANK_INDEX[a[1][0]] for a in attack_actions)
    candidates = [a for a in attack_actions if _RANK_INDEX[a[1][0]] == best_rank]
    if len(candidates) == 1:
        return candidates[0]
    return candidates[rng.randrange(len(candidates))]


def heuristic_policy(state: State, rng: random.Random) -> Action | None:
    """Deterministische Rollout-Policy (Zufall nur bei echten Gleichständen).

    Gibt ``None`` zurück, wenn ``state`` keine legalen Aktionen mehr hat
    (Spielende erreicht).
    """
    if state.phase == "DEFEND":
        chosen = _defend_choice(state, rng)
        if chosen is not None:
            return chosen
        transfer_actions = [a for a in legal_actions(state) if a[0] == "TRANSFER"]
        if transfer_actions and len(state.hand_attacker) > len(state.hand_defender):
            if len(transfer_actions) == 1:
                return transfer_actions[0]
            return transfer_actions[rng.randrange(len(transfer_actions))]
        return ("TAKE", None)
    if state.phase == "ATTACK":
        return _attack_choice(state, rng)
    return None


# --------------------------------------------------------------------------
# ISMCTS-Agent
# --------------------------------------------------------------------------


class ISMCTSAgent:
    """ISMCTS-Agent für einen festen Spieler-Index gegen imperfekte Information."""

    def __init__(self, player_idx: int, time_budget: float = 2.0, seed: int = 0) -> None:
        if player_idx not in (0, 1):
            raise ValueError(f"player_idx muss 0 oder 1 sein, erhalten: {player_idx!r}")
        self.player_idx = player_idx
        self.time_budget = time_budget
        self._rng = random.Random(seed)

    # -- Determinisierung ---------------------------------------------------

    def determinize(self, state: State) -> State:
        """Erzeugt einen vollinformierten Kandidaten-State, konsistent mit
        allem, was der Agent aus ``state`` wissen darf (siehe Moduldoc).
        """
        if state.attacker_idx == self.player_idx:
            own_hand = state.hand_attacker
            opp_hand_size = len(state.hand_defender)
        else:
            own_hand = state.hand_defender
            opp_hand_size = len(state.hand_attacker)

        own_set = set(own_hand)
        seen_set = set(state.seen_cards)
        discarded_set = set(discarded_cards(state))
        trump_reserved = {state.trump_card} if state.trump_card is not None else set()

        known_opponent = seen_set - own_set - discarded_set - trump_reserved
        # Defensive Absicherung: sollte durch die Env-Invarianten nie greifen.
        known_opponent = set(list(known_opponent)[:opp_hand_size])

        unknown_pool = [c for c in FULL_DECK if c not in own_set and c not in seen_set]
        self._rng.shuffle(unknown_pool)

        need = max(0, opp_hand_size - len(known_opponent))
        sampled = unknown_pool[:need]
        opp_hand = sort_hand(list(known_opponent) + sampled)
        deck = tuple(unknown_pool[need:])

        if state.attacker_idx == self.player_idx:
            hand_attacker, hand_defender = own_hand, opp_hand
        else:
            hand_attacker, hand_defender = opp_hand, own_hand

        return replace(state, hand_attacker=hand_attacker, hand_defender=hand_defender, deck=deck)

    # -- Öffentliche API ------------------------------------------------------

    def choose_action(self, state: State) -> Action:
        legal = legal_actions(state)
        if not legal:
            raise ValueError("Keine legalen Aktionen: state ist terminal.")
        if len(legal) == 1:
            return legal[0]

        root = _Node()
        start = time.monotonic()
        while True:
            det_state = self.determinize(state)
            self._simulate(root, det_state)
            if time.monotonic() - start >= self.time_budget:
                break

        return max(root.children, key=lambda a: root.visits[a])

    # -- ISMCTS-Kern ------------------------------------------------------

    def _simulate(self, root: _Node, det_state: State) -> None:
        node = root
        sim_state = det_state
        path: list[tuple[_Node, Action]] = []

        while True:
            legal = legal_actions(sim_state)
            if not legal:
                result = terminal_reward(sim_state, self.player_idx)
                self._backprop(path, result)
                return

            for action in legal:
                node.avail[action] = node.avail.get(action, 0) + 1

            untried = [a for a in legal if a not in node.children]
            if untried:
                action = untried[self._rng.randrange(len(untried))]
                child = _Node()
                node.children[action] = child
                node.visits[action] = 0
                node.value[action] = 0.0
                path.append((node, action))
                sim_state, _reward, done = step(sim_state, action)
                if done:
                    result = terminal_reward(sim_state, self.player_idx)
                else:
                    result = self._rollout(sim_state)
                self._backprop(path, result)
                return

            action = _ucb_select(node, legal, self._rng)
            path.append((node, action))
            sim_state, _reward, done = step(sim_state, action)
            node = node.children[action]
            if done:
                result = terminal_reward(sim_state, self.player_idx)
                self._backprop(path, result)
                return

    def _rollout(self, state: State) -> float:
        plies = 0
        while plies < _MAX_ROLLOUT_PLIES:
            action = heuristic_policy(state, self._rng)
            if action is None:
                break
            state, _reward, done = step(state, action)
            plies += 1
            if done:
                return float(terminal_reward(state, self.player_idx))

        # Sicherheitsnetz (sollte praktisch nie erreicht werden): heuristische
        # Bewertung aus Sicht des Agenten, siehe Aufgabenstellung.
        if state.attacker_idx == self.player_idx:
            own_hand, opp_hand = state.hand_attacker, state.hand_defender
        else:
            own_hand, opp_hand = state.hand_defender, state.hand_attacker
        return -len(own_hand) + 0.5 * len(opp_hand)

    @staticmethod
    def _backprop(path: list[tuple[_Node, Action]], result: float) -> None:
        for node, action in path:
            node.visits[action] += 1
            node.value[action] += result
