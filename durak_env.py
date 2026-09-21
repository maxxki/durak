"""Durak-KI, Stufe 1: deterministische Spielumgebung (36 Karten, 2 Spieler).

Variante: Podkidnoy Durak mit Transfer (Perevodnoy).

Überblick über das Modell
-------------------------
* Der ``State`` ist ein unveränderlicher, hashbarer Vollzustand (beide Hände
  sind enthalten).  Ein Agent für Spiele mit imperfekter Information (Stufe 2,
  ISMCTS) darf daraus nur seine eigene Sicht verwenden und muss die fremde
  Hand samt Deck determinisieren.
* Die Hände sind rollenbezogen: ``hand_attacker`` gehört immer dem Spieler, der
  gerade Angreifer ist.  ``attacker_idx`` (0 oder 1) sagt, welcher der beiden
  festen Spieler das ist.  Bei jedem Rollentausch (erfolgreiche Abwehr,
  Transfer) werden die Hände zwischen den Feldern getauscht und
  ``attacker_idx`` kippt.
* Hände werden stets in kanonischer Reihenfolge (Rang, dann Farbe) gehalten,
  damit gleiche Situationen bit-identische States ergeben.
* ``step`` ist eine reine Funktion (kein RNG).  Zufall gibt es nur beim Mischen
  in ``initial_state``; dort sorgt der Seed für Reproduzierbarkeit.

Ablauf einer Runde
------------------
1. Phase ``ATTACK``, Tisch leer: Angreifer muss mit ``("ATTACK", card)``
   eröffnen (PASS ist verboten).
2. Phase ``DEFEND``: Verteidiger antwortet mit ``DEFEND`` (auf die *oberste*
   unbeantwortete Angriffskarte, d. h. die mit dem höchsten Tisch-Index),
   ``TRANSFER`` oder ``TAKE``.
3. Sind alle Angriffskarten beantwortet, wechselt die Phase zurück zu
   ``ATTACK``: der Angreifer darf nachwerfen (Rang muss bereits auf dem Tisch
   liegen, Limit siehe ``max_attack_cards``) oder mit ``PASS`` beenden.
4. Rundenende: Nach ``PASS`` (abgewehrt) werden die Tischkarten abgelegt, es
   wird nachgezogen (erst Angreifer, dann Verteidiger, jeweils bis 6; die
   offene Trumpfkarte ist die allerletzte Karte) und die Rollen tauschen.
   Nach ``TAKE`` nimmt der Verteidiger alle Tischkarten, es wird nachgezogen
   (Angreifer zuerst) und die Rollen bleiben.

Getroffene Auslegungen der Regeln (siehe auch Lieferhinweis):
* ``TAKE`` wirkt sofort; ein Nachwerfen nach dem Aufnehmen gibt es nicht.
* ``TRANSFER`` ist nur erlaubt, solange noch keine Angriffskarte beantwortet
  wurde, der Rang zur obersten Angriffskarte passt und der neue Verteidiger
  genug Handkarten hat (siehe ``_transfer_possible``).
* Die Phasen ``TRANSFER_DECISION`` und ``ROUND_END`` sind im Schema reserviert,
  werden von dieser Umgebung aber nie erzeugt (Transfer ist Teil der
  DEFEND-Entscheidung, das Rundenende wird innerhalb von ``step`` aufgelöst).
* Reward wird immer aus Sicht von Spieler 0 gegeben (+1 Sieg, -1 Niederlage).
  ``terminal_reward(state, player)`` liefert die Sicht jedes Spielers.
* Spielende (``done``), sobald das Deck samt Trumpfkarte aufgebraucht ist und
  entweder die Angreiferhand leer ist (Angreifer gewinnt, auch bei
  "gleichzeitigem" Abspielen: der Verteidiger hat dann zuletzt abgewehrt bzw.
  aufgenommen) oder die Verteidigerhand leer ist und keine Angriffskarte mehr
  unbeantwortet liegt (Verteidiger gewinnt).  Der Gewinner ist aus dem
  Endzustand ableitbar, siehe ``winner``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Iterable, Optional

__all__ = [
    "Card",
    "Action",
    "State",
    "DurakEnv",
    "RANKS",
    "SUITS",
    "FULL_DECK",
    "HAND_SIZE",
    "ACTION_TYPES",
    "PHASES",
    "initial_state",
    "legal_actions",
    "step",
    "beats",
    "sort_hand",
    "max_attack_cards",
    "discarded_cards",
    "winner",
    "loser",
    "terminal_reward",
]

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

Card = tuple[str, str]  # (rank, suit), z. B. ("A", "S") = Pik-Ass
Action = tuple[str, Optional[Card]]

RANKS: tuple[str, ...] = ("6", "7", "8", "9", "10", "J", "Q", "K", "A")
SUITS: tuple[str, ...] = ("S", "H", "D", "C")
FULL_DECK: tuple[Card, ...] = tuple((r, s) for s in SUITS for r in RANKS)
HAND_SIZE = 6

PHASE_ATTACK = "ATTACK"
PHASE_DEFEND = "DEFEND"
PHASE_TRANSFER_DECISION = "TRANSFER_DECISION"
PHASE_ROUND_END = "ROUND_END"
PHASE_GAME_OVER = "GAME_OVER"
PHASES: tuple[str, ...] = (
    PHASE_ATTACK,
    PHASE_DEFEND,
    PHASE_TRANSFER_DECISION,
    PHASE_ROUND_END,
    PHASE_GAME_OVER,
)

ACTION_TYPES: tuple[str, ...] = ("ATTACK", "DEFEND", "TRANSFER", "TAKE", "PASS")
_PHASE_OF_ACTION = {
    "ATTACK": PHASE_ATTACK,
    "PASS": PHASE_ATTACK,
    "DEFEND": PHASE_DEFEND,
    "TRANSFER": PHASE_DEFEND,
    "TAKE": PHASE_DEFEND,
}

_RANK_VALUE = {rank: i for i, rank in enumerate(RANKS)}
_SUIT_VALUE = {suit: i for i, suit in enumerate(SUITS)}


@dataclass(frozen=True)
class State:
    """Unveränderlicher Spielzustand (Vollinformation).

    Attributes:
        hand_attacker: Hand des aktuellen Angreifers (kanonisch sortiert).
        hand_defender: Hand des aktuellen Verteidigers (kanonisch sortiert).
        table_attack: Angriffskarten in Legereihenfolge.
        table_defense: Gleich lang wie ``table_attack``; ``None`` = unbeantwortet.
        deck: Nachziehstapel, gezogen wird von Index 0.  Enthält die offene
            Trumpfkarte NICHT.
        trump_card: Offene Trumpfkarte unter dem Deck (wird als letzte Karte
            gezogen); ``None``, sobald sie gezogen wurde.
        trump_suit: Trumpffarbe; bleibt nach Aufbrauchen des Decks gültig.
        seen_cards: Offene Trumpfkarte (auch nachdem sie gezogen wurde) plus
            jede Karte, die jemals offen auf dem Tisch lag (Attack, Defense,
            Transfer), ohne Duplikate in Reihenfolge des Erscheinens.  Karten,
            die nur in einer Hand liegen, zählen nicht.  Grundlage für den
            ``unknown_pool`` in Stufe 2: unbekannt sind aus Sicht eines
            Spielers alle Karten außer seiner Hand und ``seen_cards``.
        attacker_idx: Welcher der Spieler (0 oder 1) gerade Angreifer ist.
        phase: Eine der Phasen in ``PHASES``.
    """

    hand_attacker: tuple[Card, ...]
    hand_defender: tuple[Card, ...]
    table_attack: tuple[Card, ...]
    table_defense: tuple[Card | None, ...]
    deck: tuple[Card, ...]
    trump_card: Card | None
    trump_suit: str
    seen_cards: tuple[Card, ...]
    attacker_idx: int
    phase: str


# --------------------------------------------------------------------------
# Kleine Hilfsfunktionen
# --------------------------------------------------------------------------


def _card_key(card: Card) -> tuple[int, int]:
    return (_RANK_VALUE[card[0]], _SUIT_VALUE[card[1]])


def sort_hand(cards: Iterable[Card]) -> tuple[Card, ...]:
    """Bringt Karten in die kanonische Reihenfolge (Rang aufsteigend, dann Farbe)."""
    return tuple(sorted(cards, key=_card_key))


def beats(defense: Card, attack: Card, trump_suit: str) -> bool:
    """True, wenn ``defense`` die Angriffskarte ``attack`` schlägt.

    Gleiche Farbe und höherer Rang, oder ``defense`` ist Trumpf und ``attack``
    nicht.  Trumpf schlägt jede andere Farbe, ein Nicht-Trumpf nie einen Trumpf.
    """
    d_rank, d_suit = defense
    a_rank, a_suit = attack
    if d_suit == a_suit:
        return _RANK_VALUE[d_rank] > _RANK_VALUE[a_rank]
    return d_suit == trump_suit


def _remove_card(hand: tuple[Card, ...], card: Card) -> tuple[Card, ...]:
    cards = list(hand)
    cards.remove(card)
    return tuple(cards)


def _add_seen(seen: tuple[Card, ...], cards: Iterable[Card]) -> tuple[Card, ...]:
    result = list(seen)
    for card in cards:
        if card not in result:
            result.append(card)
    return tuple(result)


def _ranks_on_table(state: State) -> set[str]:
    ranks = {card[0] for card in state.table_attack}
    ranks.update(card[0] for card in state.table_defense if card is not None)
    return ranks


def _top_unanswered(state: State) -> int | None:
    """Index der obersten (zuletzt gelegten) unbeantworteten Angriffskarte."""
    for idx in range(len(state.table_defense) - 1, -1, -1):
        if state.table_defense[idx] is None:
            return idx
    return None


def max_attack_cards(state: State) -> int:
    """Maximale Zahl an Angriffskarten dieser Runde.

    Entspricht der Handgröße des Verteidigers bei Rundenbeginn (aktuelle
    Handkarten plus bereits zur Abwehr ausgespielte Karten), höchstens 6.
    """
    defended = sum(1 for card in state.table_defense if card is not None)
    return min(HAND_SIZE, len(state.hand_defender) + defended)


def _transfer_possible(state: State) -> bool:
    """Transfer-Vorbedingungen, unabhängig von der konkreten Karte.

    Transfer-Logik: Der Verteidiger reicht den Angriff zurück, indem er eine
    Karte mit dem Rang der obersten Angriffskarte dazulegt.  Das geht nur,
    solange noch keine Angriffskarte beantwortet wurde (dann liegen ohnehin
    nur Karten desselben Rangs auf dem Tisch), und nur, wenn der neue
    Verteidiger (bisheriger Angreifer) alle Angriffskarten inklusive der neuen
    abwehren *könnte*: ``len(table_attack) + 1 <= min(6, len(hand_attacker))``.
    """
    if any(card is not None for card in state.table_defense):
        return False
    if not state.table_attack:
        return False
    return len(state.table_attack) + 1 <= min(HAND_SIZE, len(state.hand_attacker))


# --------------------------------------------------------------------------
# Legale Aktionen
# --------------------------------------------------------------------------


def legal_actions(state: State) -> list[Action]:
    """Alle gültigen Züge des aktuell handelnden Spielers im Action-Schema.

    Der Trumpf-Ass wird hier nie herausgefiltert; das ist Sache der Bewertung
    (Stufe 2).  Im Endzustand (und in den reservierten Phasen) ist die Liste
    leer.
    """
    if state.phase == PHASE_ATTACK:
        return _legal_attack(state)
    if state.phase == PHASE_DEFEND:
        return _legal_defend(state)
    return []


def _legal_attack(state: State) -> list[Action]:
    if not state.table_attack:
        # Rundeneröffnung: jede Karte, kein PASS.
        return [("ATTACK", card) for card in state.hand_attacker]
    actions: list[Action] = []
    if len(state.table_attack) < max_attack_cards(state):
        ranks = _ranks_on_table(state)
        actions = [("ATTACK", card) for card in state.hand_attacker if card[0] in ranks]
    actions.append(("PASS", None))
    return actions


def _legal_defend(state: State) -> list[Action]:
    idx = _top_unanswered(state)
    if idx is None:
        return []
    top = state.table_attack[idx]
    actions: list[Action] = [
        ("DEFEND", card)
        for card in state.hand_defender
        if beats(card, top, state.trump_suit)
    ]
    if _transfer_possible(state):
        rank = state.table_attack[-1][0]
        actions.extend(
            ("TRANSFER", card) for card in state.hand_defender if card[0] == rank
        )
    actions.append(("TAKE", None))
    return actions


def _explain_illegal(state: State, action: object) -> str:
    """Erzeugt eine klare Fehlermeldung für einen ungültigen Zug."""
    if state.phase == PHASE_GAME_OVER:
        return "Das Spiel ist beendet (phase=GAME_OVER); es sind keine Aktionen mehr erlaubt."
    if state.phase not in (PHASE_ATTACK, PHASE_DEFEND):
        return f"Phase {state.phase!r} ist in dieser Umgebung nicht spielbar."
    if not isinstance(action, tuple) or len(action) != 2:
        return f"Aktion muss ein Tupel (action_type, payload) sein, erhalten: {action!r}."
    atype, payload = action
    if atype not in ACTION_TYPES:
        return f"Unbekannter Aktionstyp {atype!r}; erlaubt: {ACTION_TYPES}."
    needed = _PHASE_OF_ACTION[atype]
    if state.phase != needed:
        return f"{atype} ist nur in Phase {needed} erlaubt, aktuelle Phase: {state.phase}."
    if atype in ("TAKE", "PASS"):
        if payload is not None:
            return f"{atype} erwartet payload None, erhalten: {payload!r}."
        if atype == "PASS" and not state.table_attack:
            return "PASS ist nicht erlaubt: der Angreifer muss die Runde mit einer Angriffskarte eröffnen."
        return f"{atype} ist im aktuellen Zustand nicht erlaubt."
    hand = state.hand_attacker if atype == "ATTACK" else state.hand_defender
    owner = "Angreifers" if atype == "ATTACK" else "Verteidigers"
    if payload not in hand:
        return f"Karte {payload!r} ist nicht auf der Hand des {owner}."
    if atype == "ATTACK":
        if payload[0] not in _ranks_on_table(state):
            return (
                f"Nachwerfen nur mit Rängen, die bereits auf dem Tisch liegen "
                f"({sorted(_ranks_on_table(state), key=_RANK_VALUE.get)}); Karte {payload!r} passt nicht."
            )
        return (
            f"Nachwerfen-Limit erreicht: höchstens {max_attack_cards(state)} Angriffskarten "
            f"(Handgröße des Verteidigers bei Rundenbeginn, max. {HAND_SIZE})."
        )
    if atype == "DEFEND":
        idx = _top_unanswered(state)
        top = state.table_attack[idx] if idx is not None else None
        return (
            f"Karte {payload!r} schlägt die oberste unbeantwortete Angriffskarte {top!r} "
            f"nicht (Trumpf: {state.trump_suit})."
        )
    # TRANSFER
    if any(card is not None for card in state.table_defense):
        return "Transfer ist nicht mehr erlaubt, sobald eine Angriffskarte beantwortet wurde."
    if payload[0] != state.table_attack[-1][0]:
        return (
            f"Transfer erfordert den gleichen Rang wie die oberste Angriffskarte "
            f"{state.table_attack[-1][0]!r}, erhalten: {payload[0]!r}."
        )
    return (
        "Transfer nicht möglich: der neue Verteidiger hat zu wenige Handkarten "
        f"({len(state.hand_attacker)}) für {len(state.table_attack) + 1} Angriffskarten."
    )


# --------------------------------------------------------------------------
# Zustandsübergänge
# --------------------------------------------------------------------------


def _draw(
    hand: tuple[Card, ...], deck: tuple[Card, ...], trump_card: Card | None
) -> tuple[tuple[Card, ...], tuple[Card, ...], Card | None]:
    """Zieht bis 6 Karten; die offene Trumpfkarte kommt als allerletzte."""
    cards = list(hand)
    pile = list(deck)
    while len(cards) < HAND_SIZE:
        if pile:
            cards.append(pile.pop(0))
        elif trump_card is not None:
            cards.append(trump_card)
            trump_card = None
        else:
            break
    return sort_hand(cards), tuple(pile), trump_card


def _end_round(
    state: State,
    hand_att: tuple[Card, ...],
    hand_def: tuple[Card, ...],
    swap_roles: bool,
) -> State:
    """Räumt den Tisch, zieht nach (Angreifer zuerst) und tauscht ggf. die Rollen."""
    hand_att, deck, trump_card = _draw(hand_att, state.deck, state.trump_card)
    hand_def, deck, trump_card = _draw(hand_def, deck, trump_card)
    attacker_idx = state.attacker_idx
    if swap_roles:
        hand_att, hand_def = hand_def, hand_att
        attacker_idx = 1 - attacker_idx
    return replace(
        state,
        hand_attacker=hand_att,
        hand_defender=hand_def,
        table_attack=(),
        table_defense=(),
        deck=deck,
        trump_card=trump_card,
        attacker_idx=attacker_idx,
        phase=PHASE_ATTACK,
    )


def _apply_attack(state: State, card: Card) -> State:
    return replace(
        state,
        hand_attacker=_remove_card(state.hand_attacker, card),
        table_attack=state.table_attack + (card,),
        table_defense=state.table_defense + (None,),
        seen_cards=_add_seen(state.seen_cards, (card,)),
        phase=PHASE_DEFEND,
    )


def _apply_defend(state: State, card: Card) -> State:
    idx = _top_unanswered(state)
    defense = list(state.table_defense)
    defense[idx] = card
    all_answered = all(entry is not None for entry in defense)
    return replace(
        state,
        hand_defender=_remove_card(state.hand_defender, card),
        table_defense=tuple(defense),
        seen_cards=_add_seen(state.seen_cards, (card,)),
        phase=PHASE_ATTACK if all_answered else PHASE_DEFEND,
    )


def _apply_transfer(state: State, card: Card) -> State:
    """Transfer: Verteidiger legt eine gleichrangige Karte dazu und tauscht die Rollen.

    Die gespielte Karte wird als weitere (unbeantwortete) Angriffskarte
    ausgelegt.  Der bisherige Verteidiger wird Angreifer, der bisherige
    Angreifer wird Verteidiger und muss nun alle Angriffskarten beantworten
    (oder aufnehmen, oder seinerseits per Transfer weiterreichen).  Da die
    Hände rollenbezogen gespeichert sind, werden sie getauscht.
    """
    return replace(
        state,
        hand_attacker=_remove_card(state.hand_defender, card),
        hand_defender=state.hand_attacker,
        table_attack=state.table_attack + (card,),
        table_defense=state.table_defense + (None,),
        seen_cards=_add_seen(state.seen_cards, (card,)),
        attacker_idx=1 - state.attacker_idx,
        phase=PHASE_DEFEND,
    )


def _apply_take(state: State) -> State:
    taken = state.table_attack + tuple(
        card for card in state.table_defense if card is not None
    )
    hand_def = sort_hand(state.hand_defender + taken)
    return _end_round(state, state.hand_attacker, hand_def, swap_roles=False)


def _apply_pass(state: State) -> State:
    # Abgewehrt: Tischkarten wandern aus dem Spiel (stehen weiter in seen_cards).
    return _end_round(
        state, state.hand_attacker, state.hand_defender, swap_roles=True
    )


def _finalize(state: State) -> State:
    """Setzt GAME_OVER, wenn das Spiel entschieden ist (siehe Modul-Docstring)."""
    if state.deck or state.trump_card is not None:
        return state
    unanswered = any(card is None for card in state.table_defense)
    if not state.hand_attacker or (not state.hand_defender and not unanswered):
        return replace(state, phase=PHASE_GAME_OVER)
    return state


def winner(state: State) -> int | None:
    """Index des Gewinners im Endzustand, sonst ``None``.

    Leere Angreiferhand => Angreifer gewinnt (das gilt auch, wenn beide Hände
    gleichzeitig leer wären: dann hat der Verteidiger zuletzt abgewehrt).
    Leere Verteidigerhand => Verteidiger gewinnt.  Der andere Spieler ist der
    Durak.
    """
    if state.phase != PHASE_GAME_OVER:
        return None
    if not state.hand_attacker:
        return state.attacker_idx
    if not state.hand_defender:
        return 1 - state.attacker_idx
    return None


def loser(state: State) -> int | None:
    """Index des Durak im Endzustand, sonst ``None``."""
    won = winner(state)
    return None if won is None else 1 - won


def terminal_reward(state: State, player: int) -> int:
    """+1 / -1 / 0 (laufendes Spiel) aus Sicht von ``player``."""
    won = winner(state)
    if won is None:
        return 0
    return 1 if won == player else -1


def step(state: State, action: Action) -> tuple[State, int, bool]:
    """Wendet ``action`` an und liefert ``(new_state, reward, done)``.

    Reward: +1 wenn Spieler 0 gewinnt, -1 wenn Spieler 0 verliert, sonst 0.

    Raises:
        ValueError: Wenn ``action`` im Zustand nicht legal ist (mit Begründung).
    """
    if action not in legal_actions(state):
        raise ValueError(_explain_illegal(state, action))
    atype, card = action
    if atype == "ATTACK":
        new_state = _apply_attack(state, card)
    elif atype == "DEFEND":
        new_state = _apply_defend(state, card)
    elif atype == "TRANSFER":
        new_state = _apply_transfer(state, card)
    elif atype == "TAKE":
        new_state = _apply_take(state)
    else:  # "PASS"
        new_state = _apply_pass(state)
    new_state = _finalize(new_state)
    done = new_state.phase == PHASE_GAME_OVER
    reward = terminal_reward(new_state, 0) if done else 0
    return new_state, reward, done


def discarded_cards(state: State) -> tuple[Card, ...]:
    """Karten im Ablagestapel (Abwehr-Karten früherer Runden).

    Abgeleitet aus ``seen_cards``: alles, was einmal offen lag, sich aber
    weder in einer Hand, auf dem Tisch noch im Deck befindet.
    """
    live = set(state.hand_attacker) | set(state.hand_defender)
    live |= set(state.table_attack)
    live |= {card for card in state.table_defense if card is not None}
    live |= set(state.deck)
    if state.trump_card is not None:
        live.add(state.trump_card)
    return tuple(card for card in state.seen_cards if card not in live)


# --------------------------------------------------------------------------
# Start und Umgebung
# --------------------------------------------------------------------------


def _deal(rng: random.Random) -> State:
    deck = list(FULL_DECK)
    rng.shuffle(deck)
    hand_attacker = sort_hand(deck[0:HAND_SIZE])
    hand_defender = sort_hand(deck[HAND_SIZE : 2 * HAND_SIZE])
    trump_card = deck[-1]
    rest = tuple(deck[2 * HAND_SIZE : -1])
    return State(
        hand_attacker=hand_attacker,
        hand_defender=hand_defender,
        table_attack=(),
        table_defense=(),
        deck=rest,
        trump_card=trump_card,
        trump_suit=trump_card[1],
        seen_cards=(trump_card,),
        attacker_idx=0,
        phase=PHASE_ATTACK,
    )


def initial_state(seed: int = 0) -> State:
    """Startzustand für ``seed``: gemischt, 6+6 ausgeteilt, Trumpf aufgedeckt.

    ``attacker_idx = 0``, ``phase = "ATTACK"``.  Gleicher Seed => gleicher State.
    """
    return _deal(random.Random(seed))


class DurakEnv:
    """Dünner, seedbarer Wrapper um die reinen Funktionen dieses Moduls.

    ``initial_state()`` zieht bei jedem Aufruf ein neues, aber reproduzierbares
    Spiel aus dem internen RNG.  Zwei Envs mit gleichem Seed liefern dieselbe
    Folge von Startzuständen; ``step`` und ``legal_actions`` sind deterministisch.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def initial_state(self) -> State:
        return _deal(self._rng)

    def legal_actions(self, state: State) -> list[Action]:
        return legal_actions(state)

    def step(self, state: State, action: Action) -> tuple[State, int, bool]:
        return step(state, action)
