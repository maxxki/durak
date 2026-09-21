"""Tests für durak_env.py (Stufe 1).

Die acht geforderten Tests sind als ``test_01_...`` bis ``test_08_...``
nummeriert; danach folgen zusätzliche Absicherungen (Startzustand und
Zufallspartien mit Invarianten-Prüfung).
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from durak_env import (
    ACTION_TYPES,
    FULL_DECK,
    DurakEnv,
    State,
    beats,
    discarded_cards,
    initial_state,
    legal_actions,
    loser,
    sort_hand,
    step,
    terminal_reward,
    winner,
)


# --------------------------------------------------------------------------
# Hilfen
# --------------------------------------------------------------------------


def c(text: str):
    """'7S' -> ('7', 'S'), '10H' -> ('10', 'H')."""
    return (text[:-1], text[-1])


def cs(text: str):
    """'7S 8H' -> (('7','S'), ('8','H'))."""
    return tuple(c(part) for part in text.split())


def make_state(**overrides) -> State:
    base = dict(
        hand_attacker=(),
        hand_defender=(),
        table_attack=(),
        table_defense=(),
        deck=(),
        trump_card=None,
        trump_suit="D",
        seen_cards=(),
        attacker_idx=0,
        phase="ATTACK",
    )
    base.update(overrides)
    for key in ("hand_attacker", "hand_defender"):
        base[key] = sort_hand(base[key])
    return State(**base)


def actions_of(state: State, atype: str) -> set:
    return {a for a in legal_actions(state) if a[0] == atype}


# --------------------------------------------------------------------------
# 1. Trumpf-Logik
# --------------------------------------------------------------------------


def test_01_trump_beats_every_other_suit():
    t = "H"
    assert beats(c("6H"), c("AS"), t)  # kleinster Trumpf schlägt Ass
    assert not beats(c("AS"), c("6H"), t)  # Nicht-Trumpf schlägt nie Trumpf
    assert beats(c("KS"), c("QS"), t)  # gleiche Farbe, höherer Rang
    assert not beats(c("QS"), c("KS"), t)
    assert not beats(c("AD"), c("6S"), t)  # andere Nicht-Trumpf-Farbe
    assert beats(c("7H"), c("6H"), t)  # Trumpf gegen Trumpf
    assert not beats(c("6H"), c("7H"), t)
    assert not beats(c("6S"), c("6S"), t)  # gleiche Karte schlägt sich nicht

    state = make_state(
        hand_attacker=cs("9C 10C"),
        hand_defender=cs("6H KS 7D AH"),
        table_attack=(c("AS"),),
        table_defense=(None,),
        trump_suit="H",
        phase="DEFEND",
    )
    assert actions_of(state, "DEFEND") == {("DEFEND", c("6H")), ("DEFEND", c("AH"))}
    new, reward, done = step(state, ("DEFEND", c("6H")))
    assert new.table_defense == (c("6H"),)
    assert c("6H") not in new.hand_defender
    assert new.phase == "ATTACK"  # alles beantwortet: Angreifer darf nachwerfen/passen
    assert (reward, done) == (0, False)


# --------------------------------------------------------------------------
# 2. Transfer
# --------------------------------------------------------------------------


def test_02_transfer_rank_equality_and_role_swap():
    state = make_state(
        hand_attacker=cs("9D KC QS"),
        hand_defender=cs("7H 7C AD"),
        table_attack=(c("7S"),),
        table_defense=(None,),
        trump_suit="H",
        phase="DEFEND",
        attacker_idx=0,
    )
    actions = legal_actions(state)
    assert ("TRANSFER", c("7C")) in actions
    assert ("TRANSFER", c("7H")) in actions
    assert ("TRANSFER", c("AD")) not in actions  # falscher Rang

    new, reward, done = step(state, ("TRANSFER", c("7C")))
    assert new.attacker_idx == 1  # Rollentausch
    assert new.hand_defender == state.hand_attacker  # alter Angreifer verteidigt jetzt
    assert new.hand_attacker == sort_hand(cs("7H AD"))
    assert new.table_attack == (c("7S"), c("7C"))
    assert new.table_defense == (None, None)
    assert new.phase == "DEFEND"
    assert c("7C") in new.seen_cards
    assert (reward, done) == (0, False)

    with pytest.raises(ValueError, match="gleichen Rang"):
        step(state, ("TRANSFER", c("AD")))


def test_02b_transfer_not_allowed_after_defense_or_with_too_few_cards():
    # Schon eine Karte beantwortet -> kein Transfer mehr.
    answered = make_state(
        hand_attacker=cs("9D KC QS"),
        hand_defender=cs("7H 7C AD"),
        table_attack=cs("7S 7D"),
        table_defense=(c("8S"), None),
        trump_suit="H",
        phase="DEFEND",
    )
    assert not actions_of(answered, "TRANSFER")
    with pytest.raises(ValueError, match="nicht mehr erlaubt"):
        step(answered, ("TRANSFER", c("7C")))

    # Neuer Verteidiger hat nur 1 Karte, brauchte aber 2 -> kein Transfer.
    too_small = make_state(
        hand_attacker=cs("KC"),
        hand_defender=cs("7H 7C AD"),
        table_attack=(c("7S"),),
        table_defense=(None,),
        trump_suit="D",
        phase="DEFEND",
    )
    assert not actions_of(too_small, "TRANSFER")
    with pytest.raises(ValueError, match="zu wenige Handkarten"):
        step(too_small, ("TRANSFER", c("7H")))

    # Transfer-Kette: der neue Verteidiger darf mit gleichem Rang weiterreichen.
    chain = make_state(
        hand_attacker=cs("7H 9D KC AS"),
        hand_defender=cs("7C 8D 9H"),
        table_attack=cs("7S 7D"),
        table_defense=(None, None),
        trump_suit="C",
        phase="DEFEND",
    )
    assert actions_of(chain, "TRANSFER") == {("TRANSFER", c("7C"))}


# --------------------------------------------------------------------------
# 3. Nachwerfen-Limit
# --------------------------------------------------------------------------


def test_03_throw_in_limit_equals_defender_hand_size():
    # Fall A: Verteidiger hatte 4 Karten (2 abgewehrt + 2 auf der Hand) -> Nachwerfen ok.
    a = make_state(
        hand_attacker=cs("7D 7C 8D KS"),
        hand_defender=cs("9C 10C"),
        table_attack=cs("7S 7H"),
        table_defense=cs("8S 8H"),
        trump_suit="D",
        phase="ATTACK",
    )
    assert actions_of(a, "ATTACK") == {
        ("ATTACK", c("7D")),
        ("ATTACK", c("7C")),
        ("ATTACK", c("8D")),
    }  # KS hat keinen Rang auf dem Tisch
    assert ("PASS", None) in legal_actions(a)

    # Fall B: Verteidiger hatte nur 2 Karten (beide abgewehrt) -> Limit erreicht.
    b = make_state(
        hand_attacker=cs("7D 7C 8D KS"),
        hand_defender=(),
        table_attack=cs("7S 7H"),
        table_defense=cs("8S 8H"),
        deck=cs("6C"),
        trump_card=c("6H"),
        trump_suit="H",
        phase="ATTACK",
    )
    assert legal_actions(b) == [("PASS", None)]
    with pytest.raises(ValueError, match="Nachwerfen-Limit"):
        step(b, ("ATTACK", c("7D")))

    # Fall C: absolute Obergrenze 6, auch wenn der Verteidiger mehr Karten hält.
    c6 = make_state(
        hand_attacker=cs("6C 7C 8C"),
        hand_defender=cs("9S 9H 10S 10H QS QH"),
        table_attack=cs("6S 6H 7S 7H 8S 8H"),
        table_defense=cs("6D 7D 8D 9D 10D JD"),
        trump_suit="D",
        phase="ATTACK",
    )
    assert legal_actions(c6) == [("PASS", None)]

    # Fall D: Nachwerfen mit Rang, der nicht auf dem Tisch liegt, wird abgelehnt.
    with pytest.raises(ValueError, match="Nachwerfen nur mit Rängen"):
        step(a, ("ATTACK", c("KS")))


# --------------------------------------------------------------------------
# 4. Nachziehreihenfolge
# --------------------------------------------------------------------------


def _draw_fixture():
    att = cs("6C 9C 10C")
    dfn = cs("JC QC KC AC")
    trump = c("6D")
    used = set(att) | set(dfn) | set(cs("7S 8S")) | {trump}
    deck = tuple(card for card in FULL_DECK if card not in used)[:10]
    state = make_state(
        hand_attacker=att,
        hand_defender=dfn,
        table_attack=(c("7S"),),
        table_defense=(c("8S"),),
        deck=deck,
        trump_card=trump,
        trump_suit="D",
        seen_cards=(trump, c("7S"), c("8S")),
        phase="ATTACK",
    )
    return state, att, dfn, deck, trump


def test_04_draw_order_attacker_first_fill_to_six():
    state, att, dfn, deck, trump = _draw_fixture()
    new, reward, done = step(state, ("PASS", None))
    # Angreifer zieht zuerst (deck[0:3]), Verteidiger danach (deck[3:5]).
    assert new.hand_defender == sort_hand(att + deck[:3])  # alter Angreifer, jetzt Verteidiger
    assert new.hand_attacker == sort_hand(dfn + deck[3:5])  # alter Verteidiger, jetzt Angreifer
    assert len(new.hand_attacker) == len(new.hand_defender) == 6
    assert new.deck == deck[5:]
    assert new.trump_card == trump
    assert new.table_attack == () and new.table_defense == ()
    assert new.attacker_idx == 1 and new.phase == "ATTACK"
    assert (reward, done) == (0, False)
    assert set(discarded_cards(new)) == {c("7S"), c("8S")}


def test_04b_trump_card_is_drawn_last():
    state, att, dfn, deck, trump = _draw_fixture()
    short = replace(state, deck=deck[:2])
    new, _, done = step(short, ("PASS", None))
    # Angreifer braucht 3: zwei Deckkarten + die Trumpfkarte; der Verteidiger geht leer aus.
    assert new.hand_defender == sort_hand(att + deck[:2] + (trump,))
    assert new.hand_attacker == sort_hand(dfn)
    assert new.deck == () and new.trump_card is None
    assert new.trump_suit == "D"
    assert not done


def test_04c_after_take_attacker_draws_and_roles_stay():
    state = make_state(
        hand_attacker=cs("6C 9C 10C JC QC"),
        hand_defender=cs("KC AC 6H 7H 8H 9H"),
        table_attack=(c("7S"),),
        table_defense=(None,),
        deck=cs("6S 8S 9S"),
        trump_card=c("6D"),
        trump_suit="D",
        seen_cards=cs("6D 7S"),
        phase="DEFEND",
    )
    new, reward, done = step(state, ("TAKE", None))
    assert new.hand_defender == sort_hand(state.hand_defender + (c("7S"),))
    assert new.hand_attacker == sort_hand(state.hand_attacker + (c("6S"),))
    assert new.deck == cs("8S 9S")
    assert new.attacker_idx == 0  # Rollen unverändert
    assert new.phase == "ATTACK" and new.table_attack == ()
    assert (reward, done) == (0, False)


# --------------------------------------------------------------------------
# 5. Endspiel-Erkennung und Durak-Bestimmung
# --------------------------------------------------------------------------


@pytest.mark.parametrize("defender_cards", ["KH AD", "KH"])
@pytest.mark.parametrize("attacker_idx", [0, 1])
def test_05_attacker_plays_last_card_wins(defender_cards, attacker_idx):
    # Auch "gleichzeitiges" Ausspielen (Verteidiger hat nur 1 Karte): Angreifer gewinnt.
    state = make_state(
        hand_attacker=cs("7S"),
        hand_defender=cs(defender_cards),
        trump_suit="D",
        attacker_idx=attacker_idx,
        phase="ATTACK",
    )
    new, reward, done = step(state, ("ATTACK", c("7S")))
    assert done and new.phase == "GAME_OVER"
    assert winner(new) == attacker_idx
    assert loser(new) == 1 - attacker_idx
    assert reward == (1 if attacker_idx == 0 else -1)  # Sicht von Spieler 0
    assert terminal_reward(new, 1) == -reward
    assert legal_actions(new) == []


def test_05b_defender_defends_with_last_card_wins():
    state = make_state(
        hand_attacker=cs("KH AD"),
        hand_defender=cs("8S"),
        table_attack=(c("7S"),),
        table_defense=(None,),
        trump_suit="D",
        attacker_idx=0,
        phase="DEFEND",
    )
    new, reward, done = step(state, ("DEFEND", c("8S")))
    assert done and new.phase == "GAME_OVER"
    assert winner(new) == 1 and loser(new) == 0  # Angreifer mit Karten ist der Durak
    assert reward == -1


def test_05c_game_over_after_round_end_when_defender_is_out():
    # Verteidiger hat abgewehrt und ist leer; der Angreifer zieht das restliche Deck samt Trumpf.
    state = make_state(
        hand_attacker=cs("6C 9C 10C JC"),
        hand_defender=(),
        table_attack=(c("7S"),),
        table_defense=(c("8S"),),
        deck=(c("QC"),),
        trump_card=c("KC"),
        trump_suit="C",
        seen_cards=cs("KC 7S 8S"),
        attacker_idx=0,
        phase="ATTACK",
    )
    new, reward, done = step(state, ("PASS", None))
    assert done and new.phase == "GAME_OVER"
    assert new.attacker_idx == 1  # Rollen wurden getauscht
    assert winner(new) == 1  # der Spieler, der zuletzt verteidigt hat, ist raus
    assert reward == -1
    assert len(new.hand_defender) == 6  # Durak hält alle restlichen Karten


def test_05d_empty_hand_with_cards_left_in_deck_is_not_game_over():
    state = make_state(
        hand_attacker=cs("7S"),
        hand_defender=cs("KH AD 9C"),
        deck=(c("QC"),),
        trump_card=c("KC"),
        trump_suit="C",
        phase="ATTACK",
    )
    new, reward, done = step(state, ("ATTACK", c("7S")))
    assert not done and new.phase == "DEFEND" and reward == 0


# --------------------------------------------------------------------------
# 6. Ungültige Züge
# --------------------------------------------------------------------------


def test_06_invalid_actions_raise_value_error():
    attack_state = make_state(
        hand_attacker=cs("7S KH"),
        hand_defender=cs("8S 9C"),
        deck=cs("6C"),
        trump_card=c("6H"),
        trump_suit="H",
        phase="ATTACK",
    )
    defend_state = make_state(
        hand_attacker=cs("KH 9C"),
        hand_defender=cs("6D 8H 9S"),
        table_attack=(c("7S"),),
        table_defense=(None,),
        trump_suit="H",
        phase="DEFEND",
    )
    over = replace(attack_state, phase="GAME_OVER")

    cases = [
        (attack_state, ("FOO", None), "Unbekannter Aktionstyp"),
        (attack_state, ["ATTACK", c("7S")], "Tupel"),
        (attack_state, ("ATTACK", c("AS")), "nicht auf der Hand"),
        (attack_state, ("PASS", None), "PASS"),  # Eröffnung ohne Angriffskarte
        (attack_state, ("TAKE", None), "nur in Phase DEFEND"),
        (attack_state, ("DEFEND", c("8S")), "nur in Phase DEFEND"),
        (defend_state, ("DEFEND", c("6D")), "schlägt"),  # 6D schlägt 7S nicht (Trumpf ist H)
        (defend_state, ("DEFEND", c("KS")), "nicht auf der Hand"),
        (defend_state, ("ATTACK", c("KH")), "nur in Phase ATTACK"),
        (defend_state, ("TAKE", c("6D")), "payload None"),
        (over, ("ATTACK", c("7S")), "beendet"),
    ]
    for state, action, fragment in cases:
        with pytest.raises(ValueError, match=fragment):
            step(state, action)


def test_06b_legal_actions_use_only_the_schema():
    state = initial_state(seed=5)
    for action in legal_actions(state):
        assert isinstance(action, tuple) and len(action) == 2
        assert action[0] in ACTION_TYPES
        if action[0] in ("TAKE", "PASS"):
            assert action[1] is None
        else:
            assert action[1] in state.hand_attacker


# --------------------------------------------------------------------------
# 7. Edge Case: leeres Deck
# --------------------------------------------------------------------------


def test_07_empty_deck():
    state = make_state(
        hand_attacker=cs("9C 10C"),
        hand_defender=cs("JC QC"),
        table_attack=(c("7S"),),
        table_defense=(c("8S"),),
        deck=(),
        trump_card=None,
        trump_suit="D",
        seen_cards=cs("7S 8S"),
        phase="ATTACK",
    )
    # Abgewehrt bei leerem Deck: nichts wird nachgezogen, nur Rollentausch.
    new, reward, done = step(state, ("PASS", None))
    assert not done and reward == 0
    assert new.deck == () and new.trump_card is None and new.trump_suit == "D"
    assert new.hand_attacker == state.hand_defender
    assert new.hand_defender == state.hand_attacker
    assert new.attacker_idx == 1 and new.phase == "ATTACK"
    assert actions_of(new, "ATTACK") == {("ATTACK", card) for card in new.hand_attacker}

    # Aufnehmen bei leerem Deck: Verteidiger behält die Rolle, kein Nachziehen.
    defend = make_state(
        hand_attacker=cs("9C 10C"),
        hand_defender=cs("JC QC"),
        table_attack=(c("7S"),),
        table_defense=(None,),
        trump_suit="D",
        phase="DEFEND",
    )
    taken, reward, done = step(defend, ("TAKE", None))
    assert not done and reward == 0
    assert taken.hand_defender == sort_hand(cs("JC QC 7S"))
    assert taken.hand_attacker == defend.hand_attacker
    assert taken.attacker_idx == 0 and taken.phase == "ATTACK"
    assert taken.table_attack == () and taken.deck == ()


# --------------------------------------------------------------------------
# 8. Seed-Reproduzierbarkeit
# --------------------------------------------------------------------------


def test_08_seed_reproducibility():
    env1, env2 = DurakEnv(123), DurakEnv(123)
    s1, s2 = env1.initial_state(), env2.initial_state()
    assert s1 == s2 and repr(s1) == repr(s2) and hash(s1) == hash(s2)
    assert DurakEnv(124).initial_state() != s1
    assert initial_state(seed=123) == s1  # Modulfunktion == erste Partie der Env

    rng = random.Random(0)
    for _ in range(80):
        actions = env1.legal_actions(s1)
        action = actions[rng.randrange(len(actions))]
        s1, r1, d1 = env1.step(s1, action)
        s2, r2, d2 = env2.step(s2, action)
        assert (s1, r1, d1) == (s2, r2, d2)
        assert repr(s1) == repr(s2)
        if d1:
            break

    # Auch die folgenden Partien beider Envs stimmen überein.
    assert env1.initial_state() == env2.initial_state()


# --------------------------------------------------------------------------
# Zusätzliche Absicherungen
# --------------------------------------------------------------------------


def test_09_initial_state_layout():
    s = initial_state(seed=7)
    cards = list(s.hand_attacker) + list(s.hand_defender) + list(s.deck) + [s.trump_card]
    assert len(cards) == 36 and set(cards) == set(FULL_DECK)
    assert len(s.hand_attacker) == len(s.hand_defender) == 6
    assert len(s.deck) == 23
    assert s.trump_suit == s.trump_card[1]
    assert s.seen_cards == (s.trump_card,)
    assert s.attacker_idx == 0 and s.phase == "ATTACK"
    assert s.table_attack == () and s.table_defense == ()
    # Erste Aktion: jede Handkarte, kein PASS.
    assert legal_actions(s) == [("ATTACK", card) for card in s.hand_attacker]


def _check_invariants(state: State) -> None:
    on_table = [card for card in state.table_defense if card is not None]
    everything = (
        list(state.hand_attacker)
        + list(state.hand_defender)
        + list(state.table_attack)
        + on_table
        + list(state.deck)
        + list(discarded_cards(state))
    )
    if state.trump_card is not None:
        everything.append(state.trump_card)
    assert len(everything) == 36 and set(everything) == set(FULL_DECK)
    assert len(state.table_attack) == len(state.table_defense) <= 6
    assert len(set(state.seen_cards)) == len(state.seen_cards)
    assert not set(state.deck) & set(state.seen_cards)  # Deck ist nie offen gewesen
    assert set(state.table_attack) <= set(state.seen_cards)
    assert set(on_table) <= set(state.seen_cards)
    assert state.hand_attacker == sort_hand(state.hand_attacker)
    assert state.hand_defender == sort_hand(state.hand_defender)


@pytest.mark.parametrize("seed", range(40))
def test_10_random_playthrough_keeps_invariants_and_terminates(seed):
    env = DurakEnv(seed)
    rng = random.Random(seed)
    state = env.initial_state()
    for _ in range(3000):
        _check_invariants(state)
        actions = env.legal_actions(state)
        assert actions, "kein legaler Zug in einem Nicht-Endzustand"
        state, reward, done = env.step(state, rng.choice(actions))
        if done:
            _check_invariants(state)
            assert state.phase == "GAME_OVER"
            assert winner(state) in (0, 1)
            assert loser(state) == 1 - winner(state)
            assert reward == (1 if winner(state) == 0 else -1)
            assert env.legal_actions(state) == []
            return
        assert reward == 0
    pytest.fail("Partie endete nicht innerhalb von 3000 Zügen")
