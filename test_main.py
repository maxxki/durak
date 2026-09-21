"""Tests für main.py (Stufe 4). Vier Tests, siehe stufe_4_prompt.md.

Alle laufen ohne Agenten-Suche in Sekunden -- keiner ist
``@pytest.mark.slow``.
"""

from __future__ import annotations

import pytest

from durak_env import initial_state, legal_actions
from main import (
    CommandError,
    build_action,
    format_card,
    parse_command,
    render_state,
)


# --------------------------------------------------------------------------
# 1. Eingabe-Parser
# --------------------------------------------------------------------------


def test_01_parse_command_recognizes_all_command_forms():
    assert parse_command("a 1") == ("ATTACK", 1)
    assert parse_command("d 3") == ("DEFEND", 3)
    assert parse_command("t 2") == ("TRANSFER", 2)
    assert parse_command("pass") == ("PASS", None)
    assert parse_command("take") == ("TAKE", None)
    assert parse_command("help") == ("HELP", None)
    assert parse_command("quit") == ("QUIT", None)
    # Groß-/Kleinschreibung und Whitespace sind unerheblich.
    assert parse_command("  A 1  ") == ("ATTACK", 1)
    assert parse_command("PASS") == ("PASS", None)

    with pytest.raises(CommandError):
        parse_command("quatsch")
    with pytest.raises(CommandError):
        parse_command("")
    with pytest.raises(CommandError):
        parse_command("a")  # fehlende Kartennummer
    with pytest.raises(CommandError):
        parse_command("a x")  # keine Zahl


# --------------------------------------------------------------------------
# 2. Karten-Formatierung
# --------------------------------------------------------------------------


def test_02_format_card_renders_rank_and_suit_symbol():
    assert format_card(("A", "S")) == "A\u2660"
    assert format_card(("10", "H")) == "10\u2665"
    assert format_card(("6", "D")) == "6\u2666"
    assert format_card(("Q", "C")) == "Q\u2663"


# --------------------------------------------------------------------------
# 3. Aktions-Mapping
# --------------------------------------------------------------------------


def test_03_build_action_maps_valid_index_and_rejects_invalid():
    # Frischer Startzustand: attacker_idx=0, phase=ATTACK -> Spieler 0 (Mensch)
    # ist am Zug und darf mit jeder Handkarte angreifen (Tisch ist leer).
    state = initial_state(seed=0)
    assert state.attacker_idx == 0
    assert state.phase == "ATTACK"

    cmd = parse_command("a 1")
    action = build_action(cmd, state, human_player=0)
    assert action == ("ATTACK", state.hand_attacker[0])
    assert action in legal_actions(state)

    # Kartennummer außerhalb der Hand -> CommandError, kein step().
    with pytest.raises(CommandError):
        build_action(("ATTACK", 99), state, human_player=0)

    # Aktionstyp, der gerade nicht erlaubt ist (DEFEND während ATTACK-Phase
    # des Angreifers) -> CommandError mit Hinweis auf erlaubte Typen.
    with pytest.raises(CommandError):
        build_action(("DEFEND", 1), state, human_player=0)


# --------------------------------------------------------------------------
# 4. Rendering-Smoke-Test
# --------------------------------------------------------------------------


def test_04_render_state_runs_without_exception_and_has_key_sections():
    state = initial_state(seed=0)
    output = render_state(state, human_player=0)
    assert "Trumpf" in output
    assert "Hand" in output
    assert "Tisch" in output

    # Auch mit --cheat darf es nicht crashen und muss zusätzliche Infos zeigen.
    cheat_output = render_state(state, human_player=0, cheat=True)
    assert "CHEAT" in cheat_output
