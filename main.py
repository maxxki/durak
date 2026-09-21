"""Durak-KI, Stufe 4: Terminal-UI (Mensch vs. ISMCTS-Agent).

Interaktives Terminal-Spiel gegen den ISMCTS-Agenten aus Stufe 2a. Reines
stdin/stdout, keine GUI, keine Farbbibliotheken.

Start (Fedora Toolbox):

    toolbox create --distro fedora --release 40 durak-dev
    toolbox enter durak-dev
    pip install -r requirements.txt
    python main.py --seed 42

CLI-Argumente:

    python main.py --seed 42 --time-budget 2.0 --human-player 0 --cheat

    --seed INT           Seed für initial_state und Agent (Default 0)
    --time-budget FLOAT  Sekunden pro KI-Zug (Default 2.0)
    --human-player {0,1} Sitzplatz des Menschen (Default 0)
    --cheat              Debug-Modus: zeigt Gegnerhand und Deck an

Im Spiel:

    a <n>   Karte <n> angreifen
    d <n>   Karte <n> verteidigen
    t <n>   Transfer mit Karte <n>
    take    aufnehmen
    pass    Runde abgeben
    help    Befehlsübersicht
    quit    Spiel beenden
"""

from __future__ import annotations

import argparse
import sys

from durak_env import Action, Card, State, initial_state, legal_actions, step, winner
from ismcts_agent import ISMCTSAgent

__all__ = [
    "CommandError",
    "parse_command",
    "format_card",
    "get_actor",
    "human_hand",
    "build_action",
    "describe_agent_action",
    "render_state",
    "prompt_human_action",
    "play",
    "main",
]

SUIT_SYMBOLS = {"S": "\u2660", "H": "\u2665", "D": "\u2666", "C": "\u2663"}  # ♠ ♥ ♦ ♣
SUIT_NAMES = {"S": "Pik", "H": "Herz", "D": "Karo", "C": "Kreuz"}

_INDEXED_TYPES = {"a": "ATTACK", "d": "DEFEND", "t": "TRANSFER"}

HELP_TEXT = """\
Befehle:
  a <n>   Karte <n> angreifen         (z. B. 'a 1')
  d <n>   Karte <n> verteidigen       (z. B. 'd 3')
  t <n>   Transfer mit Karte <n>      (z. B. 't 2')
  take    aufnehmen
  pass    Runde abgeben (nur wenn legal)
  help    diese Übersicht anzeigen
  quit    Spiel beenden
<n> ist die Nummer der Karte in deiner Hand (siehe Anzeige oben)."""


# --------------------------------------------------------------------------
# Fehler
# --------------------------------------------------------------------------


class CommandError(ValueError):
    """Eingabe konnte nicht geparst oder nicht auf eine legale Aktion abgebildet werden."""


# --------------------------------------------------------------------------
# Eingabe-Parsing
# --------------------------------------------------------------------------


def parse_command(text: str) -> tuple[str, int | None]:
    """Parst eine Zeile Benutzereingabe in ``(TYP, index_oder_None)``.

    TYP ist eines von ``ATTACK``, ``DEFEND``, ``TRANSFER``, ``TAKE``,
    ``PASS``, ``HELP``, ``QUIT``. Wirft ``CommandError`` mit einer
    verständlichen deutschen Meldung bei unbekannten oder falsch
    geformten Eingaben.
    """
    stripped = text.strip()
    if not stripped:
        raise CommandError("Bitte einen Befehl eingeben (z. B. 'a 1', 'pass', 'help').")

    parts = stripped.split()
    head = parts[0].lower()

    if head in ("help", "?"):
        return ("HELP", None)
    if head == "quit":
        return ("QUIT", None)
    if head == "take":
        return ("TAKE", None)
    if head == "pass":
        return ("PASS", None)

    if head in _INDEXED_TYPES:
        if len(parts) != 2:
            raise CommandError(f"'{head}' erwartet genau eine Kartennummer, z. B. '{head} 1'.")
        try:
            idx = int(parts[1])
        except ValueError:
            raise CommandError(f"'{parts[1]}' ist keine gültige Kartennummer.")
        return (_INDEXED_TYPES[head], idx)

    raise CommandError(f"Unbekannter Befehl: '{stripped}'. Gib 'help' ein für eine Übersicht.")


# --------------------------------------------------------------------------
# Karten-Darstellung
# --------------------------------------------------------------------------


def format_card(card: Card) -> str:
    """('A', 'S') -> 'A♠', ('10', 'H') -> '10♥'."""
    rank, suit = card
    return f"{rank}{SUIT_SYMBOLS[suit]}"


# --------------------------------------------------------------------------
# Rollen- und Hand-Hilfen
# --------------------------------------------------------------------------


def get_actor(state: State) -> int:
    """Index des gerade handelnden Spielers (siehe durak_env-Moduldoc)."""
    return state.attacker_idx if state.phase == "ATTACK" else 1 - state.attacker_idx


def human_hand(state: State, human_player: int) -> tuple[Card, ...]:
    """Die Hand des Menschen, unabhängig davon, wer gerade am Zug ist."""
    if state.attacker_idx == human_player:
        return state.hand_attacker
    return state.hand_defender


def _opponent_hand(state: State, human_player: int) -> tuple[Card, ...]:
    if state.attacker_idx == human_player:
        return state.hand_defender
    return state.hand_attacker


# --------------------------------------------------------------------------
# Eingabe -> Action
# --------------------------------------------------------------------------


def build_action(cmd: tuple[str, int | None], state: State, human_player: int) -> Action:
    """Bildet einen geparsten Befehl auf eine legale ``Action`` ab.

    Wirft ``CommandError``, wenn die Karten-Nummer außerhalb der Hand liegt
    oder wenn die resultierende Aktion gerade nicht in ``legal_actions``
    steht -- ``step`` wird hier nie aufgerufen, also nie mit einer illegalen
    Aktion erreicht.
    """
    ctype, idx = cmd

    if ctype in ("ATTACK", "DEFEND", "TRANSFER"):
        hand = human_hand(state, human_player)
        if idx is None or not (1 <= idx <= len(hand)):
            raise CommandError(
                f"Karte {idx} ist nicht auf deiner Hand. Verfügbar: 1-{len(hand)}."
            )
        action: Action = (ctype, hand[idx - 1])
    elif ctype in ("TAKE", "PASS"):
        action = (ctype, None)
    else:
        raise CommandError(f"'{ctype}' ist kein spielbarer Befehl hier.")

    legal = legal_actions(state)
    if action not in legal:
        allowed = sorted({a[0] for a in legal})
        raise CommandError(
            f"Diese Aktion ist gerade nicht erlaubt. Erlaubte Aktionstypen: "
            f"{', '.join(allowed)}."
        )
    return action


# --------------------------------------------------------------------------
# KI-Zug-Anzeige
# --------------------------------------------------------------------------


def describe_agent_action(action: Action) -> str:
    atype, card = action
    if atype == "ATTACK":
        return f"KI greift mit {format_card(card)} an."
    if atype == "DEFEND":
        return f"KI verteidigt mit {format_card(card)}."
    if atype == "TRANSFER":
        return f"KI transferiert mit {format_card(card)}."
    if atype == "TAKE":
        return "KI nimmt auf."
    if atype == "PASS":
        return "KI passt."
    return f"KI spielt {atype}."  # Sicherheitsnetz, sollte nie erreicht werden.


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_state(state: State, human_player: int, cheat: bool = False) -> str:
    lines: list[str] = []
    actor = get_actor(state)
    is_human_turn = actor == human_player
    role_label = "Angreifer" if state.attacker_idx == human_player else "Verteidiger"

    lines.append("=" * 64)
    if state.phase == "GAME_OVER":
        lines.append("Durak -- Spielende")
    else:
        turn_label = "Dein Zug" if is_human_turn else "Zug der KI"
        lines.append(f"Durak -- {turn_label} (Du bist {role_label})")
    lines.append("=" * 64)
    lines.append("")

    suit = state.trump_suit
    if state.trump_card is not None:
        lines.append(
            f"Trumpf: {SUIT_SYMBOLS[suit]} {SUIT_NAMES[suit]} "
            f"(offene Karte: {format_card(state.trump_card)})"
        )
    else:
        lines.append(f"Trumpf: {SUIT_SYMBOLS[suit]} {SUIT_NAMES[suit]} (bereits gezogen)")
    lines.append(f"Deck: {len(state.deck)} Karten verbleibend")

    own_hand = human_hand(state, human_player)
    opp_hand = _opponent_hand(state, human_player)
    lines.append(f"Gegner: {len(opp_hand)} Karten auf der Hand")
    lines.append("")

    lines.append("Tisch:")
    if state.table_attack:
        attack_str = " ".join(format_card(c) for c in state.table_attack)
        defense_str = " ".join(
            format_card(c) if c is not None else "--" for c in state.table_defense
        )
        lines.append(f"[Angriff] {attack_str}")
        lines.append(f"[Abwehr ] {defense_str}")
    else:
        lines.append("(leer)")
    lines.append("")

    lines.append(f"Deine Hand ({len(own_hand)} Karten):")
    parts = []
    for i, card in enumerate(own_hand, start=1):
        marker = "*" if card[1] == state.trump_suit else ""
        parts.append(f"{i}) {format_card(card)}{marker}")
    lines.append("    " + "  ".join(parts))
    lines.append("")

    if cheat:
        lines.append("[CHEAT] Gegnerhand: " + ", ".join(format_card(c) for c in opp_hand))
        lines.append(
            "[CHEAT] Deck: "
            + (", ".join(format_card(c) for c in state.deck) if state.deck else "(leer)")
        )
        lines.append("")

    if is_human_turn and state.phase != "GAME_OVER":
        lines.append("Aktionen:")
        lines.append("  a <n>   Karte <n> angreifen")
        lines.append("  d <n>   Karte <n> verteidigen")
        lines.append("  t <n>   Transfer mit Karte <n>")
        lines.append("  take    aufnehmen")
        lines.append("  pass    Runde abgeben")
        lines.append("  help    Befehlsübersicht")
        lines.append("  quit    Spiel beenden")
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Mensch-Eingabeschleife
# --------------------------------------------------------------------------


def prompt_human_action(state: State, human_player: int) -> Action:
    """Fragt so lange nach Eingabe, bis eine legale Aktion vorliegt.

    Reicht nie eine illegale Aktion an step() weiter und nie einen rohen
    ValueError aus durak_env nach außen -- jede Ablehnung erzeugt eine
    klare deutsche Fehlermeldung und eine erneute Nachfrage.
    """
    while True:
        try:
            raw = input("    > ")
        except EOFError:
            print("\nEingabe beendet. Spiel abgebrochen.")
            sys.exit(0)

        try:
            cmd = parse_command(raw)
        except CommandError as exc:
            print(f"Fehler: {exc}")
            continue

        ctype, _idx = cmd
        if ctype == "HELP":
            print(HELP_TEXT)
            continue
        if ctype == "QUIT":
            print("Spiel beendet.")
            sys.exit(0)

        try:
            return build_action(cmd, state, human_player)
        except CommandError as exc:
            print(f"Fehler: {exc}")
            continue


# --------------------------------------------------------------------------
# Spielschleife
# --------------------------------------------------------------------------


def play(seed: int, time_budget: float, human_player: int, cheat: bool) -> None:
    state = initial_state(seed=seed)
    agent = ISMCTSAgent(player_idx=1 - human_player, time_budget=time_budget, seed=seed)
    plies = 0

    while state.phase != "GAME_OVER":
        print(render_state(state, human_player, cheat=cheat))

        actor = get_actor(state)
        if actor == human_player:
            action = prompt_human_action(state, human_player)
        else:
            action = agent.choose_action(state)
            print(describe_agent_action(action))
            print()

        state, _reward, done = step(state, action)
        plies += 1

    winner_idx = winner(state)
    print("=" * 64)
    if winner_idx == human_player:
        print(f"Du hast gewonnen! ({plies} Züge)")
    elif winner_idx is not None:
        print(f"Der Durak bist du! Die KI hat gewonnen. ({plies} Züge)")
    else:
        print(f"Kein Gewinner ermittelbar (Abbruch). ({plies} Züge)")
    print("=" * 64)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Durak -- Mensch gegen ISMCTS-Agent")
    parser.add_argument("--seed", type=int, default=0, help="Seed für initial_state und Agent")
    parser.add_argument(
        "--time-budget", type=float, default=2.0, help="Sekunden pro KI-Zug"
    )
    parser.add_argument(
        "--human-player", type=int, choices=(0, 1), default=0, help="Sitzplatz des Menschen"
    )
    parser.add_argument(
        "--cheat", action="store_true", help="Debug: zeigt Gegnerhand und Deck an"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    try:
        play(
            seed=args.seed,
            time_budget=args.time_budget,
            human_player=args.human_player,
            cheat=args.cheat,
        )
    except KeyboardInterrupt:
        print("\nSpiel abgebrochen.")


if __name__ == "__main__":
    main()
