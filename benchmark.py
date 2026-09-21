"""Durak-KI, Stufe 3: Benchmark-Framework (2a-Baseline + Vorbereitung 2b-Vergleich).

Siehe ``stufe_3_prompt.md`` für den vollständigen Kontext. Kurzfassung:

* ``play_game``/``run_benchmark`` sind generisch über die Spieler-Rolle:
  ein Spieler ist entweder ein ``ISMCTSAgent`` (hat ``choose_action``) oder
  eine ``Policy``-Funktion ``(state, rng) -> Action`` wie ``random_policy``.
* ``run_benchmark`` koppelt für einen gegebenen Seed den Startzustand
  (``initial_state``) UND den Zufalls-Policy-RNG deterministisch, damit zwei
  Benchmarks mit denselben Seeds (z. B. 2a vs. Random und 2b vs. Random)
  paarweise vergleichbar sind (siehe ``paired_comparison``).
* Statistik: Wilson-Score-Konfidenzintervall (robust nahe 0/1, im Gegensatz
  zur Normalapproximation) und ein exakter McNemar-Test für gepaarte
  Vergleiche (Chi²/df=1 über ``math.erfc``, keine scipy-Abhängigkeit).
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Union

from durak_env import Action, State, initial_state, legal_actions, step, winner
from ismcts_agent import ISMCTSAgent

import random as random_module  # Modulname vermeiden, das Parameter oft "rng" heißen

__all__ = [
    "Policy",
    "Player",
    "GameResult",
    "BenchmarkResult",
    "PairedComparisonResult",
    "random_policy",
    "play_game",
    "run_benchmark",
    "wilson_ci_95",
    "paired_comparison",
]

Policy = Callable[[State, "random_module.Random"], Action]
Player = Union[ISMCTSAgent, Policy]

_Z_95 = 1.959963985  # 97.5%-Quantil der Standardnormalverteilung


# --------------------------------------------------------------------------
# Spieler-Abstraktion
# --------------------------------------------------------------------------


def random_policy(state: State, rng: "random_module.Random") -> Action:
    """Uniform zufällige Policy über ``legal_actions(state)``."""
    legal = legal_actions(state)
    return legal[rng.randrange(len(legal))]


def _is_agent(player: Player) -> bool:
    return hasattr(player, "choose_action")


def _choose(player: Player, state: State, rng: "random_module.Random") -> Action:
    if _is_agent(player):
        return player.choose_action(state)  # type: ignore[union-attr]
    return player(state, rng)  # type: ignore[operator]


def _actor(state: State) -> int:
    """Index des gerade handelnden Spielers."""
    return state.attacker_idx if state.phase == "ATTACK" else 1 - state.attacker_idx


# --------------------------------------------------------------------------
# Eine Partie
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GameResult:
    seed: int
    winner: int | None  # 0, 1, oder None bei Abbruch durch max_plies
    plies: int
    player0_move_times: tuple[float, ...]
    player1_move_times: tuple[float, ...]


def play_game(
    seed: int,
    player0: Player,
    player1: Player,
    opponent_rng_seed: int,
    max_plies: int = 2000,
) -> GameResult:
    """Spielt eine volle Partie zwischen ``player0`` (Sitzplatz 0) und
    ``player1`` (Sitzplatz 1). ``opponent_rng_seed`` speist den RNG, der an
    Policy-Funktionen (nicht an ``ISMCTSAgent``-Instanzen, die ihren eigenen
    Seed im Konstruktor bekommen) durchgereicht wird.
    """
    state = initial_state(seed=seed)
    rng = random_module.Random(opponent_rng_seed)
    p0_times: list[float] = []
    p1_times: list[float] = []
    plies = 0
    aborted = False

    while True:
        legal = legal_actions(state)
        if not legal:
            break
        if plies >= max_plies:
            aborted = True
            break

        actor = _actor(state)
        player = player0 if actor == 0 else player1
        times_list = p0_times if actor == 0 else p1_times

        t0 = time.monotonic()
        action = _choose(player, state, rng)
        dt = time.monotonic() - t0
        if _is_agent(player):
            times_list.append(dt)

        state, _reward, done = step(state, action)
        plies += 1
        if done:
            break

    game_winner = None if aborted else winner(state)
    return GameResult(
        seed=seed,
        winner=game_winner,
        plies=plies,
        player0_move_times=tuple(p0_times),
        player1_move_times=tuple(p1_times),
    )


# --------------------------------------------------------------------------
# Mehrere Partien
# --------------------------------------------------------------------------


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def wilson_ci_95(successes: int, n: int) -> tuple[float, float]:
    """Wilson-Score-95%-Konfidenzintervall für eine Erfolgsquote.

    Robust nahe 0 oder 1 (im Gegensatz zur Normalapproximation, deren
    Intervallgrenzen dort außerhalb von [0, 1] liegen können).
    """
    if n == 0:
        return (0.0, 0.0)
    z = _Z_95
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    halfwidth = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    lo = max(0.0, center - halfwidth)
    hi = min(1.0, center + halfwidth)
    return (lo, hi)


@dataclass(frozen=True)
class BenchmarkResult:
    games: tuple[GameResult, ...]
    wins_player0: int
    win_rate_player0: float
    wilson_ci_95: tuple[float, float]
    avg_plies: float
    avg_move_time_player0: float
    p95_move_time_player0: float


def run_benchmark(
    n_games: int,
    make_player0: Callable[[int], Player],
    make_player1: Callable[[int], Player],
    base_seed: int = 0,
    max_plies: int = 2000,
) -> BenchmarkResult:
    """Spielt ``n_games`` Partien mit Seeds ``base_seed .. base_seed+n_games-1``.

    Für jeden Seed ``s`` werden ``make_player0(s)``/``make_player1(s)``
    genau einmal aufgerufen (frische Spieler-Instanz pro Partie, kein
    geteilter RNG-Zustand über Partien hinweg). ``opponent_rng_seed`` wird
    deterministisch aus ``base_seed`` und ``s`` abgeleitet, damit zwei
    ``run_benchmark``-Aufrufe mit denselben Seeds (gleicher ``base_seed``,
    gleiches ``n_games``) für Policy-Spieler exakt dieselben Zufallszüge
    erzeugen -- Voraussetzung für ``paired_comparison``.
    """
    games: list[GameResult] = []
    for i in range(n_games):
        s = base_seed + i
        player0 = make_player0(s)
        player1 = make_player1(s)
        opponent_rng_seed = base_seed + s
        games.append(play_game(s, player0, player1, opponent_rng_seed, max_plies=max_plies))

    games_t = tuple(games)
    n = len(games_t)
    wins0 = sum(1 for g in games_t if g.winner == 0)
    win_rate = wins0 / n if n else 0.0
    ci = wilson_ci_95(wins0, n)
    avg_plies = statistics.fmean(g.plies for g in games_t) if games_t else 0.0

    all_p0_times = [t for g in games_t for t in g.player0_move_times]
    avg_move_time = statistics.fmean(all_p0_times) if all_p0_times else 0.0
    p95_move_time = _percentile(all_p0_times, 0.95)

    return BenchmarkResult(
        games=games_t,
        wins_player0=wins0,
        win_rate_player0=win_rate,
        wilson_ci_95=ci,
        avg_plies=avg_plies,
        avg_move_time_player0=avg_move_time,
        p95_move_time_player0=p95_move_time,
    )


# --------------------------------------------------------------------------
# Gepaarter A/B-Vergleich (McNemar) -- Vorbereitung für Stufe 2b
# --------------------------------------------------------------------------


def _chi2_sf_df1(x: float) -> float:
    """P(Chi2_1 > x), exakt über die Standardnormalverteilung: für
    df=1 gilt P(Chi2_1 > x) = 2*(1 - Phi(sqrt(x))) = erfc(sqrt(x/2)).
    """
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


@dataclass(frozen=True)
class PairedComparisonResult:
    n_pairs: int
    b: int  # A gewann, B verlor (bei gleichem Seed)
    c: int  # B gewann, A verlor (bei gleichem Seed)
    statistic: float
    p_value: float
    note: str | None = None


def paired_comparison(result_a: BenchmarkResult, result_b: BenchmarkResult) -> PairedComparisonResult:
    """McNemar-Test über gepaarte Partien (gleicher Seed in A und B).

    Voraussetzung: ``result_a`` und ``result_b`` wurden mit identischen
    Seed-Mengen erzeugt (siehe ``run_benchmark``: gleicher ``base_seed`` und
    ``n_games``), sonst ist der Vergleich nicht sinnvoll gepaart.
    """
    seeds_a = {g.seed for g in result_a.games}
    seeds_b = {g.seed for g in result_b.games}
    if seeds_a != seeds_b:
        only_a = sorted(seeds_a - seeds_b)
        only_b = sorted(seeds_b - seeds_a)
        raise ValueError(
            "paired_comparison erfordert identische Seed-Mengen in result_a "
            f"und result_b; nur in A: {only_a}, nur in B: {only_b}."
        )

    by_seed_a = {g.seed: g for g in result_a.games}
    by_seed_b = {g.seed: g for g in result_b.games}

    b = 0
    c = 0
    for seed in seeds_a:
        a_won = by_seed_a[seed].winner == 0
        b_won = by_seed_b[seed].winner == 0
        if a_won and not b_won:
            b += 1
        elif b_won and not a_won:
            c += 1

    n_pairs = len(seeds_a)
    if b + c == 0:
        return PairedComparisonResult(
            n_pairs=n_pairs,
            b=b,
            c=c,
            statistic=0.0,
            p_value=1.0,
            note="Keine diskordanten Paare; McNemar-Test nicht anwendbar.",
        )

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = _chi2_sf_df1(statistic)
    return PairedComparisonResult(n_pairs=n_pairs, b=b, c=c, statistic=statistic, p_value=p_value)
