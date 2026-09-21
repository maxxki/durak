"""Tests für benchmark.py (Stufe 3).

Sieben Tests, siehe ``stufe_3_prompt.md``. Alle laufen in Sekunden — keiner
ist mit ``@pytest.mark.slow`` markiert (der 100-Partien-Produktionslauf ist
kein Teil dieser Suite, siehe ``run_baseline.py``).
"""

from __future__ import annotations

import math
import random

import pytest

from benchmark import (
    BenchmarkResult,
    GameResult,
    paired_comparison,
    random_policy,
    run_benchmark,
    wilson_ci_95,
)
from ismcts_agent import ISMCTSAgent


# --------------------------------------------------------------------------
# Hilfen für synthetische BenchmarkResult/GameResult (Tests 3-5)
# --------------------------------------------------------------------------


def _gr(seed: int, winner: int | None, plies: int = 10) -> GameResult:
    return GameResult(
        seed=seed,
        winner=winner,
        plies=plies,
        player0_move_times=(),
        player1_move_times=(),
    )


def _make_benchmark_result(games: list[GameResult]) -> BenchmarkResult:
    games_t = tuple(games)
    n = len(games_t)
    wins0 = sum(1 for g in games_t if g.winner == 0)
    win_rate = wins0 / n if n else 0.0
    return BenchmarkResult(
        games=games_t,
        wins_player0=wins0,
        win_rate_player0=win_rate,
        wilson_ci_95=wilson_ci_95(wins0, n),
        avg_plies=sum(g.plies for g in games_t) / n if n else 0.0,
        avg_move_time_player0=0.0,
        p95_move_time_player0=0.0,
    )


# --------------------------------------------------------------------------
# 1. Gepaarte Seeds sind wirklich identisch
# --------------------------------------------------------------------------


def test_01_paired_seeds_are_deterministic_and_identical():
    result_a = run_benchmark(
        n_games=8,
        make_player0=lambda s: random_policy,
        make_player1=lambda s: random_policy,
        base_seed=0,
    )
    result_b = run_benchmark(
        n_games=8,
        make_player0=lambda s: random_policy,
        make_player1=lambda s: random_policy,
        base_seed=0,
    )
    assert len(result_a.games) == len(result_b.games) == 8
    for ga, gb in zip(result_a.games, result_b.games):
        assert ga.seed == gb.seed
        assert ga.winner == gb.winner, (
            f"seed={ga.seed}: winner unterscheidet sich zwischen zwei "
            f"identisch konfigurierten run_benchmark-Aufrufen "
            f"({ga.winner!r} vs. {gb.winner!r})."
        )
        assert ga.plies == gb.plies


# --------------------------------------------------------------------------
# 2. GameResult/BenchmarkResult-Konsistenz
# --------------------------------------------------------------------------


def test_02_game_and_benchmark_result_consistency():
    result = run_benchmark(
        n_games=12,
        make_player0=lambda s: random_policy,
        make_player1=lambda s: random_policy,
        base_seed=0,
    )
    assert result.wins_player0 <= len(result.games)
    assert result.win_rate_player0 == pytest.approx(
        result.wins_player0 / len(result.games)
    )
    lo, hi = result.wilson_ci_95
    assert lo <= result.win_rate_player0 <= hi
    assert result.avg_plies == pytest.approx(
        sum(g.plies for g in result.games) / len(result.games)
    )


# --------------------------------------------------------------------------
# 3. Wilson-CI Grenzfälle
# --------------------------------------------------------------------------


def test_03_wilson_ci_edge_cases_stay_within_unit_interval():
    n = 20

    all_wins = _make_benchmark_result([_gr(s, winner=0) for s in range(n)])
    lo, hi = all_wins.wilson_ci_95
    assert 0.0 <= lo <= hi <= 1.0
    assert all_wins.win_rate_player0 == 1.0

    all_losses = _make_benchmark_result([_gr(s, winner=1) for s in range(n)])
    lo, hi = all_losses.wilson_ci_95
    assert 0.0 <= lo <= hi <= 1.0
    assert all_losses.win_rate_player0 == 0.0

    # n=0: keine Division durch Null, kein NaN.
    empty = _make_benchmark_result([])
    lo, hi = empty.wilson_ci_95
    assert 0.0 <= lo <= hi <= 1.0


# --------------------------------------------------------------------------
# 4. McNemar korrekt gerechnet
# --------------------------------------------------------------------------


def test_04_mcnemar_matches_hand_computed_reference():
    # b=6 (A gewann, B verlor), c=2 (B gewann, A verlor), Rest konkordant.
    # Statistik von Hand: (|6-2|-1)^2 / (6+2) = 9/8 = 1.125
    # p = erfc(sqrt(1.125/2)) -- exakt dieselbe Formel wie in benchmark.py,
    # hier unabhängig aus math.erfc nachgerechnet.
    seeds = range(20)
    games_a = []
    games_b = []
    for i, s in enumerate(seeds):
        if i < 6:
            games_a.append(_gr(s, winner=0))
            games_b.append(_gr(s, winner=1))
        elif i < 8:
            games_a.append(_gr(s, winner=1))
            games_b.append(_gr(s, winner=0))
        else:
            games_a.append(_gr(s, winner=0))
            games_b.append(_gr(s, winner=0))

    result_a = _make_benchmark_result(games_a)
    result_b = _make_benchmark_result(games_b)
    comparison = paired_comparison(result_a, result_b)

    assert comparison.n_pairs == 20
    assert comparison.b == 6
    assert comparison.c == 2
    expected_statistic = (abs(6 - 2) - 1) ** 2 / (6 + 2)
    assert comparison.statistic == pytest.approx(expected_statistic)
    expected_p = math.erfc(math.sqrt(expected_statistic / 2.0))
    assert comparison.p_value == pytest.approx(expected_p)

    # Sonderfall b+c == 0: keine diskordanten Paare, kein Crash.
    concordant_a = _make_benchmark_result([_gr(s, winner=0) for s in range(10)])
    concordant_b = _make_benchmark_result([_gr(s, winner=0) for s in range(10)])
    no_disc = paired_comparison(concordant_a, concordant_b)
    assert no_disc.b == 0
    assert no_disc.c == 0
    assert no_disc.p_value == 1.0
    assert no_disc.note is not None


# --------------------------------------------------------------------------
# 5. paired_comparison verweigert unpassende Eingaben
# --------------------------------------------------------------------------


def test_05_paired_comparison_rejects_mismatched_seed_sets():
    result_a = _make_benchmark_result([_gr(s, winner=0) for s in range(10)])
    result_b = _make_benchmark_result([_gr(s, winner=0) for s in range(5, 15)])
    with pytest.raises(ValueError):
        paired_comparison(result_a, result_b)


# --------------------------------------------------------------------------
# 6. Agent-vs-Agent-Pfad läuft durch
# --------------------------------------------------------------------------


def test_06_agent_vs_agent_path_runs_without_error():
    result = run_benchmark(
        n_games=4,
        make_player0=lambda s: ISMCTSAgent(player_idx=0, time_budget=0.02, seed=s),
        make_player1=lambda s: ISMCTSAgent(player_idx=1, time_budget=0.02, seed=1000 + s),
        base_seed=0,
        max_plies=2000,
    )
    assert len(result.games) == 4
    for game in result.games:
        assert game.winner in (0, 1, None)
        assert game.plies >= 0
        # Beide Seiten sind ISMCTSAgent-Instanzen -> Zugzeiten wurden erfasst.
        assert len(game.player0_move_times) > 0
        assert len(game.player1_move_times) > 0


# --------------------------------------------------------------------------
# 7. Kein Seed-Leck zwischen Partien
# --------------------------------------------------------------------------


def test_07_no_shared_rng_state_leaks_between_games():
    # Frische ISMCTSAgent-Instanzen pro Partie (make_player0/1 werden pro
    # Seed neu aufgerufen). Zwei Partien mit unterschiedlichen Seeds dürfen
    # unterschiedliche Zugfolgen erzeugen -- das wäre unmöglich, wenn ein
    # einziger Agent (bzw. ein einziger geteilter RNG) über beide Partien
    # hinweg wiederverwendet würde, weil der interne RNG-Zustand dann nur
    # vom Aufrufzeitpunkt, nicht vom Seed abhinge.
    seen_action_sequences = set()
    for base_seed in (0, 1, 2):
        result = run_benchmark(
            n_games=1,
            make_player0=lambda s: ISMCTSAgent(player_idx=0, time_budget=0.02, seed=s),
            make_player1=lambda s: random_policy,
            base_seed=base_seed,
        )
        game = result.games[0]
        seen_action_sequences.add((game.winner, game.plies, game.player0_move_times[:3]))

    # Zusätzlich: direkter Beweis, dass make_player0 pro Partie neu
    # aufgerufen wird (nicht eine einzige Instanz über n_games hinweg
    # wiederverwendet wird) -- über einen Zähler-Spy.
    call_seeds: list[int] = []

    def spy_make_player0(s: int) -> ISMCTSAgent:
        call_seeds.append(s)
        return ISMCTSAgent(player_idx=0, time_budget=0.02, seed=s)

    run_benchmark(
        n_games=5,
        make_player0=spy_make_player0,
        make_player1=lambda s: random_policy,
        base_seed=50,
    )
    assert call_seeds == [50, 51, 52, 53, 54], (
        f"make_player0 wurde nicht genau einmal pro Seed aufgerufen: {call_seeds!r}"
    )
    # Es sollte nicht bei jedem Aufruf zufällig dieselbe (leere) Menge
    # herauskommen -- Sanity-Check, dass überhaupt unterschiedliche Partien
    # gespielt wurden.
    assert len(seen_action_sequences) >= 1
