"""Durak-KI, Stufe 3: Produktionslauf der 2a-vs-Random-Baseline.

Spielt 100 Partien ``ISMCTSAgent`` (Stufe 2a, ``time_budget=2.0``) gegen
``random_policy``, mit ``player_idx`` abwechselnd 0/1 über die Seeds verteilt
(damit die Baseline nicht nur "Agent beginnt immer als Angreifer" misst).
Schreibt das Ergebnis als JSON nach ``baseline_2a_vs_random.json`` -- diese
Datei ist die Referenz, gegen die Stufe 2b später ``paired_comparison``
aufruft.

Laufzeit: bei ~45 Agentenzügen/Partie x 2,0 s sind das grob 1,5-2,5 Stunden
für 100 Partien. Wird NICHT als Teil der pytest-Suite ausgeführt -- separat
starten, z. B.:

    nohup python run_baseline.py > run_baseline.log 2>&1 &
"""

from __future__ import annotations

import json
import sys
import time

from benchmark import BenchmarkResult, GameResult, random_policy, run_benchmark
from ismcts_agent import ISMCTSAgent

N_GAMES = 100
TIME_BUDGET = 2.0
BASE_SEED = 0
MAX_PLIES = 2000
OUTPUT_PATH = "baseline_2a_vs_random.json"


def _make_agent(seed: int) -> ISMCTSAgent:
    """player_idx alterniert über die Seeds (gerade -> 0, ungerade -> 1),
    damit der Agent nicht in jeder Partie als Angreifer startet.
    """
    player_idx = seed % 2
    return ISMCTSAgent(player_idx=player_idx, time_budget=TIME_BUDGET, seed=seed)


def _make_random(seed: int):
    # random_policy ist zustandslos (der RNG kommt separat über
    # opponent_rng_seed in run_benchmark/play_game) -- dieselbe Funktion
    # kann für jeden Seed unverändert zurückgegeben werden.
    return random_policy


def _player0_is_agent(seed: int) -> bool:
    return seed % 2 == 0


def _game_to_dict(game: GameResult, seed: int) -> dict:
    agent_idx = 0 if _player0_is_agent(seed) else 1
    agent_move_times = (
        game.player0_move_times if agent_idx == 0 else game.player1_move_times
    )
    return {
        "seed": game.seed,
        "winner": game.winner,
        "plies": game.plies,
        "agent_player_idx": agent_idx,
        "agent_won": game.winner == agent_idx,
        "agent_move_times": list(agent_move_times),
    }


def _result_to_dict(result: BenchmarkResult, elapsed_seconds: float) -> dict:
    # Hinweis: "wins_player0" aus run_benchmark zählt Siege von player0 in
    # dessen fester Sitzrolle, NICHT Siege des Agenten (der alterniert
    # zwischen Rolle 0 und 1). Für die eigentliche Kennzahl wird deshalb
    # separat über agent_won aus den Einzelpartien gezählt.
    agent_wins = sum(1 for g in result.games if g.winner == (g.seed % 2))
    n = len(result.games)
    from benchmark import _percentile, wilson_ci_95

    agent_win_rate = agent_wins / n if n else 0.0
    agent_ci = wilson_ci_95(agent_wins, n)

    # result.avg_move_time_player0/p95_move_time_player0 beziehen sich auf
    # den FESTEN Sitzplatz 0 -- der hier aber zwischen Agent und Random
    # alterniert (siehe _player0_is_agent). Für die Agenten-Zugzeit muss
    # daher pro Partie die jeweils richtige Seite ausgewählt werden, nicht
    # das vorgefertigte Feld aus BenchmarkResult verwendet werden.
    agent_move_times: list[float] = []
    for g in result.games:
        times = g.player0_move_times if _player0_is_agent(g.seed) else g.player1_move_times
        agent_move_times.extend(times)
    avg_move_time_agent = (
        sum(agent_move_times) / len(agent_move_times) if agent_move_times else 0.0
    )
    p95_move_time_agent = _percentile(agent_move_times, 0.95)

    return {
        "config": {
            "n_games": N_GAMES,
            "time_budget": TIME_BUDGET,
            "base_seed": BASE_SEED,
            "max_plies": MAX_PLIES,
            "opponent": "random_policy",
            "player_idx_alternates": True,
        },
        "summary": {
            "agent_wins": agent_wins,
            "agent_win_rate": agent_win_rate,
            "agent_wilson_ci_95": list(agent_ci),
            "avg_plies": result.avg_plies,
            "avg_move_time_agent": avg_move_time_agent,
            "p95_move_time_agent": p95_move_time_agent,
            "elapsed_seconds": elapsed_seconds,
        },
        "games": [_game_to_dict(g, g.seed) for g in result.games],
    }


def main() -> None:
    print(
        f"Starte Produktionslauf: {N_GAMES} Partien, time_budget={TIME_BUDGET}s, "
        f"base_seed={BASE_SEED}. Erwartete Laufzeit: ~1,5-2,5 Stunden.",
        file=sys.stderr,
    )
    start = time.monotonic()

    # run_benchmark erwartet make_player0/make_player1 fest an Sitzplatz 0/1
    # gebunden. Um player_idx über die Seeds alternieren zu lassen (Agent
    # mal Sitz 0, mal Sitz 1), wird hier direkt zwischen Agent und Random je
    # nach Seed-Parität zugewiesen, statt run_benchmark mit festen Rollen zu
    # verwenden.
    def make_player0(seed: int):
        return _make_agent(seed) if _player0_is_agent(seed) else _make_random(seed)

    def make_player1(seed: int):
        return _make_random(seed) if _player0_is_agent(seed) else _make_agent(seed)

    result = run_benchmark(
        n_games=N_GAMES,
        make_player0=make_player0,
        make_player1=make_player1,
        base_seed=BASE_SEED,
        max_plies=MAX_PLIES,
    )

    elapsed = time.monotonic() - start
    payload = _result_to_dict(result, elapsed)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary = payload["summary"]
    print(
        f"Fertig in {elapsed / 60:.1f} min. "
        f"Agent-Winrate: {summary['agent_win_rate']:.1%} "
        f"(95%-CI: {summary['agent_wilson_ci_95'][0]:.1%}-"
        f"{summary['agent_wilson_ci_95'][1]:.1%}), "
        f"{summary['agent_wins']}/{N_GAMES} Siege. "
        f"Geschrieben nach {OUTPUT_PATH}.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
