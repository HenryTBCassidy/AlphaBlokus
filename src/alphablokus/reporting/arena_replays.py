"""The interactive arena-replay viewer: embedded HTML/CSS/JS template and
its per-generation section builder."""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from alphablokus.registry import instantiate_game

if TYPE_CHECKING:

    import pandas as pd

    from alphablokus.config import RunConfig



_REPLAY_MAX_GENERATIONS = 16
_REPLAY_MAX_GAMES_PER_GEN = 6


def _evenly_sample(values: list[int], n: int) -> list[int]:
    """Pick up to ``n`` values spread evenly across ``values`` (sorted),
    always including the first and last. Returns all of them if ``len <= n``."""
    if n <= 0 or len(values) <= n:
        return list(values)
    idxs = {round(i * (len(values) - 1) / (n - 1)) for i in range(n)}
    return sorted(values[i] for i in idxs)


def build_arena_replays_section(
    df: pd.DataFrame,
    config: RunConfig,
) -> tuple[str, str]:
    """Build the arena-replays UI as **two separate outputs**:

    1. A small **link card** that lives in the main training report —
       points the reader at the standalone replays page rather than
       inlining hundreds of game-board fragments and bloating the report.
    2. A **standalone HTML page** at ``Reporting/arena_replays.html`` with
       the full interactive replay viewer. Each turn card defaults to
       showing only the actual board after the move; clicking the turn
       header expands the top-K candidate panel for that move (the
       "hybrid" navigation Henry asked for — scroll the actuals, expand
       on click for details).

    Returns ``(link_card_html, standalone_html)``. The caller writes the
    standalone HTML to disk and inserts ``link_card_html`` in the main
    report body.

    All replay data is embedded as JSON in the standalone HTML so the
    file stays self-contained — no JS frameworks, no external fetches.
    """
    import json

    from alphablokus.reporting.display import get_renderer

    renderer = get_renderer(config.game)
    game = instantiate_game(config)

    df = df.copy()
    df["generation"] = df["generation"].astype(int)

    # Bound the volume rendered (see module constants above) so long runs
    # don't OOM report generation. Sample generations evenly, cap games/gen.
    all_gens = sorted(df["generation"].unique())
    sampled_gens = _evenly_sample(all_gens, _REPLAY_MAX_GENERATIONS)
    df = df[
        df["generation"].isin(sampled_gens)
        & (df["game_idx"] < _REPLAY_MAX_GAMES_PER_GEN)
    ]
    if len(sampled_gens) < len(all_gens) or _REPLAY_MAX_GAMES_PER_GEN < 50:
        logger.info(
            "Arena replays: rendering {} of {} generations (evenly sampled) "
            "× up to {} games/gen to bound report size",
            len(sampled_gens), len(all_gens), _REPLAY_MAX_GAMES_PER_GEN,
        )

    df = df.sort_values(["generation", "game_idx", "move_idx"])

    games_by_gen: dict[int, list[dict]] = {}

    for (gen, game_idx), group in df.groupby(["generation", "game_idx"]):
        moves = group.sort_values("move_idx")
        first = moves.iloc[0]
        player1_was_white = bool(first["player1_was_white"])

        # Roles: Player 1 = previous best, Player 2 = new candidate (current).
        if player1_was_white:
            role_by_player = {1: "previous net", -1: "new net (this gen)"}
        else:
            role_by_player = {1: "new net (this gen)", -1: "previous net"}

        board = game.initialise_board()  # actual (non-canonical) board state
        turns_html: list[str] = []

        # Colour-name framing: TTT shows literal X/O glyphs on the board so
        # the suffix is informative; Blokus shows piece numbers so it's just
        # noise. Trim it for Blokus.
        if config.game == "tictactoe":
            colour_for_player = {1: "White (X)", -1: "Black (O)"}
        else:
            colour_for_player = {1: "White", -1: "Black"}

        for _, m in moves.iterrows():
            action = int(m["action"])
            player = int(m["player"])
            colour_name = colour_for_player[player]
            role = role_by_player[player]
            player_label = f"{colour_name} — {role}"
            turn_idx = int(m["move_idx"]) + 1

            top_k_actions = [int(a) for a in m["top_k_actions"]]
            top_k_probs = [float(p) for p in m["top_k_probs"]]
            # Defensive: drop any zero-probability entries that older runs
            # may have persisted (pre-fix in evaluation/arena._extract_top_k).
            visited = {
                a: p for a, p in zip(top_k_actions, top_k_probs, strict=False)
                if p > 0
            }
            # Played-action probability: prefer the explicitly-persisted
            # column added in MoveRecord.played_prob; fall back to looking
            # the played action up in the top-K for older parquets.
            if "played_prob" in m and m["played_prob"] is not None:
                played_prob = float(m["played_prob"])
            else:
                played_prob = visited.get(action, 0.0)
            # Candidates panel shows the *alternatives* MCTS considered —
            # the next-best moves the model thought about other than the one
            # it actually played. Surfacing the played action via the
            # caption below it (with its own probability) avoids the
            # "the actual move isn't in the top-3" confusion that happens
            # when many actions tie at the max visit count.
            alternatives = {a: p for a, p in visited.items() if a != action}

            played_caption = _format_played_action_caption(
                game, action, played_prob, colour_name,
            )
            policy_html = renderer.render_policy_html(
                board, alternatives,
                annotation=f"{colour_name}'s top alternatives",
                current_player=player,
            )
            # Apply move and render the resulting state.
            board, _ = game.get_next_state(board, player, action)
            after_html = renderer.render_board_html(
                board, last_action=action, annotation=played_caption,
            )

            turn_card = (
                '<div class="replay-turn">'
                f'<button class="replay-turn-toggle" type="button" '
                f'onclick="alphaBlokus_toggleTurn(this)">'
                f'<span class="replay-turn-label">'
                f'Turn {turn_idx} — {player_label}'
                f'</span>'
                f'<span class="replay-turn-hint">↓ click to show top candidates</span>'
                f'</button>'
                f'<div class="replay-turn-actual">{after_html}</div>'
                f'<div class="replay-turn-candidates" hidden>{policy_html}</div>'
                "</div>"
            )
            turns_html.append(turn_card)

        # Outcome: stored from Player 1's POV (+1 = P1 won, -1 = P2 won).
        outcome = float(first["outcome"])
        if outcome > 0.5:
            winner_colour = "White" if player1_was_white else "Black"
            outcome_label = f"{winner_colour} wins — previous net"
            outcome_class = "result-prev"
        elif outcome < -0.5:
            winner_colour = "Black" if player1_was_white else "White"
            outcome_label = f"{winner_colour} wins — new net"
            outcome_class = "result-new"
        else:
            outcome_label = "Draw"
            outcome_class = "result-draw"

        # Final result banner at the bottom of the replay.
        turns_html.append(
            f'<div class="replay-result">{outcome_label}</div>'
        )

        games_by_gen.setdefault(int(gen), []).append({
            "game_idx": int(game_idx),
            "outcome": outcome,
            "outcome_label": outcome_label,
            "outcome_class": outcome_class,
            "player1_was_white": player1_was_white,
            "turns_html": turns_html,
        })

    payload = json.dumps(games_by_gen)
    gen_options = "\n".join(
        f'<option value="{g}">Generation {g}</option>'
        for g in sorted(games_by_gen)
    )
    first_gen = min(games_by_gen)
    initial_game_options = _render_game_options(games_by_gen[first_gen])

    total_games = sum(len(games) for games in games_by_gen.values())
    standalone_html = _ARENA_REPLAYS_STANDALONE_TEMPLATE.format(
        run_name=config.run_name,
        total_games=total_games,
        num_gens=len(games_by_gen),
        board_css=_blokus_board_css(config),
        gen_options=gen_options,
        initial_game_options=initial_game_options,
        payload=payload,
    )

    link_card = _ARENA_REPLAYS_LINK_CARD.format(
        total_games=total_games,
        num_gens=len(games_by_gen),
    )
    return link_card, standalone_html


def _format_played_action_caption(
    game, action_id: int, played_prob: float, colour_name: str,  # noqa: ARG001
) -> str:
    """Build the annotation for the actual-board panel — describes the move
    in human-readable terms and surfaces its raw MCTS visit probability.

    ``played_prob`` is the share of MCTS visits this action received before
    temperature sampling. With temp=0 the action played is one of the tied
    top-visit options; the displayed probability is the raw visit fraction
    so the reader can see how confident MCTS was relative to alternatives.

    A ``played_prob`` of exactly 0.0 is treated as "unknown" — that's the
    sentinel for older arena-replay parquets persisted before the explicit
    ``MoveRecord.played_prob`` field landed, where the played action may
    have fallen outside the captured top-K.
    """
    prob_suffix = (
        f" — {played_prob * 100:.1f}% of visits"
        if played_prob > 0 else " — visit % not recorded"
    )
    if game.__class__.__name__ == "BlokusDuoGame":
        if game.action_codec.is_pass(action_id):
            return f"Played: PASS{prob_suffix}"
        decoded = game.action_codec.decode(action_id)
        piece = game.piece_manager.pieces[decoded.piece_id]
        return (
            f"Played: Piece {decoded.piece_id} ({piece.name}, "
            f"{decoded.orientation.value}) at "
            f"({decoded.x_coordinate}, {decoded.y_coordinate}){prob_suffix}"
        )
    return f"Played action {action_id}{prob_suffix}"


def _blokus_board_css(config: RunConfig) -> str:
    """Inline the right per-game board CSS into the standalone page."""
    if config.game == "blokusduo":
        from alphablokus.reporting.display_blokusduo import BOARD_CSS
        return BOARD_CSS
    # TTT renders board styles inline via :func:`display_tictactoe`, so the
    # standalone page only needs the shared replay layout CSS.
    return ""


def _render_game_options(games: list[dict]) -> str:
    """Server-side game-dropdown options for the initial generation.

    The ``class`` on each option drives the background colour — green when the
    new (current) net won, red when the previous net won, plain for a draw.
    Browsers vary in how willing they are to style ``<option>`` directly, but
    Chromium/Firefox on desktop both honour it; falls back to no colour
    elsewhere, which is harmless.
    """
    return "\n".join(
        f'<option class="{g["outcome_class"]}" value="{g["game_idx"]}">'
        f'G{g["game_idx"] + 1} — {g["outcome_label"]}'
        f"</option>"
        for g in games
    )



_ARENA_REPLAYS_LINK_CARD = """\
<section>
<h2>Arena Game Replays</h2>
<p class="section-desc">
{total_games} recorded games across {num_gens} generations are available in
the dedicated replay viewer — board-by-board playback with expand-on-click
top-3 candidate previews per move. Pulled out of this report so the
training-metrics view stays focused.
</p>
<a href="arena_replays.html" class="open-replays-button" target="_blank">
  Open arena replay viewer →
</a>
<style>
.open-replays-button {{
    display: inline-block; margin-top: 8px;
    padding: 10px 18px; background: #636efa; color: white;
    text-decoration: none; border-radius: 6px;
    font-size: 14px; font-weight: 600;
}}
.open-replays-button:hover {{
    background: #4a55d4;
}}
</style>
</section>
"""


_ARENA_REPLAYS_STANDALONE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Arena Replays — {run_name}</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1400px; margin: 0 auto; padding: 24px 32px; color: #2a3f5f;
    background: #ffffff;
}}
h1 {{ border-bottom: 2px solid #636efa; padding-bottom: 8px; margin-bottom: 4px; }}
.subtitle {{ color: #6b7280; font-size: 14px; margin-bottom: 24px; }}

.arena-controls {{ margin: 16px 0 24px 0; font-size: 14px; }}
.arena-controls select {{
    padding: 6px 12px; font-size: 14px; margin-left: 4px;
    border: 1px solid #e5e7eb; border-radius: 6px; background: #f8f9fb;
    color: #2a3f5f; min-width: 220px;
}}
.arena-controls label {{ font-weight: 600; margin-right: 16px; }}
/* Coloured dropdown rows: green = new net won, red = previous net won. */
#alphaBlokus-game-select option.result-new {{ background: #dcfce7; color: #14532d; }}
#alphaBlokus-game-select option.result-prev {{ background: #fee2e2; color: #7f1d1d; }}
#alphaBlokus-game-select option.result-draw {{ background: transparent; color: inherit; }}

.replay-body {{ display: flex; flex-direction: column; gap: 12px; }}

.replay-turn {{
    border: 1px solid #e5e7eb; border-radius: 6px; background: #fafafa;
    overflow: hidden;
}}
.replay-turn-toggle {{
    display: flex; align-items: center; justify-content: space-between;
    width: 100%; padding: 10px 16px; border: 0;
    background: none; cursor: pointer; font-family: inherit;
    color: #2a3f5f; font-size: 13px; font-weight: 600;
    text-align: left;
}}
.replay-turn-toggle:hover {{ background: #f3f4f6; }}
.replay-turn-hint {{
    font-size: 11px; font-weight: 400; color: #6b7280; letter-spacing: 0.3px;
}}

.replay-turn-actual {{ padding: 0 16px 16px 16px; }}
.replay-turn-candidates {{
    padding: 12px 16px 16px 16px;
    border-top: 1px dashed #e5e7eb; background: #ffffff;
}}
.replay-turn-candidates[hidden] {{ display: none; }}

.replay-result {{
    padding: 16px 18px; text-align: center;
    font-size: 17px; font-weight: 700; letter-spacing: 0.4px;
    color: #1f2937; background: #f3f4f6;
    border: 1px solid #d1d5db; border-radius: 6px;
}}

/* Inlined TTT board styles — keep the policy / actual boards aligned. */
.ttt-board {{ display: inline-block; margin: 0; }}
.ttt-board table.ttt-grid {{ border-collapse: collapse; }}
.ttt-board td {{ padding: 0; }}
.ttt-board th {{
    padding: 2px 6px; font-size: 10px; color: #9ca3af;
    background: none; border: none; font-weight: normal; text-align: center;
}}
.ttt-board th.corner {{ width: 16px; }}
.ttt-board th.row-label {{ text-align: right; padding-right: 6px; }}
.ttt-board .board-annotation {{
    font-size: 11px; color: #4b5563; text-align: center;
    margin-bottom: 6px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px;
}}

{board_css}
</style>
</head>
<body>
<h1>Arena Game Replays — {run_name}</h1>
<p class="subtitle">
  {total_games} games across {num_gens} generations. Each replay shows the
  actual board after every move by default; click a turn header to expand
  the top-3 candidate previews the model was considering for that move.
  Player 1 is the previous best network and Player 2 is the new candidate
  trained this generation.
</p>

<div class="arena-controls">
    <label>Generation:
        <select id="alphaBlokus-gen-select"
                onchange="alphaBlokus_onGenChange(parseInt(this.value))">
{gen_options}
        </select>
    </label>
    <label>Game:
        <select id="alphaBlokus-game-select"
                onchange="alphaBlokus_onGameChange(parseInt(this.value))">
{initial_game_options}
        </select>
    </label>
    <label>
        <button type="button" onclick="alphaBlokus_toggleAll()">Expand / collapse all candidates</button>
    </label>
</div>
<div id="alphaBlokus-replay-body" class="replay-body"></div>

<script>
(function() {{
    const REPLAYS = {payload};

    function genGames(gen) {{ return REPLAYS[gen] || []; }}

    function onGenChange(gen) {{
        const select = document.getElementById('alphaBlokus-game-select');
        const games = genGames(gen);
        select.innerHTML = games.map(g => {{
            return '<option class="' + g.outcome_class + '" value="'
                 + g.game_idx + '">G' + (g.game_idx + 1)
                 + ' — ' + g.outcome_label + '</option>';
        }}).join('');
        if (games.length > 0) {{
            select.value = games[0].game_idx;
            renderGame(gen, games[0].game_idx);
        }} else {{
            document.getElementById('alphaBlokus-replay-body').innerHTML = '';
        }}
    }}

    function onGameChange(gameIdx) {{
        const gen = parseInt(document.getElementById('alphaBlokus-gen-select').value);
        renderGame(gen, gameIdx);
    }}

    function renderGame(gen, gameIdx) {{
        const games = genGames(gen);
        const game = games.find(g => g.game_idx === gameIdx);
        if (!game) return;
        document.getElementById('alphaBlokus-replay-body').innerHTML =
            game.turns_html.join('');
    }}

    function toggleTurn(buttonEl) {{
        const turn = buttonEl.parentElement;
        const candidates = turn.querySelector('.replay-turn-candidates');
        if (!candidates) return;
        candidates.hidden = !candidates.hidden;
        turn.classList.toggle('expanded', !candidates.hidden);
        const hint = buttonEl.querySelector('.replay-turn-hint');
        if (hint) hint.textContent = candidates.hidden
            ? '↓ click to show top candidates'
            : '↑ hide candidates';
    }}

    function toggleAll() {{
        const turns = document.querySelectorAll('.replay-turn');
        const allHidden = Array.from(turns).every(t => {{
            const c = t.querySelector('.replay-turn-candidates');
            return !c || c.hidden;
        }});
        turns.forEach(t => {{
            const c = t.querySelector('.replay-turn-candidates');
            if (!c) return;
            c.hidden = !allHidden;
            t.classList.toggle('expanded', allHidden);
            const hint = t.querySelector('.replay-turn-hint');
            if (hint) hint.textContent = allHidden
                ? '↑ hide candidates'
                : '↓ click to show top candidates';
        }});
    }}

    document.addEventListener('DOMContentLoaded', () => {{
        const genSelect = document.getElementById('alphaBlokus-gen-select');
        if (!genSelect) return;
        const gen = parseInt(genSelect.value);
        const games = genGames(gen);
        if (games.length > 0) {{
            renderGame(gen, games[0].game_idx);
        }}
    }});

    window.alphaBlokus_onGenChange = onGenChange;
    window.alphaBlokus_onGameChange = onGameChange;
    window.alphaBlokus_toggleTurn = toggleTurn;
    window.alphaBlokus_toggleAll = toggleAll;
}})();
</script>
</body>
</html>
"""


