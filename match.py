"""Play the engine against Stockfish (or any UCI engine) and report the score.

    python match.py --elo 1500 --sf-elo 1500 --games 20
    python match.py --elo 1000 1500 2000 --sf-elo 1500 --games 20     # one match per bot rating
    python match.py --elo 1200 --sf-skill 3 --games 10 --show         # weaker than 1320: use Skill Level
    python match.py --elo 1800 --opponent C:/path/to/lc0.exe --opt WeightsFile=maia-1500.pb.gz --nodes 1

Colours alternate every game. Games are saved to matches/<name>.pgn. The bot's
clock is simulated from its predicted think times, so it can still feel time
pressure in long games; the opponent's clock is assumed to match it.

Stockfish's UCI_Elo is calibrated at longer time controls than --sf-time, so
treat the performance ratings below as rough.
"""

import argparse
import datetime
import math
import shutil
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from hc.engine import HumanEngine

MAX_PLIES = 400


def find_stockfish():
    path = shutil.which("stockfish")
    if path:
        return path
    link = Path.home() / "AppData/Local/Microsoft/WinGet/Links/stockfish.exe"   # winget installs here
    if link.exists():
        return str(link)
    raise SystemExit("Stockfish not found: install it or pass --opponent /path/to/engine")


def parse_opts(items):
    opts = {}
    for item in items:
        name, _, value = item.partition("=")
        opts[name] = int(value) if value.lstrip("-").isdigit() else value
    return opts


def play_game(bot, opp, limit, elo, opp_elo, bot_white, base, inc, show=False):
    board = chess.Board()
    bot_color = chess.WHITE if bot_white else chess.BLACK
    clock = base
    resigned = None
    thinks, loss_moves = [], 0

    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        if board.turn == bot_color:
            d = bot.decide(board, elo, opp_elo, clock, clock, int(base), inc)
            if bot.should_resign(d, board):
                resigned = bot_color
                break
            if show:
                top = ", ".join(f"{board.san(m)} {p:.0%}" for m, p in d.probs[:4])
                print(f"  bot   {board.san(d.move):7} ({top})  {d.think_seconds:.1f}s")
            thinks.append(d.think_seconds)
            clock = max(1.0, clock - d.think_seconds + inc)
            board.push(d.move)
        else:
            result = opp.play(board, limit)
            if result.resigned:
                resigned = not bot_color
                break
            if show:
                print(f"  opp   {board.san(result.move)}")
            board.push(result.move)

    if resigned is not None:
        result = "0-1" if resigned == chess.WHITE else "1-0"
        reason = "bot resigned" if resigned == bot_color else "opponent resigned"
    elif board.ply() >= MAX_PLIES and not board.is_game_over(claim_draw=True):
        result, reason = "1/2-1/2", f"adjudicated draw after {MAX_PLIES} plies"
    else:
        result = board.result(claim_draw=True)
        reason = board.outcome(claim_draw=True).termination.name.lower().replace("_", " ")
    return board, result, reason


def score_for_bot(result, bot_white):
    if result == "1/2-1/2":
        return 0.5
    return 1.0 if (result == "1-0") == bot_white else 0.0


def performance(score, n, opp_elo):
    if opp_elo is None or n == 0:
        return None
    s = min(max(score / n, 0.5 / n), 1 - 0.5 / n)    # keep 0/n and n/n finite
    return round(opp_elo + 400 * math.log10(s / (1 - s)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/run/model.pt")
    p.add_argument("--elo", type=int, nargs="+", default=[1500], help="rating(s) the bot imitates")
    p.add_argument("--mode", choices=["sample", "top"], default="sample")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--opponent", help="UCI engine path (default: Stockfish)")
    strength = p.add_mutually_exclusive_group()
    strength.add_argument("--sf-elo", type=int, help="Stockfish UCI_Elo, 1320-3190")
    strength.add_argument("--sf-skill", type=int, help="Stockfish Skill Level, 0-20 (for below 1320)")
    p.add_argument("--opt", action="append", default=[], help="extra UCI option for the opponent, name=value")
    p.add_argument("--sf-time", type=float, default=0.1, help="opponent seconds per move")
    p.add_argument("--nodes", type=int, help="opponent nodes per move instead of time (e.g. 1 for Maia)")
    p.add_argument("--minutes", type=float, default=5, help="time control the bot believes it is playing")
    p.add_argument("--inc", type=int, default=3)
    p.add_argument("--show", action="store_true", help="print every move with the bot's thoughts")
    p.add_argument("--out", type=Path, default=Path("matches"))
    args = p.parse_args()

    path = args.opponent or find_stockfish()
    opp = chess.engine.SimpleEngine.popen_uci(path)
    opts = parse_opts(args.opt)
    opp_elo, opp_label = None, Path(path).stem
    if args.sf_elo:
        opts.update({"UCI_LimitStrength": True, "UCI_Elo": args.sf_elo})
        opp_elo, opp_label = args.sf_elo, f"Stockfish {args.sf_elo}"
    elif args.sf_skill is not None:
        opts["Skill Level"] = args.sf_skill
        opp_label = f"Stockfish skill {args.sf_skill}"
    opp.configure(opts)
    limit = chess.engine.Limit(nodes=args.nodes) if args.nodes else chess.engine.Limit(time=args.sf_time)

    bot = HumanEngine(args.checkpoint, mode=args.mode)
    base = args.minutes * 60
    args.out.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    pgn_path = args.out / f"{stamp}-{opp_label.replace(' ', '_')}.pgn"
    summary = []

    try:
        for elo in args.elo:
            print(f"\nHumanChess {elo} ({args.mode}) vs {opp_label}, {args.games} games")
            score, wdl = 0.0, [0, 0, 0]
            for g in range(args.games):
                bot_white = g % 2 == 0
                t0 = time.time()
                board, result, reason = play_game(bot, opp, limit, elo, opp_elo or elo, bot_white,
                                                  base, args.inc, args.show)
                s = score_for_bot(result, bot_white)
                score += s
                wdl[0 if s == 1 else 1 if s == 0.5 else 2] += 1

                game = chess.pgn.Game.from_board(board)
                bot_name, opp_name = f"HumanChess {elo}", opp_label
                game.headers.update({
                    "Event": f"HumanChess {elo} vs {opp_label}", "Round": str(g + 1),
                    "Date": datetime.date.today().strftime("%Y.%m.%d"),
                    "White": bot_name if bot_white else opp_name,
                    "Black": opp_name if bot_white else bot_name,
                    "Result": result, "Termination": reason,
                })
                with open(pgn_path, "a", encoding="utf-8") as f:
                    print(game, file=f, end="\n\n")

                side = "W" if bot_white else "B"
                print(f"  game {g + 1:>3} bot as {side}  {result:>7}  {board.ply() // 2 + 1:>3} moves  "
                      f"{reason:<22} {time.time() - t0:5.1f}s   running score {score:g}/{g + 1}")
            perf = performance(score, args.games, opp_elo)
            summary.append((elo, wdl, score, perf))
    finally:
        opp.quit()

    print(f"\nvs {opp_label}  ({args.games} games each, games saved to {pgn_path})")
    print(f"{'bot elo':>8}  {'W-D-L':>9}  {'score':>7}  {'perf':>6}")
    for elo, (w, d, l), score, perf in summary:
        pct = f"{score / args.games:.0%}"
        print(f"{elo:>8}  {f'{w}-{d}-{l}':>9}  {pct:>7}  {perf if perf is not None else '-':>6}")


if __name__ == "__main__":
    main()
