"""UCI wrapper, so the engine works in any chess GUI and with lichess-bot.

    python uci.py --checkpoint checkpoints/run/model.pt

UCI options: UCI_Elo (the level to imitate), OpponentElo (0 = same as UCI_Elo),
Mode (sample / top), Temperature (x100), HumanTiming (actually wait the predicted
think time before moving).
"""

import argparse
import sys
import time

import chess

from hc.engine import HumanEngine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/run/model.pt")
    args = p.parse_args()

    engine = None
    opts = {"UCI_Elo": 1500, "OpponentElo": 0, "Mode": "sample", "Temperature": 100, "HumanTiming": False}
    board = chess.Board()
    start_clock = {}

    def send(s):
        print(s, flush=True)

    for line in sys.stdin:
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd == "uci":
            send("id name HumanChess")
            send("id author georgepol1023")
            send("option name UCI_LimitStrength type check default true")
            send("option name UCI_Elo type spin default 1500 min 600 max 2900")
            send("option name OpponentElo type spin default 0 min 0 max 2900")
            send("option name Mode type combo default sample var sample var top")
            send("option name Temperature type spin default 100 min 10 max 300")
            send("option name HumanTiming type check default false")
            send("uciok")
        elif cmd == "isready":
            if engine is None:
                engine = HumanEngine(args.checkpoint)
            send("readyok")
        elif cmd == "setoption" and "name" in parts and "value" in parts:
            name = " ".join(parts[parts.index("name") + 1:parts.index("value")])
            value = " ".join(parts[parts.index("value") + 1:])
            if name in ("UCI_Elo", "OpponentElo", "Temperature"):
                opts[name] = int(value)
            elif name == "Mode":
                opts[name] = value
            elif name == "HumanTiming":
                opts[name] = value.lower() == "true"
        elif cmd == "ucinewgame":
            board, start_clock = chess.Board(), {}
        elif cmd == "position":
            if parts[1] == "startpos":
                board, rest = chess.Board(), parts[2:]
            else:
                fen_end = parts.index("moves") if "moves" in parts else len(parts)
                board, rest = chess.Board(" ".join(parts[2:fen_end])), parts[fen_end:]
            if rest and rest[0] == "moves":
                for mv in rest[1:]:
                    board.push_uci(mv)
        elif cmd == "go":
            if engine is None:
                engine = HumanEngine(args.checkpoint)
            kv = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1)}
            me, them = ("w", "b") if board.turn == chess.WHITE else ("b", "w")
            clock = int(kv.get(f"{me}time", 300_000)) / 1000
            opp_clock = int(kv.get(f"{them}time", clock * 1000)) / 1000
            inc = int(kv.get(f"{me}inc", 0)) / 1000
            start_clock.setdefault("base", max(clock, opp_clock))   # best guess at the time control
            engine.mode, engine.temperature = opts["Mode"], opts["Temperature"] / 100
            elo = opts["UCI_Elo"]
            d = engine.decide(board, elo, opts["OpponentElo"] or elo, clock, opp_clock,
                              start_clock["base"], inc)
            if opts["HumanTiming"]:
                time.sleep(d.think_seconds)
            top = d.probs[0]
            send(f"info string human-likely {board.san(top[0])} {top[1]:.0%}, think {d.think_seconds:.1f}s")
            send(f"bestmove {d.move.uci()}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
