"""Play against the engine in the terminal.

    python play.py --elo 1200
    python play.py --elo 1800 --color black --thoughts

Enter moves as SAN (Nf3, exd5, O-O) or UCI (g1f3). Type "resign" to resign.
"""

import argparse
import time

import chess

from hc.engine import HumanEngine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/run/model.pt")
    p.add_argument("--elo", type=int, default=1500)
    p.add_argument("--your-elo", type=int, help="your rating, if you want the bot to know it")
    p.add_argument("--color", choices=["white", "black"], default="white")
    p.add_argument("--minutes", type=float, default=5)
    p.add_argument("--inc", type=int, default=3)
    p.add_argument("--mode", choices=["sample", "top"], default="sample")
    p.add_argument("--thoughts", action="store_true", help="show what the bot was considering")
    args = p.parse_args()

    engine = HumanEngine(args.checkpoint, mode=args.mode)
    board = chess.Board()
    you = chess.WHITE if args.color == "white" else chess.BLACK
    base = args.minutes * 60
    clock = {chess.WHITE: base, chess.BLACK: base}
    your_elo = args.your_elo or args.elo

    while not board.is_game_over():
        print("\n" + board.unicode(invert_color=True, borders=True, orientation=you))
        print(f"clocks  you {clock[you]:.0f}s   bot {clock[not you]:.0f}s")
        if board.turn == you:
            t0 = time.time()
            while True:
                s = input("your move: ").strip()
                if s == "resign":
                    print("you resigned")
                    return
                try:
                    move = board.parse_san(s)
                except ValueError:
                    try:
                        move = chess.Move.from_uci(s)
                        if move not in board.legal_moves:
                            raise ValueError
                    except ValueError:
                        print("not a legal move")
                        continue
                break
            clock[you] = clock[you] - (time.time() - t0) + args.inc
            board.push(move)
        else:
            d = engine.decide(board, args.elo, your_elo, clock[board.turn], clock[not board.turn],
                              int(base), args.inc)
            if engine.should_resign(d, board):
                print(f"\nthe bot resigns (it gives itself a {d.outcome['loss']:.0%} chance of losing)")
                return
            if args.thoughts:
                top = ", ".join(f"{board.san(m)} {p:.0%}" for m, p in d.probs[:4])
                print(f"a {args.elo} player here would consider: {top}")
            san = board.san(d.move)
            clock[board.turn] = clock[board.turn] - d.think_seconds + args.inc
            board.push(d.move)
            print(f"bot plays {san}  (thought for {d.think_seconds:.1f}s)")
    print("\n" + board.unicode(invert_color=True, borders=True, orientation=you))
    print("game over:", board.result())


if __name__ == "__main__":
    main()
