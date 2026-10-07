"""Play against the engine in your browser, watch it play Stockfish, or replay
the games match.py saved.

    python gui.py
    python gui.py --checkpoint checkpoints/run/model.pt --port 8642

Then open http://localhost:8642. The page shows what the bot considered on every
move, its predicted think time and how it rates its chances.
"""

import argparse
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import chess
import chess.engine
import chess.pgn

from hc.engine import HumanEngine
from match import find_stockfish

PAGE = Path(__file__).with_name("gui.html")
MATCHES = Path(__file__).with_name("matches")


class Game:
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.Lock()
        self.sf = None
        self.new({})

    def new(self, cfg):
        self.board = chess.Board()
        self.elo = int(cfg.get("elo", 1500))
        self.your_elo = int(cfg.get("your_elo") or self.elo)
        self.you = chess.WHITE if cfg.get("color", "white") == "white" else chess.BLACK
        self.base = float(cfg.get("minutes", 5)) * 60
        self.inc = int(cfg.get("inc", 3))
        self.engine.mode = cfg.get("mode", "sample")
        self.engine.temperature = float(cfg.get("temperature", 1.0))
        self.resigned = None
        self.thoughts = None
        return self.state()

    def move(self, uci):
        move = chess.Move.from_uci(uci)
        if self.board.turn != self.you or move not in self.board.legal_moves:
            raise ValueError("illegal move")
        self.board.push(move)
        self.thoughts = None
        return self.state()

    def bot(self, clock_bot, clock_you):
        if self.board.turn == self.you or self.over():
            raise ValueError("not the bot's turn")
        board = self.board
        d = self.engine.decide(board, self.elo, self.your_elo, clock_bot, clock_you, int(self.base), self.inc)
        thoughts = describe(board, d)
        if self.engine.should_resign(d, board):
            self.resigned = not self.you
            thoughts["played"] = None
        else:
            thoughts["played"] = board.san(d.move)
            board.push(d.move)
        self.thoughts = thoughts
        return self.state()

    def stockfish(self, sf_elo, seconds):
        """Let Stockfish make your move, so you can watch the bot play it."""
        if self.board.turn != self.you or self.over():
            raise ValueError("not your turn")
        if self.sf is None:
            self.sf = chess.engine.SimpleEngine.popen_uci(find_stockfish())
        self.sf.configure({"UCI_LimitStrength": True, "UCI_Elo": max(1320, min(3190, sf_elo))})
        result = self.sf.play(self.board, chess.engine.Limit(time=seconds))
        self.board.push(result.move)
        self.thoughts = None
        return self.state()

    def analyse(self, fen, uci, elo, opp_elo):
        """What the bot weighed before playing `uci` in a saved game."""
        board = chess.Board(fen)
        thoughts = describe(board, self.engine.decide(board, elo, opp_elo))
        move = chess.Move.from_uci(uci)
        thoughts["played"] = board.san(move)
        thoughts["uci"] = uci
        return thoughts

    def undo(self):
        # take back to the last position where it was your move
        if self.board.move_stack:
            self.board.pop()
        while self.board.move_stack and self.board.turn != self.you:
            self.board.pop()
        self.resigned = None
        self.thoughts = None
        return self.state()

    def resign(self):
        self.resigned = self.you
        return self.state()

    def over(self):
        return self.resigned is not None or self.board.is_game_over()

    def state(self):
        b = self.board
        replay, sans = chess.Board(), []
        for m in b.move_stack:
            sans.append(replay.san(m))
            replay.push(m)
        result, reason = None, None
        if self.resigned is not None:
            result = "0-1" if self.resigned == chess.WHITE else "1-0"
            reason = "you resigned" if self.resigned == self.you else "the bot resigned"
        elif b.is_game_over():
            outcome = b.outcome()
            result, reason = b.result(), outcome.termination.name.lower().replace("_", " ")
        return {
            "fen": b.fen(),
            "turn": "white" if b.turn else "black",
            "you": "white" if self.you else "black",
            "legal": [m.uci() for m in b.legal_moves] if b.turn == self.you and result is None else [],
            "moves": sans,
            "last": b.peek().uci() if b.move_stack else None,
            "check": chess.square_name(b.king(b.turn)) if b.is_check() else None,
            "result": result,
            "reason": reason,
            "elo": self.elo,
            "base": self.base,
            "inc": self.inc,
            "thoughts": self.thoughts,   # what the bot considered on its last move
        }


def describe(board, d):
    return {
        "considered": [{"san": board.san(m), "uci": m.uci(), "p": p} for m, p in d.probs[:8]],
        "think": d.think_seconds,
        "outcome": d.outcome,
    }


def list_matches():
    files = []
    for path in sorted(MATCHES.glob("*.pgn"), reverse=True):
        games = []
        with open(path, encoding="utf-8") as f:
            while (g := chess.pgn.read_game(f)) is not None:
                h = g.headers
                games.append({"white": h.get("White"), "black": h.get("Black"), "result": h.get("Result"),
                              "termination": h.get("Termination", ""), "event": h.get("Event", ""),
                              "plies": sum(1 for _ in g.mainline_moves())})
        files.append({"name": path.name, "games": games})
    return {"files": files}


def rating(name):
    m = re.search(r"(\d{3,4})$", name or "")
    return int(m.group(1)) if m else None


def load_match(name, index):
    path = MATCHES / Path(name).name    # never read outside matches/
    if path.suffix != ".pgn" or not path.exists():
        raise ValueError("no such match file")
    with open(path, encoding="utf-8") as f:
        for _ in range(index):
            chess.pgn.skip_game(f)
        game = chess.pgn.read_game(f)
    if game is None:
        raise ValueError("no such game")
    h = game.headers
    bot = "white" if h.get("White", "").startswith("HumanChess") else "black"
    opp = h.get("Black") if bot == "white" else h.get("White")
    board = game.board()
    plies = []
    for move in game.mainline_moves():
        before = board.fen()
        san = board.san(move)
        board.push(move)
        plies.append({"uci": move.uci(), "san": san, "before": before, "fen": board.fen(),
                      "check": chess.square_name(board.king(board.turn)) if board.is_check() else None})
    return {"white": h.get("White"), "black": h.get("Black"), "result": h.get("Result"),
            "termination": h.get("Termination", ""), "bot": bot,
            "elo": rating(h.get("White") if bot == "white" else h.get("Black")) or 1500,
            "opp": opp, "opp_elo": rating(opp), "start": game.board().fen(), "plies": plies}


def make_handler(game):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    with game.lock:
                        self.send(200, game.state())
                elif url.path == "/api/matches":
                    self.send(200, list_matches())
                elif url.path == "/api/match":
                    self.send(200, load_match(q["file"], int(q.get("game", 0))))
                else:
                    self.send(404, {"error": "not found"})
            except (ValueError, KeyError) as e:
                self.send(400, {"error": str(e)})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            routes = {
                "/api/new": lambda: game.new(body),
                "/api/move": lambda: game.move(body["uci"]),
                "/api/bot": lambda: game.bot(float(body["clock_bot"]), float(body["clock_you"])),
                "/api/stockfish": lambda: game.stockfish(int(body.get("elo", 1500)), float(body.get("time", 0.3))),
                "/api/analyse": lambda: game.analyse(body["fen"], body["uci"], int(body["elo"]),
                                                     int(body.get("opp_elo") or body["elo"])),
                "/api/undo": game.undo,
                "/api/resign": game.resign,
            }
            if self.path not in routes:
                return self.send(404, {"error": "not found"})
            try:
                with game.lock:
                    self.send(200, routes[self.path]())
            except (ValueError, KeyError) as e:
                self.send(400, {"error": str(e)})

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/run/model.pt")
    p.add_argument("--port", type=int, default=8642)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    print("loading model...")
    game = Game(HumanEngine(args.checkpoint))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(game))
    url = f"http://localhost:{args.port}"
    print(f"playing at {url}  (ctrl+c to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if game.sf:
            game.sf.quit()


if __name__ == "__main__":
    main()
