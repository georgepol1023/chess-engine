"""Stream games from the Lichess open database and turn them into training data.

    python download.py --month 2025-01 --games 300000

The monthly files are tens of gigabytes, so nothing is saved to disk except the
processed positions: the file is streamed, decompressed on the fly, filtered, and
the download stops once enough games have been kept.

A local file works too (plain .pgn or .pgn.zst):
    python download.py --pgn my_games.pgn

Output: data/train.npy and data/val.npy. Games are split by game, never by
position, so no game has positions on both sides of the split.
"""

import argparse
import io
import os
import time
import urllib.request
import zlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from hc.data import Filter, game_records, merge_chunks, quick_accept, save_chunk, split_games

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def open_lines(source: str):
    import zstandard

    if source.startswith("http"):
        req = urllib.request.Request(source, headers={"User-Agent": "human-chess-research"})
        raw = urllib.request.urlopen(req)
    else:
        raw = open(source, "rb")
    if source.endswith(".zst"):
        raw = zstandard.ZstdDecompressor().stream_reader(raw)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def process_batch(texts, flt):
    out = []
    for text in texts:
        rec = game_records(text, flt)
        if len(rec):
            out.append((text, rec))
    return out


def batched(it, n):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--month", default="2025-01", help="Lichess database month, YYYY-MM")
    src.add_argument("--pgn", help="local .pgn or .pgn.zst file instead of downloading")
    p.add_argument("--games", type=int, default=300_000, help="stop after keeping this many games")
    p.add_argument("--val-percent", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--chunk-positions", type=int, default=2_000_000)
    p.add_argument("--min-base", type=int, default=Filter.min_base, help="minimum base time, seconds")
    args = p.parse_args()

    flt = Filter(min_base=args.min_base)
    source = args.pgn or URL.format(month=args.month)
    print(f"reading {source}")

    kept, seen, t0 = 0, 0, time.time()
    buffers = {"train": [], "val": []}
    counts = {"train": 0, "val": 0}
    chunks = {"train": [], "val": []}

    def flush(split):
        if not buffers[split]:
            return
        path = args.out / "chunks" / f"{split}_{len(chunks[split]):04d}.npy"
        save_chunk(buffers[split], path)
        chunks[split].append(path)
        buffers[split], counts[split] = [], 0

    def candidates():
        nonlocal seen
        for text in split_games(open_lines(source)):
            seen += 1
            if quick_accept(text, flt):
                yield text

    with Pool(args.workers) as pool:
        work = pool.imap(partial(process_batch, flt=flt), batched(candidates(), 64))
        for results in work:
            for text, rec in results:
                key = text[text.find('[Site "'):text.find('[Site "') + 40]
                split = "val" if zlib.crc32(key.encode()) % 10_000 < args.val_percent * 100 else "train"
                buffers[split].append(rec)
                counts[split] += len(rec)
                if counts[split] >= args.chunk_positions:
                    flush(split)
                kept += 1
                if kept % 5000 == 0:
                    rate = kept / (time.time() - t0)
                    print(f"  {kept:,} games kept of {seen:,} read ({rate:.0f} games/s)")
                if kept >= args.games:
                    break
            if kept >= args.games:
                pool.terminate()
                break

    for split in ("train", "val"):
        flush(split)
        n = merge_chunks(chunks[split], args.out / f"{split}.npy") if chunks[split] else 0
        print(f"{split}: {n:,} positions -> {args.out / f'{split}.npy'}")
    (args.out / "chunks").rmdir() if (args.out / "chunks").exists() else None
    print(f"kept {kept:,} of {seen:,} games in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
