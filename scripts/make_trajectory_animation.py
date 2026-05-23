#!/usr/bin/env python3
"""由轨迹 JSON/CSV 生成 GIF/MP4 动画。"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from visualization.trajectory_animation import make_animation


def main() -> None:
    ap = argparse.ArgumentParser(description="轨迹动画")
    ap.add_argument("--trajectory_path", type=str, required=True)
    ap.add_argument("--save_path", type=str, required=True)
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    out = make_animation(args.trajectory_path, args.save_path, title=args.title, fps=args.fps)
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
