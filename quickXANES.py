#!/usr/bin/env python


#               _      _    __   __          _   _ ______  _____ 
#              (_)    | |   \ \ / /    /\   | \ | |  ____|/ ____|
#    __ _ _   _ _  ___| | __ \ V /    /  \  |  \| | |__  | (___  
#   / _` | | | | |/ __| |/ /  > <    / /\ \ | . ` |  __|  \___ \ 
#  | (_| | |_| | | (__|   <  / . \  / ____ \| |\  | |____ ____) |
#   \__, |\__,_|_|\___|_|\_\/_/ \_\/_/    \_\_| \_|______|_____/ 
#      | |                                                       
#      |_|                                                       


"""quickXANES: minimal XANES toolkit (Bin -> QC -> Normalize -> LCF).

    python quickXANES.py                                   # open the app, load a folder in the UI
    python quickXANES.py --data ../SSRL/bl143_nov26/xas    # open with a data folder loaded
    python quickXANES.py --params <processed>/params/quickxanes_params.json        # resume a session
    python quickXANES.py --params <...>.json --run         # headless: rewrite every output from the JSON
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qxanes.session import Session  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder of .xdi/.dat scans")
    ap.add_argument("--out", help="output folder (default: <data folder>/../processed)")
    ap.add_argument("--params", help="parameters JSON saved by quickXANES")
    ap.add_argument("--run", action="store_true", help="reproduce all outputs from --params without the app")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    S = Session.from_json(args.params) if args.params else Session()
    if args.data:
        S.load_dir(args.data, args.out)
    elif S.params["data_dir"] and os.path.isdir(S.params["data_dir"]):
        S.load_dir(S.params["data_dir"], args.out)

    if args.run:
        if not args.params:
            ap.error("--run needs --params")
        for step, paths in S.run_all().items():
            print(f"{step:11s} {len(paths):3d} files  {os.path.dirname(paths[0]) if paths else ''}")
        for w in S.lcf_warnings:
            print(f"warning: {w}")
        if S.standard_groups() and S.lcf_undecided():
            print(f"warning: no locked-in LCF for {', '.join(S.lcf_undecided())}; left out of lcf_results.csv")
        return

    from qxanes.app import create_app
    app = create_app(S)
    url = f"http://127.0.0.1:{args.port}"
    print(f"quickXANES running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
