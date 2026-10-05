"""Swap your old trading bot for this one, using your own GitHub login.

You need git and the GitHub CLI, logged in once (Windows):
    winget install --id Git.Git -e
    winget install --id GitHub.cli -e
    gh auth login --scopes workflow
Then, from this folder:
    python switch_over.py            # steps 1 to 4, then it stops so you can check
    python switch_over.py --go-live  # step 5, on a weekday between 08:00 and 16:30 UK time

1. Disables the old bot's GitHub workflows so the two bots can't fight. Its repository is kept
   exactly as it is, so you can turn it back on from its Actions tab.
2. Creates a new private repository (isa-bot) and pushes this bot to it.
3. Asks for your Trading 212 Stocks ISA API key and secret. They go straight to GitHub as
   encrypted secrets: never saved to disk or shown. Sets T212_ENV=live, TRADING_ENABLED=false.
4. Runs the ticker check and a dry run that includes selling the old bot's holdings, and prints
   what the bot would do. Nothing is traded.
5. --go-live sets TRADING_ENABLED=true and runs the switch-over for real. After that the bot
   runs by itself every weekday.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

WORKFLOW = "trade.yml"


def sh(*args, input=None, check=True, capture=True):
    r = subprocess.run(list(args), input=input, text=True, capture_output=capture,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        sys.exit(f"\nThis failed: {' '.join(args[:4])}\n{(r.stderr or r.stdout or '').strip()}")
    return r


def gh_json(*args):
    out = sh("gh", *args).stdout.strip()
    return json.loads(out) if out else None


def uk_now(now: datetime | None = None) -> datetime:
    """UK clock time without needing a time-zone database (BST: last Sunday of March to last
    Sunday of October, changing at 01:00 UTC)."""
    now = now or datetime.now(timezone.utc)

    def last_sunday(month):
        d = datetime(now.year, month, 31, 1, tzinfo=timezone.utc)
        return d - timedelta(days=(d.weekday() + 1) % 7)
    return now + timedelta(hours=1 if last_sunday(3) <= now < last_sunday(10) else 0)


def market_open_uk(now: datetime | None = None) -> bool:
    t = uk_now(now)
    return t.weekday() < 5 and (8, 0) <= (t.hour, t.minute) < (16, 30)


def run_ids(repo: str) -> set:
    runs = gh_json("run", "list", "--repo", repo, "--workflow", WORKFLOW, "--limit", "20", "--json", "databaseId")
    return {r["databaseId"] for r in runs or []}


def dispatch(repo: str, **inputs) -> int:
    """Start the Trade workflow and return the new run's id."""
    before = run_ids(repo)
    args = ["gh", "workflow", "run", WORKFLOW, "--repo", repo]
    for k, v in inputs.items():
        args += ["-f", f"{k}={'true' if v else 'false'}"]
    for _ in range(12):                      # a just-pushed workflow can take a moment to register
        r = sh(*args, check=False)
        if r.returncode == 0:
            break
        time.sleep(5)
    else:
        sys.exit(f"Couldn't start the workflow: {r.stderr.strip()}")
    for _ in range(60):
        new = run_ids(repo) - before
        if new:
            return max(new)
        time.sleep(3)
    sys.exit("The run didn't appear. Check the repository's Actions tab.")


def wait_and_show(repo: str, run_id: int, step: str) -> tuple[bool, str]:
    print(f"\nWaiting for run {run_id} (usually 3 to 5 minutes)...")
    sh("gh", "run", "watch", str(run_id), "--repo", repo, "--interval", "15", check=False, capture=False)
    ok = (gh_json("run", "view", str(run_id), "--repo", repo, "--json", "conclusion") or {}).get("conclusion") == "success"
    log = sh("gh", "run", "view", str(run_id), "--repo", repo, "--log", check=False).stdout
    lines, grouped = [], False
    for line in log.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3 or parts[1] != step:
            continue
        text = parts[2].split(" ", 1)[1] if " " in parts[2] else ""
        if text.startswith("##[group]"):            # GitHub's echo of the command and its settings
            grouped = True
        elif text.startswith("##[endgroup]"):
            grouped = False
        elif not grouped and not text.startswith("##["):
            lines.append(text)
    out = "\n".join(lines).strip()
    print("-" * 70 + "\n" + (out or "(no output found; open the run in the Actions tab)") + "\n" + "-" * 70)
    return ok, out


def pick_old_repo(given: str | None) -> str | None:
    if given:
        return None if given.lower() == "none" else given
    repos = gh_json("repo", "list", "--limit", "100", "--json", "nameWithOwner,pushedAt") or []
    repos.sort(key=lambda r: r.get("pushedAt") or "", reverse=True)
    print("\nWhich repository runs your OLD bot?")
    for i, r in enumerate(repos[:20], 1):
        print(f"  {i:2d}. {r['nameWithOwner']}")
    print("   0. none of these (the old bot isn't on GitHub)")
    while True:
        a = input("Number: ").strip()
        if a.isdigit() and 0 <= int(a) <= min(20, len(repos)):
            return None if a == "0" else repos[int(a) - 1]["nameWithOwner"]


def stop_old_bot(old: str | None) -> None:
    if not old:
        print("\nNo old repository chosen. If the old bot runs somewhere else, stop it there first.")
        return
    wfs = gh_json("workflow", "list", "--repo", old, "--all", "--json", "id,name,state") or []
    active = [w for w in wfs if w.get("state") == "active"]
    for w in active:
        sh("gh", "workflow", "disable", str(w["id"]), "--repo", old)
        print(f"Disabled '{w['name']}' in {old}")
    if not active:
        print(f"{old} has no active workflows.")


def set_up(args, login: str) -> str:
    repo = f"{login}/{args.new}"
    old = pick_old_repo(args.old)
    print("\nStep 1: stopping the old bot")
    stop_old_bot(old)

    print(f"\nStep 2: creating {repo} (private) with this bot")
    if sh("gh", "repo", "view", repo, "--json", "name", check=False).returncode == 0:
        print(f"{repo} already exists, so it is used as it is (code not pushed again).")
    else:
        if not os.path.isdir(".git"):
            sh("git", "init", "-b", "main")
        if not sh("git", "config", "user.email", check=False).stdout.strip():
            sh("git", "config", "user.email", f"{login}@users.noreply.github.com")
            sh("git", "config", "user.name", login)
        sh("git", "add", "-A")
        sh("git", "commit", "-m", "ISA bot: passive 60/40 core with guardrails", check=False)
        sh("gh", "repo", "create", repo, "--private", "--source", ".", "--remote", "origin", "--push")
        print(f"Pushed. https://github.com/{repo}")

    print("\nStep 3: your Trading 212 key. Make it in the app with the STOCKS ISA selected, with these")
    print("permissions: account, portfolio, orders:read, orders:execute, metadata, history:transactions.")
    print("What you type isn't shown. It goes straight to GitHub as an encrypted secret.")
    key = getpass.getpass("API key: ").strip()
    secret = getpass.getpass("API secret: ").strip()
    if not key or not secret:
        sys.exit("Both the key and the secret are needed.")
    sh("gh", "secret", "set", "T212_API_KEY", "--repo", repo, input=key)
    sh("gh", "secret", "set", "T212_API_SECRET", "--repo", repo, input=secret)
    sh("gh", "variable", "set", "T212_ENV", "--repo", repo, "--body", "live")
    sh("gh", "variable", "set", "TRADING_ENABLED", "--repo", repo, "--body", "false")
    print("Secrets and settings saved (real account, trading OFF).")

    print("\nStep 4a: checking the ticker mapping")
    ok, out = wait_and_show(repo, dispatch(repo, check_tickers=True), "Check ticker mapping")
    if not ok or "PROBLEM" in out or "refused the API key" in out:
        sys.exit("\nStopped: fix the problem shown above, then run this script again.")

    print("\nStep 4b: dry run, including selling the old bot's holdings (nothing is sent)")
    ok, out = wait_and_show(repo, dispatch(repo, dry_run=True, sell_other_holdings=True), "Run the bot")
    if not ok:
        sys.exit("\nStopped: the dry run failed. The output above says why.")
    print("\nAll set, and nothing has been traded. Check the plan above. When you're happy, run this")
    print("on a weekday between 08:00 and 16:30 UK time:\n")
    print("    python switch_over.py --go-live\n")
    return repo


def go_live(args, login: str) -> None:
    repo = args.repo or f"{login}/{args.new}"
    if not market_open_uk():
        sys.exit("The London market is closed now. Run this on a weekday between 08:00 and 16:30 UK time.")
    print(f"This turns on real trading in {repo}. The bot will sell the old bot's holdings now and")
    print("start building the 60/40. Afterwards it runs by itself every weekday.")
    if input("Type YES to go ahead: ").strip() != "YES":
        sys.exit("Nothing changed.")
    sh("gh", "variable", "set", "TRADING_ENABLED", "--repo", repo, "--body", "true")
    ok, _ = wait_and_show(repo, dispatch(repo, dry_run=False, sell_other_holdings=True), "Run the bot")
    if ok:
        print("\nDone. The bot now runs by itself every weekday at 13:17 UTC (14:17 UK time in summer).")
    else:
        print("\nThe run reported a problem (see above). Trading is still switched on, so check it.")
    print("Kill switch, any time:")
    print(f"    gh variable set TRADING_ENABLED --body false --repo {repo}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go-live", action="store_true", help="turn trading on and run the switch-over")
    ap.add_argument("--old", help="old bot's repository as OWNER/NAME, or 'none' (asked if not given)")
    ap.add_argument("--new", default="isa-bot", help="name for the new private repository (default isa-bot)")
    ap.add_argument("--repo", help="with --go-live: the new bot's repository as OWNER/NAME")
    args = ap.parse_args()
    for tool in ("git", "gh"):
        if not shutil.which(tool):
            sys.exit(f"{tool} isn't installed. See the top of this file for the two install commands.")
    auth = sh("gh", "auth", "status", check=False)
    if auth.returncode != 0:
        sys.exit("The GitHub CLI isn't logged in. Run:  gh auth login --scopes workflow")
    scopes = (auth.stdout or "") + (auth.stderr or "")
    if "Token scopes" in scopes and "workflow" not in scopes:
        sys.exit("The GitHub login needs the 'workflow' permission to upload the bot's schedule. Run:\n"
                 "    gh auth refresh -s workflow\nthen run this script again.")
    if not os.path.exists("live.py"):
        sys.exit("Run this from the unzipped isa_quant folder.")
    login = sh("gh", "api", "user", "--jq", ".login").stdout.strip()
    go_live(args, login) if args.go_live else set_up(args, login)


if __name__ == "__main__":
    main()
