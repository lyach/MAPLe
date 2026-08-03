"""Resolves the user's DRAC Slurm billing account: env var, then cached
file, then interactive prompt.
"""

from __future__ import annotations

import json
import os
import sys

from .config import DRAC_CREDENTIALS_FILE


def _load_saved_account() -> str | None:
    """Read the cached account, or None if missing/corrupted."""
    if not DRAC_CREDENTIALS_FILE.exists():
        return None
    try:
        data = json.loads(DRAC_CREDENTIALS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("account") or None


def _save_account(account: str) -> None:
    """Cache the account locally for future runs."""
    try:
        DRAC_CREDENTIALS_FILE.write_text(json.dumps({"account": account}, indent=2))
    except OSError as exc:
        print(f"Warning: couldn't save DRAC account for next time ({exc}).")


def clear_saved_drac_account() -> bool:
    """Delete the cached account. Returns True if a file was removed."""
    if DRAC_CREDENTIALS_FILE.exists():
        DRAC_CREDENTIALS_FILE.unlink()
        return True
    return False


def _prompt_for_account() -> str:
    """Interactively ask for the user's Slurm billing account."""
    print()
    print("Submitting an annotation job to a DRAC cluster needs your Slurm billing")
    print("account (e.g. 'def-yourpi' or 'rrg-yourpi') -- this is tied to")
    print("your PI's allocation, so it's asked for locally rather than")
    print("hardcoded in this repo.")
    print()
    print("Don't know yours? SSH into the cluster and run:")
    print("    sacctmgr show associations user=$USER format=account%30")
    print("Prefer a CPU account if you have both CPU and GPU variants --")
    print("this pipeline needs no GPU.")
    print()

    while True:
        account = input("Your Slurm billing account: ").strip()
        if account:
            break
        print("That was empty -- try again.")

    print()
    answer = input("Save this locally so you're not asked again? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        _save_account(account)
        print(f"Saved to {DRAC_CREDENTIALS_FILE.name} (this file is gitignored -- never commit it).")
    print()

    return account


def resolve_drac_account() -> str:
    """Returns the Slurm billing account: env var, cache, or prompt."""
    env_account = os.environ.get("DRAC_ACCOUNT", "").strip()
    if env_account:
        return env_account

    saved = _load_saved_account()
    if saved:
        return saved

    if not sys.stdin.isatty():
        raise RuntimeError(
            "No DRAC billing account is configured, and this isn't an "
            "interactive session to prompt for one. Set the DRAC_ACCOUNT "
            "environment variable before running this script in a "
            "non-interactive context."
        )

    return _prompt_for_account()
