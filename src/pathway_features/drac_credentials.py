"""
drac_credentials.py
====================
Submitting a job to a DRAC (Digital Research Alliance of Canada) HPC cluster
requires a Slurm billing account string (e.g. "def-yourpi" or "rrg-yourpi")
-- this is tied to a specific PI's allocation, so it can't be hardcoded in a
repo meant to be reused outside one lab.

Resolved in three steps, trying each in order and stopping at the first one
that works, mirroring strain_metadata/credentials.py's NCBI email pattern:

    1. Environment variable (DRAC_ACCOUNT) -- best for CI or anyone who
       already manages secrets this way.
    2. A small local JSON file saved from a previous run (gitignored, never
       committed, lives at the project root as .drac_credentials.json).
    3. An interactive prompt, asked only once -- after which step 2 takes
       over on every subsequent run.

This means a new collaborator (in or outside the lab) gets prompted for
their own account the first time they run prepare_modular_manifest.py,
instead of the script silently submitting jobs against someone else's
allocation.
"""

from __future__ import annotations

import json
import os
import sys

from .config import DRAC_CREDENTIALS_FILE


def _load_saved_account() -> str | None:
    """Read the local credentials cache, if it exists. Returns None (rather
    than raising) if the file is missing or corrupted, since a bad cache
    should just fall through to re-prompting, not crash."""
    if not DRAC_CREDENTIALS_FILE.exists():
        return None
    try:
        data = json.loads(DRAC_CREDENTIALS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("account") or None


def _save_account(account: str) -> None:
    """Write the account to the local cache so future runs don't ask again.
    Failure to save (e.g. read-only filesystem) is non-fatal -- we just log
    it and move on, since the run itself can still proceed."""
    try:
        DRAC_CREDENTIALS_FILE.write_text(json.dumps({"account": account}, indent=2))
    except OSError as exc:
        print(f"Warning: couldn't save DRAC account for next time ({exc}).")


def clear_saved_drac_account() -> bool:
    """Delete the local credentials cache, e.g. because a saved account was
    a typo. Returns True if a file was actually removed."""
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
    """Top-level entry point used by the rest of the pipeline. Returns the
    Slurm billing account string.

    Raises RuntimeError only if no account is available AND we're not
    attached to an interactive terminal (so a cron job or CI run fails fast
    with a clear message instead of hanging on input())."""

    # --- Tier 1: environment variable ---
    env_account = os.environ.get("DRAC_ACCOUNT", "").strip()
    if env_account:
        return env_account

    # --- Tier 2: previously-saved local file ---
    saved = _load_saved_account()
    if saved:
        return saved

    # --- Tier 3: interactive prompt (only reachable if tiers 1 and 2 missed) ---
    if not sys.stdin.isatty():
        raise RuntimeError(
            "No DRAC billing account is configured, and this isn't an "
            "interactive session to prompt for one. Set the DRAC_ACCOUNT "
            "environment variable before running this script in a "
            "non-interactive context."
        )

    return _prompt_for_account()
