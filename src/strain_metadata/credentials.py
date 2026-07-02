"""
credentials.py
===============
NCBI requires every Entrez API user to identify themselves with an email
address (an API key is optional, but raises the allowed request rate).

Anyone cloning this repo needs a way to supply that email without editing
source code. This module resolves it in three steps, trying each in order
and stopping at the first one that works:

    1. Environment variables (NCBI_ENTREZ_EMAIL / NCBI_API_KEY) -- best for
       CI, cron jobs, or anyone who already manages secrets this way.
    2. A small local JSON file saved from a previous run (gitignored, never
       committed, lives at the project root as .ncbi_credentials.json).
    3. An interactive prompt, asked only once -- after which step 2 takes
       over on every subsequent run.

This means `python scripts/get_strain_metadata.py` works for a brand new
collaborator with zero setup beyond typing their email when asked.
"""

from __future__ import annotations

# json: read/write the small local credentials cache as a plain JSON file.
import json
# re: a quick sanity check that what the user typed looks like an email.
import re
# os: read the environment variables NCBI_ENTREZ_EMAIL / NCBI_API_KEY.
import os
# sys: check whether we're attached to an interactive terminal before
# trying to prompt (so this doesn't hang forever in a non-interactive job).
import sys
# Optional: type hint meaning "this value or None".
from typing import Optional

# Re-use the single shared path constant rather than redefining it here.
from .config import CREDENTIALS_FILE

# A deliberately loose email check (NCBI doesn't actually verify the
# address, it's just used to identify you in case they need to contact you
# about unusual API usage) -- this just catches obvious typos like a
# missing "@" or ".".
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _load_saved_credentials() -> dict:
    """Read the local credentials cache, if it exists. Returns an empty
    dict (rather than raising) if the file is missing or corrupted, since
    a bad cache should just fall through to re-prompting, not crash."""
    if not CREDENTIALS_FILE.exists():
        return {}
    try:
        # .read_text() loads the whole (tiny) file as a string; json.loads
        # parses that string into a Python dict.
        return json.loads(CREDENTIALS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_credentials(email: str, api_key: Optional[str]) -> None:
    """Write email/API key to the local cache so future runs don't ask
    again. Failure to save (e.g. read-only filesystem) is non-fatal --
    we just log it and move on, since the run itself can still proceed."""
    payload = {"email": email}
    if api_key:
        payload["api_key"] = api_key
    try:
        # indent=2 just makes the file human-readable if someone opens it.
        CREDENTIALS_FILE.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        print(f"Warning: couldn't save credentials for next time ({exc}).")


def clear_saved_credentials() -> bool:
    """Delete the local credentials cache, e.g. because a saved email was
    a typo. Returns True if a file was actually removed."""
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()
        return True
    return False


def _prompt_for_credentials() -> tuple[str, Optional[str]]:
    """Interactively ask the user for their NCBI email (required) and API
    key (optional), looping on the email until it looks valid."""
    print()
    print("NCBI requires an email address to identify API users.")
    print("This is a one-time setup -- it will be saved locally on this")
    print("machine so you won't be asked again.")
    print()

    # Loop until the user types something that at least looks like an
    # email address, rather than silently accepting garbage input.
    while True:
        email = input("Your email address: ").strip()
        if _EMAIL_PATTERN.match(email):
            break
        print("That doesn't look like a valid email address -- try again.")

    print()
    print("An NCBI API key is optional, but raises your rate limit from")
    print("~3 to ~10 requests/second. Get one at:")
    print("    https://www.ncbi.nlm.nih.gov/account/settings/")
    api_key = input("API key (press Enter to skip): ").strip() or None

    print()
    answer = input("Save these locally so you're not asked again? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        _save_credentials(email, api_key)
        print(f"Saved to {CREDENTIALS_FILE.name} (this file is gitignored -- never commit it).")
    print()

    return email, api_key


def resolve_ncbi_credentials() -> tuple[str, Optional[str]]:
    """Top-level entry point used by the rest of the pipeline. Returns
    (email, api_key) -- api_key may be None.

    Raises RuntimeError only if no credentials are available AND we're not
    attached to an interactive terminal (so a cron job or CI run fails
    fast with a clear message instead of hanging on input())."""

    # --- Tier 1: environment variables ---
    env_email = os.environ.get("NCBI_ENTREZ_EMAIL", "").strip()
    if env_email:
        env_api_key = os.environ.get("NCBI_API_KEY", "").strip() or None
        return env_email, env_api_key

    # --- Tier 2: previously-saved local file ---
    saved = _load_saved_credentials()
    if saved.get("email"):
        return saved["email"], saved.get("api_key")

    # --- Tier 3: interactive prompt (only reachable if tiers 1 and 2 missed) ---
    # sys.stdin.isatty() is True only when input() can actually wait for a
    # human to type something (i.e. not inside a cron job, CI pipeline, or
    # `python script.py < /dev/null`).
    if not sys.stdin.isatty():
        raise RuntimeError(
            "No NCBI email is configured, and this isn't an interactive "
            "session to prompt for one. Set the NCBI_ENTREZ_EMAIL "
            "environment variable (and optionally NCBI_API_KEY) before "
            "running this script in a non-interactive context."
        )

    return _prompt_for_credentials()
