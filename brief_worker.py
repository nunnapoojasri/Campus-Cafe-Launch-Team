"""Run one café brief generation in an isolated child process."""

from __future__ import annotations

import json
import sys

from app import run_brief_generation


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Expected job id, assumptions JSON, and status file path.")
    job_id, assumptions_json, status_path = sys.argv[1:]
    run_brief_generation(job_id, json.loads(assumptions_json), status_path)


if __name__ == "__main__":
    main()
