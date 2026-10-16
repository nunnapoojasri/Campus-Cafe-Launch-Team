"""Run one café brief generation in an isolated child process."""

from __future__ import annotations

import json
import sys

from app import run_brief_generation


def main() -> None:
    if len(sys.argv) not in {4, 6}:
        raise SystemExit(
            "Expected job id, assumptions JSON, status file path, and optional "
            "exercise options JSON and experiment name."
        )
    job_id, assumptions_json, status_path = sys.argv[1:4]
    options = json.loads(sys.argv[4]) if len(sys.argv) == 6 else None
    experiment = sys.argv[5] if len(sys.argv) == 6 else "brief"
    run_brief_generation(
        job_id,
        json.loads(assumptions_json),
        status_path,
        options=options,
        experiment=experiment,
    )


if __name__ == "__main__":
    main()
