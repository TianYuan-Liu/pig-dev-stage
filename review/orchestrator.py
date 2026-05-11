#!/usr/bin/env python3
"""Autonomous revision-loop orchestrator for BMC Genomics R1.

This script does not spawn agents itself — that is done by the Claude session
that drives the revision. The orchestrator provides:

  1. Quality-gate functions (LaTeX compile, bib immutability, paper/thesis sync,
     no-fabrication checks) that the loop calls after every commit.
  2. A `log_attempt` helper that appends a row to `revision_log.tsv`.
  3. A `run` entry point that exercises all gates against the current working
     tree — useful for manual verification.

Usage:
    python review/orchestrator.py gates       # run all gates against HEAD
    python review/orchestrator.py compile     # only LaTeX compile gate
    python review/orchestrator.py log <commit> <point> <gates> <status> <desc>
    python review/orchestrator.py bib-sha     # print current bib SHA-256
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
REVIEW_DIR = REPO_ROOT / "review"
PAPER_DIR = REPO_ROOT / "paper"
THESIS_DIR = REPO_ROOT / "thesis" / "MPhilThesis-Latex-Template"
BIB_PATH = PAPER_DIR / "pig-age-human" / "pig-age-human.bib"
LOG_PATH = REVIEW_DIR / "revision_log.tsv"

BIB_BASELINE_SHA256 = (
    "2f55092b47ba2a066b99b15edf1a8429a7c8498bfe9ed0b7d3309886ee0a94d9"
)


# ---------------------------------------------------------------------------
# Gate primitives
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return f"[{flag}] {self.name}: {self.detail}".rstrip(": ")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gate_bib_unchanged() -> GateResult:
    """The pig-age-human.bib file must never be modified by an agent."""
    if not BIB_PATH.exists():
        return GateResult("bib_unchanged", False, "bib file missing")
    actual = sha256_of(BIB_PATH)
    return GateResult(
        "bib_unchanged",
        actual == BIB_BASELINE_SHA256,
        f"sha256={actual[:12]}... (baseline={BIB_BASELINE_SHA256[:12]}...)",
    )


def gate_latex_compiles() -> GateResult:
    """paper.tex and supplementary.tex must compile to PDF with no errors."""
    failures: list[str] = []
    for tex in ("paper.tex", "supplementary.tex"):
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex,
        ]
        # Two passes so cross-refs resolve.
        for _ in range(2):
            proc = subprocess.run(
                cmd, cwd=PAPER_DIR, capture_output=True, text=True
            )
            if proc.returncode != 0:
                failures.append(f"{tex} exit={proc.returncode}")
                break
    return GateResult(
        "latex",
        not failures,
        ",".join(failures) if failures else "paper+supplementary compiled",
    )


def gate_response_compiles() -> GateResult:
    """response_to_reviewers.tex must compile."""
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "response_to_reviewers.tex",
    ]
    for _ in range(2):
        proc = subprocess.run(
            cmd, cwd=REVIEW_DIR, capture_output=True, text=True
        )
        if proc.returncode != 0:
            return GateResult(
                "response_latex",
                False,
                f"response_to_reviewers exit={proc.returncode}",
            )
    return GateResult("response_latex", True, "response compiled")


def gate_thesis_sync(commit: str | None = None) -> GateResult:
    """If the most recent commit touched paper/{paper,supplementary}.tex, it
    must also have touched the thesis. Returns PASS if nothing relevant
    changed."""
    rev = commit or "HEAD"
    proc = subprocess.run(
        ["git", "diff", "--name-only", f"{rev}~1", rev],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return GateResult("thesis_sync", True, "no parent (initial commit)")
    files = set(proc.stdout.split())
    touched_paper = any(
        f in {"paper/paper.tex", "paper/supplementary.tex"} for f in files
    )
    touched_thesis = any(
        f.startswith("thesis/MPhilThesis-Latex-Template/") for f in files
    )
    if touched_paper and not touched_thesis:
        return GateResult(
            "thesis_sync",
            False,
            f"paper changed but thesis untouched in {rev}",
        )
    return GateResult("thesis_sync", True, "paper/thesis in sync")


def gate_no_unresolved_citations() -> GateResult:
    """After pdflatex runs, .log file lists 'Citation X undefined'. Reject."""
    bad: list[str] = []
    for stem in ("paper", "supplementary"):
        log_file = PAPER_DIR / f"{stem}.log"
        if not log_file.exists():
            continue
        text = log_file.read_text(errors="ignore")
        for line in text.splitlines():
            if "Citation" in line and "undefined" in line:
                bad.append(line.strip())
    return GateResult(
        "citations",
        not bad,
        f"{len(bad)} undefined" if bad else "all citations resolved",
    )


def run_all_gates(commit: str | None = None) -> list[GateResult]:
    return [
        gate_bib_unchanged(),
        gate_latex_compiles(),
        gate_no_unresolved_citations(),
        gate_thesis_sync(commit),
    ]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_attempt(
    commit: str,
    reviewer_point: str,
    gates_passed: Iterable[str],
    status: str,
    description: str,
) -> None:
    """Append a row to revision_log.tsv. Never rewrites history."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text(
            "commit\treviewer_point\tgates_passed\tstatus\tdescription\n"
        )
    row = (
        f"{commit}\t{reviewer_point}\t"
        f"{','.join(sorted(set(gates_passed)))}\t"
        f"{status}\t{description}\n"
    )
    with LOG_PATH.open("a") as fh:
        fh.write(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    cmd = argv[1]
    if cmd == "gates":
        results = run_all_gates()
        for r in results:
            print(r)
        return 0 if all(r.passed for r in results) else 1
    if cmd == "compile":
        for r in (gate_latex_compiles(), gate_no_unresolved_citations()):
            print(r)
        return 0
    if cmd == "bib-sha":
        print(sha256_of(BIB_PATH))
        return 0
    if cmd == "log":
        if len(argv) < 7:
            print(
                "usage: orchestrator.py log <commit> <point> "
                "<gates(comma)> <status> <desc>"
            )
            return 2
        log_attempt(
            commit=argv[2],
            reviewer_point=argv[3],
            gates_passed=argv[4].split(","),
            status=argv[5],
            description=" ".join(argv[6:]),
        )
        print(f"logged {argv[3]} as {argv[5]}")
        return 0
    print(f"unknown command: {cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
