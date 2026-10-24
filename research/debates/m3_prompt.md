Round 3, adversarial review of the IMPLEMENTED refactor (Claude implemented the plan you debated). Rules: no compliments, evidence only, read-only, under 500 words.

Review: src/strain/*.py, src/frozen_protocol_audit.py, tools/check_results.py, AGENTS.md, CLAUDE.md, research/decisions.md, README.md file map, requirements.txt, results/manifest.json. The original single-file script is backed up at the absolute path /private/tmp/claude-501/-Users-vngbh-Documents-Lab-strain-prediction/3df42ce0-5621-4e6e-b361-13433d3a8869/scratchpad/baseline/frozen_protocol_audit.py — compare function bodies for unintended changes.

Claude's claims to attack: (a) all 8 CSVs + 2 PNGs byte-identical to the pre-refactor run; (b) the 4 negative controls in gate 2 really exercise the code paths they claim (check monkeypatch targets vs where names are bound: e.g. audits.py imports calibration_prefix/session_signals by name); (c) the docs gate catches swapped numbers (Claude tested swapping 37.6/45.1 on a copy); (d) AGENTS.md preserved every rule from the old CLAUDE.md (old version: git show HEAD:CLAUDE.md, plus working-tree diff) without result numbers.

List concrete defects only (severity blocker/major/minor), each with file:line and a failure scenario. If a claim survives, say "survives" in one line.
