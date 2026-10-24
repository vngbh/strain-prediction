# Claude-Codex Debate Transcripts (October 2026)

Verbatim prompts sent by Claude to Codex (`*_prompt.md`) and Codex's replies
(`*_codex.md`), kept for transparency: every design, audit, concession, and
rebuttal can be traced here. Claude's own reasoning and the decisions taken
are recorded in the pre-registrations and notes under `research/directions/`
and in `research/decisions.md`. Paths inside prompts point to a session
scratch directory that no longer exists; the probes they mention are in
`research/probes/` on the branches listed.

| Prefix | Topic | Branch / result file |
|---|---|---|
| `r1`-`r2` | Evaluation of the thesis and first research directions | `direction/calibration-transfer-limits` |
| `m1`-`m3` (+ `m1_claude_proposal.md`) | Modularizing the code and organizing docs | `main` (`src/strain/`, `AGENTS.md`, `tools/check_results.py`) |
| `r3`-`r6` | Calibration analyses F1-F5, G1-G6, timing, first cycle | `direction/calibration-transfer-limits` |
| `r7`-`r8` | Preconditioning pre-registration and audit | `2026-10-21-preregistration-preconditioning.md` |
| `s1`-`s5` | Latent-state direction and its gates | `direction/latent-state-model` |
| `l1`-`l6` | Reference apparent alignment v1/v2 | `direction/reference-lag` |
| `g1`-`g4` | Static-dynamic gap under label re-alignment | `direction/dynamic-gain-attribution` |
| `c1`-`c5` | Literature search and one-shot dataset0 confirmation | `direction/reference-lag` |
| `sy1`-`sy2` | Review of `research/SYNTHESIS.md` | `main` |

Gaps, disclosed: `s5` has no reply (Codex usage limit; its partial audit is
summarized in `2026-10-21-preregistration-latent-gates.md`). Every other round
is complete, and no reply was edited. A few early non-debate Codex calls (CLI
setup tests) are not included.
