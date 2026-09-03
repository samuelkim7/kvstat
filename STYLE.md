# Writing style

Rules for every piece of prose in this repo: README, docs, docstrings, changelog, commit messages.

## Say it plainly

1. **Say the concrete thing.** Name what the reader will see or get, not its category and not what it is *not*. "Shows which requests hold which blocks" says more than "shows what dashboards cannot".
2. **Name the real subject.** Use the noun the reader meets in the UI or the code. In kvtop, requests hold blocks and get preempted, so "which request", never "who".
3. **Prefer plain verbs to jargon.** A technical-sounding verb usually hides a plain one that says more: "rebuilds" and "cross-checks" over "reconstructs" and "reconciles".
4. **Keep the terms that carry information; cut the ones that only sound technical.** Prometheus, TTFT and `--kv-events-config` tell an operator something exact. "Reconstructed state" tells them nothing they can act on.
5. **End on what the reader gets.** A sentence about a mechanism earns its place by closing on the consequence for the reader, not on the mechanism itself.

## Shape the text

6. **Position by contrast when one sentence can do it.** "Dashboards show the totals; kvtop shows the requests behind them" places the tool faster than a list of features.
7. **Read it aloud once before it lands.** Repeated words, accidental rhyme and stumbles are invisible on the page and obvious in the ear.

## Keep it true over time

8. **Write nothing that goes stale on its own.** No status lines, no "coming soon", no counts or dates that need a hand to stay correct. If it changes every week, it belongs in the changelog or nowhere.
9. **Short beats complete.** A README is one page: what the tool does and how to run it. The reasoning behind decisions lives in the design docs, each fact in one place.
