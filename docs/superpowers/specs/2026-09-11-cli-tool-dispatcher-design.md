# `srclight tool` — a CLI dispatcher over the MCP tools

*2026-09-11 — design*

## Problem

An agent working a large C codebase (hundreds of thousands of symbols) hit two
distinct failures using `find_pattern` through MCP. They are worth separating,
because they call for different fixes.

**Overflow.** `find_pattern` returns the full JSON: for every match, the
containing symbol plus its `matched_lines`, each carrying the complete source
line. Measured in one session:

| Query shape | Volume |
|---|---|
| One frequently called helper (kind=function, limit=300) | 161 870 characters — refused by the harness, spilled to a file |
| A common struct field access (language=c, kind=function, limit=80) | 74 654 characters — same |
| An alternation over two frequent identifiers (limit=40) | ~10k tokens swallowed at once |
| An alternation over two receiver spellings assigning a field (limit=60) | ~10.1k tokens |

When it overflows, the harness writes the result to a file and requires reading
it back in full — so the context is paid twice. The agent worked around it by
querying `.srclight/index.db` in SQL from a sandbox, which is the signal that
the tool surface is missing something.

**Silent truncation.** `find_pattern_in_symbols` stops at `limit` with no
signal (`src/srclight/db.py:1866`, `if len(results) >= limit: break`). On the
last query above the agent got `match_count: 60` with `limit: 60`, concluded
the sweep had converged, and was wrong: at least one symbol matching the
pattern was absent from the list, and only a separate SQL query over the index
revealed it.

The confusion is not the agent's. `match_count` carries two different meanings
in one response — at the root it counts **symbols** returned, capped by `limit`
(`src/srclight/server.py:2356`); inside each symbol it counts **matched lines**
(`src/srclight/db.py:1863`). `match_count: 60` reads as "60 matches found"
and means "60 symbols, and we stopped".

## Goals

1. An agent can run any srclight tool from a shell, so that a sandbox
   (context-mode and equivalents) executes it and the bytes never enter the
   agent's context.
2. The CLI follows the MCP surface automatically — a tool added to the server
   appears in the CLI with no further work.
3. A truncated result says so.

**Governing constraint: additive only.** Nothing existing breaks. No tool,
field, argument or output shape is renamed, removed or reshaped; every change
is a new command, a new optional argument, or a new key in a response. A caller
written against today's srclight must keep working untouched.

## Non-goals

- **`total_matches`.** Knowing the exact count means removing the early exit
  and running the regex over every symbol's content — seconds on a large repo.
  `truncated` plus `offset` lets a caller page to the end and learn the total
  as a by-product, only when it actually needs it.
- **A `count_only` mode.** It is dangerous precisely because it is convenient:
  an agent would call it reflexively "just to size the search" and pay the full
  scan every time. We would be trading a correctness trap for a performance
  trap. Add it if a real need shows up.
- **A compact `format: "locations"`.** The chosen consumer is a sandbox, which
  already keeps volume out of context. This becomes relevant again the day an
  agent without a sandbox calls the tool directly.
- **`srclight query` (read-only SQL).** The dispatcher covers the observed
  need. Making the index schema a public contract is a decision to take on its
  own merits, not as a passenger.

## Design

### A. `srclight tool` — dynamic dispatch

```bash
srclight tool --list                       # every tool: name + one-line summary
srclight tool find_pattern --help          # derived from the tool's JSON Schema
srclight tool find_pattern --pattern 'this->timer' --kind function --limit 80
```

**Mechanism.** The command imports `srclight.server`, configures the target
repo exactly as `serve` does (`src/srclight/cli.py:317-330` — `--workspace`,
else `--db`, else walk up from the cwd), then dispatches in-process:

```python
result = asyncio.run(mcp.call_tool(name, arguments))
click.echo(result.content[0].text)
```

Verified: `mcp.list_tools()` returns 43 tools, each with `name`, `description`
and `input_schema` (standard JSON Schema with types, defaults and `required`);
`mcp.call_tool(name, args)` returns a `CallToolResult` whose
`content[0].text` is the tool's JSON string.

No running server is required and no per-tool code is written — which is the
whole point: the CLI is a façade over the same registry the MCP clients see.

**Argument parsing.** Click cannot declare options at runtime on a fixed
command, so `srclight tool` takes free-form `--key value` pairs and validates
them against the tool's own `input_schema`:

- `type: string` → passed through
- `type: integer` / `number` → converted, with a clear error on failure
- `type: boolean` → `--flag` sets true, `--flag=false` sets false
- `anyOf: [T, null]` → optional, omitted when absent so the tool's own default
  applies
- unknown key, or a missing `required` key → error listing the accepted keys

Validating against the schema rather than a hand-written parser means the CLI
rejects exactly what an MCP client would reject, with the same wording.

**Output.** The tool's JSON goes to stdout unchanged. Diagnostics go to stderr.
Exit code is 0 on success, 1 when the tool's JSON carries an `error` key or
`CallToolResult.isError` is set, 2 on a usage error (unknown tool, bad
argument). A sandbox can therefore branch on the exit code without parsing.

**Stability.** The CLI names and arguments are the MCP names and arguments, by
construction. This is a deliberate trade: zero maintenance in exchange for the
MCP surface becoming a CLI contract. `srclight tool --list` and
`srclight tool <name> --help` state that the surface tracks the server and may
change with it.

### B. Truncation that announces itself

Three additions in the `find_pattern` path. Nothing existing is renamed, removed
or reshaped: a caller written against today's behaviour sees the same results in
the same keys.

**`truncated`, computed by asking for one more.** `find_pattern_in_symbols`
scans until it holds `limit + 1` matching symbols. Holding `limit + 1` proves
more exist: it returns the first `limit` and reports truncation. This is exact,
not a heuristic, and it preserves the early exit — the extra cost is the scan up
to *one* further matching symbol.

**`offset`.** A new parameter on both `find_pattern` (so MCP callers get it,
not only the CLI) and `find_pattern_in_symbols`. It skips the first N matching
symbols, so a caller that sees `truncated: true` can page. Ordering is already
deterministic
(`ORDER BY f.path, s.start_line`, `src/srclight/db.py:1831`), so paging is
stable as long as the index does not change under it.

**`match_count` is left alone.** An earlier draft renamed the root field to
`symbols_returned`, on the grounds that a name which means "symbols" while an
identically named per-symbol field means "lines" is what produced the wrong
conclusion. That rename is dropped: this change is additive only, and no
existing field, name or shape moves.

What cost the agent a false conclusion was not the word `match_count` — it was
that nothing said the list was incomplete. `truncated: true` removes that harm.

**`matched_lines_total` supplies the count the name implies.** `truncated`
fixes the dangerous failure but leaves a quiet one: an agent reads
`match_count: 60` at the root, sees `match_count: 3` inside each symbol, and
writes "60 matches found" when there were 60 symbols and 214 matching lines.
That error never crashes and nobody notices — the worst kind in an
agent-facing API.

Adding a synonym (`symbols_returned`) would only move the question: which of
the two is authoritative? Adding the *missing* count closes it instead. Both
numbers are present and they differ, so nothing has to be guessed: a reader
after line counts finds them one key below, and the value is not 60.

It is free — the per-symbol counts are already in hand, and summing them costs
nothing. When `truncated` is true it is a floor over the symbols returned, not
a repo-wide total; the docstring says so. That is consistent with refusing
`count_only`: no code path ever pays the full scan.

Resulting shape — every existing key unchanged, three added:

```json
{
  "pattern": "this->timer",
  "match_count": 60,
  "matched_lines_total": 214,
  "truncated": true,
  "offset": 0,
  "file_count": 37,
  "by_file": { "...": [] }
}
```

A caller written against today's response keeps working and ignores the new
keys. The docstring states plainly what each count covers.

## Testing

Both parts are testable without a network and without a running server.

**Dispatcher** — over a small indexed fixture repo, via `CliRunner`:

- `--list` names every tool `mcp.list_tools()` reports, so the two cannot drift
- a tool added to the registry at test time appears in `--list` without code
  changes (this is the "follows on its own" claim, and it deserves a test)
- arguments convert per schema: integer, boolean, nullable-omitted
- an unknown argument and a missing required one each fail with exit code 2 and
  name the accepted keys
- a tool returning an `error` key exits 1
- stdout is the tool's JSON and nothing else — a sandbox parses it directly

**Truncation** — at the `db.find_pattern_in_symbols` level, where the boundary
lives:

- more matches than `limit` → exactly `limit` results and `truncated: true`
- exactly `limit` matches → `truncated: false` (the off-by-one that makes the
  whole thing pointless if wrong)
- fewer than `limit` → `truncated: false`
- `offset` skips the right symbols and stays consistent with a single larger
  query
- the scan still stops early: with `limit=1` and many matches, the number of
  symbols whose content is regex-scanned stays bounded
- `matched_lines_total` equals the sum of the returned symbols' `match_count`,
  and differs from the root `match_count` whenever any symbol matched more than
  one line — the case that makes the two numbers worth distinguishing

**Compatibility** — the constraint deserves its own test, or it erodes:

- the same query, before and after, returns identical results in identical keys;
  the response gains `truncated` and `offset` and loses nothing
- `find_pattern` called with exactly today's arguments behaves as it does today

## Risks

**The MCP surface becomes a CLI contract.** Accepted knowingly. Renaming an MCP
tool or one of its arguments is now also a CLI break. The mitigation is
honesty, not machinery: say so in `--help`.

**`offset` paging is not snapshot-isolated.** A reindex between two pages can
shift results. Acceptable for a search tool; worth a sentence in the docstring
rather than a transaction.

**In-process import cost.** `srclight tool` pays the server module's import on
every invocation. Tolerable for a sandbox that runs it a handful of times; if it
becomes a problem, the answer is batching in the sandbox, not a daemon.
