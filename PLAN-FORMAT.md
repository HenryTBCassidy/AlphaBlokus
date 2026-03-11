# AlphaBlokus — Plan Document Format

How we write implementation plans in `docs/plans/`. Companion to the [Style Guide](STYLE-GUIDE.md), which covers code conventions.

Last updated: 2026-03-11

---

## Purpose

Plan docs are the bridge between "what we want to do" and "actually doing it." They serve as:

1. **A checklist** — track progress, know what's done and what's next.
2. **A reference** — understand why a change is needed and what it touches.
3. **A commit guide** — each checklist row should map to roughly one commit.

## Structure

Every plan doc follows this layout:

```
# Title

One-paragraph intro: what this plan covers, any prerequisites,
links to companion docs.

---

## Checklist

Table with columns: #, Item, Effort, Priority, Done.
Items numbered sequentially (S1, S2, ... or B1, B2, ... or Step 1, 2, ...).
Ordered by execution sequence, not by topic.

---

## S1. First Item Title

Detailed description: current state, what's wrong, what the fix is,
code examples, effort estimate.

---

## S2. Second Item Title

...and so on, one section per checklist row.
```

### Key rules

1. **Checklist at the top.** It's the abstract — you should be able to read just the table and know the full scope. Scroll down for details.

2. **Section numbers match checklist IDs.** If the checklist says S8, the section heading says `## S8.` No exceptions. This avoids the confusion of S8 linking to section 3 because topics were grouped differently.

3. **One section per checklist item.** Don't group multiple checklist items under one section heading. If two items are related, they can reference each other, but each gets its own section.

4. **Execution order, not topic order.** The checklist is ordered to minimise merge pain and dependency issues. Sections follow that same order. If you need to reorganise the sequence, renumber everything — don't leave gaps or out-of-order IDs.

5. **Each row ~ one commit.** The checklist should be granular enough that each item is a single, reviewable commit. If an item takes more than ~2 hours, consider splitting it.

6. **Effort and priority on every row.** Even rough estimates help with planning. Use: High/Medium/Low for priority, time estimates for effort.

7. **Done column.** Mark with ✅ as items are completed. This is the primary progress tracker.

## Checklist Table Format

```markdown
| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| S1 | Short description of the task | 30 min | High | ✅ |
| S2 | Another task | 1 hour | Medium | |
```

Optional extra columns (if useful for that plan):
- **Files** — which files are touched
- **Depends on** — prerequisite steps

## Section Format

Each section should include whatever subset of these is relevant:

- **Current state** — what exists now, what's wrong with it
- **Fix / Recommendation** — what to do about it
- **Code examples** — concrete before/after or pseudocode
- **Action items** — sub-tasks within the section (if the item is complex)
- **Estimated effort** — repeat from the checklist for quick reference

Don't pad sections with filler. If a fix is "change `> 0` to `>= 0` in 4 places," that's the whole section. Short is fine.

## Naming Conventions

- **Prefix IDs by plan:** `S1–S12` for structural refactor, `B1–B25` for bug fixes, plain `Step 1–18` for encoding plan. Pick a prefix and stick with it.
- **File names:** lowercase, hyphenated: `structural-refactor.md`, `bug-fixes.md`, `multi-channel-board-encoding.md`.
- **Location:** always in `docs/plans/`.

## Lifecycle

1. **Draft:** Write the plan, get agreement on scope.
2. **Execute:** Work through the checklist, marking items done.
3. **Archive:** When all items are done, the plan stays in `docs/plans/` as a historical record. Don't delete completed plans — they explain why the code looks the way it does.

## Anti-patterns

- **Checklist at the bottom.** Nobody scrolls past 10 sections to find the progress tracker.
- **Section numbers that don't match checklist IDs.** Causes confusion every time someone cross-references.
- **Grouping multiple checklist items under one section.** Makes it unclear which section describes which item.
- **Topic-ordered sections with execution-ordered checklist.** Pick one ordering and use it for both.
- **No effort estimates.** "This will take a while" is not a plan.
- **Giant monolithic items.** If a checklist row says "Implement the entire game" it's not useful. Break it down.
