# Task: Add Next/Previous navigation buttons in preview UI — for segments, and separately for candidates

## Problem

In the current preview UI (`src/App.tsx`), comparing segments (or a rejected
segment's discovered candidates, from the deferred-candidate-recovery feature)
requires scrolling up/down to find and click each one manually. This is slow
and annoying when reviewing many segments or comparing several candidates for
one rejected segment.

## What to build

Add two independent pairs of Next/Previous buttons to the preview UI, positioned
directly below the video comparison view (not requiring any scroll to reach):

### 1. Segment Next/Previous (always present, for every segment)

- A "Previous" and "Next" button pair that steps through the job's segment list
  (in the same order they already appear in the preview/timeline today) one at
  a time.
- This pair is shown for **every** segment being previewed, regardless of
  whether it has candidates or not — its only job is to let the user move
  between segments without scrolling.
- Clicking Next/Previous updates the preview to show that segment's matched
  movie/short-clip frames, exactly as if the user had scrolled to and clicked
  that segment manually today.
- Disable (or hide) the "Previous" button on the first segment and "Next" on
  the last segment, consistent with how other disabled/boundary states are
  handled elsewhere in this UI.

### 2. Candidate Next/Previous (only present when the segment has candidates)

- A second, visually distinct "Previous" / "Next" button pair that appears
  **only** for segments that have candidate data available (i.e., segments that
  were rejected during the main VLM pass and had background candidates
  discovered for them, per the deferred-candidate-recovery feature — fetch this
  via the candidate-fetching endpoint already added for that feature, e.g.
  `GET /api/match/:matchJobId/candidates/:segmentId`).
- For segments with no candidate data, this second button pair must not render
  at all — its mere presence is how the user recognizes "this segment had
  alternates that were checked." Do not show it disabled/greyed-out; omit it
  entirely for segments without candidates.
- Clicking this pair steps through that segment's list of candidates (up to 10,
  per the existing candidate-discovery feature) one at a time, updating the
  preview to show the candidate's movie-side frame/timestamp so the user can
  visually compare it against the short clip — without affecting or navigating
  away from which segment is currently selected via pair 1.
- Include a small indicator near this button pair showing position within the
  candidate list (e.g. "Candidate 3 of 10"), consistent with any existing
  position-indicator patterns already used in this UI (check how segment
  position/count is currently shown, if at all, and match that style).
- Disable/hide "Previous" at the first candidate and "Next" at the last
  candidate for that segment, same boundary handling as pair 1.

### Layout

- Both button pairs go directly beneath the video comparison area, with no
  scrolling required to reach either from the currently visible segment view.
- The segment Next/Previous pair (1) should read as the primary/always-there
  control; the candidate Next/Previous pair (2), when present, should be
  visually secondary/distinct (e.g. smaller, or in its own labeled sub-section)
  so the two aren't confused with each other.

## Non-negotiable constraints

- Do NOT change the matching engine, VLM verification logic, batching, or the
  final result JSON shape — this task is UI-only, consuming data that already
  exists (segment list) or was added by the separate candidate-discovery
  feature (per-segment candidate list via its own endpoint). Do not merge
  candidate data into the primary result JSON as part of this task.
- Do NOT change how segments or candidates are currently fetched/computed —
  only add navigation controls consuming that existing data.
- Preserve existing behavior for clicking a segment directly from any existing
  list/timeline view (e.g. clicking a specific segment in a sidebar or
  timeline, if that already exists) — the new Next/Previous buttons are an
  addition, not a replacement for existing selection methods.
- Keep this consistent with the rest of the app's existing UI conventions
  (button styles, disabled-state styles, spacing) rather than introducing a new
  visual pattern — check `src/App.tsx` for existing button/control styling and
  match it.

## Acceptance criteria

1. Open a completed match job's preview. Confirm a Next/Previous pair is
   visible directly below the video comparison view for the currently-selected
   segment, with no scrolling needed.
2. Click Next repeatedly — confirm it steps through all segments in order,
   updating the preview each time, and disables/hides appropriately at the last
   segment. Same for Previous at the first segment.
3. Select a segment known to have had candidates discovered for it (a
   previously-rejected segment from the deferred-candidate-recovery feature).
   Confirm a second Next/Previous pair appears near the first, with a position
   indicator (e.g. "Candidate 1 of 10"), and stepping through it updates the
   preview to show each candidate's movie-side frame without changing which
   segment is selected in pair 1.
4. Select a segment that was accepted on the first VLM attempt (no candidates
   were ever discovered for it). Confirm the candidate Next/Previous pair (2)
   does not render at all for this segment — only the segment pair (1) is
   present.
5. Confirm no changes occurred to the underlying match/VLM logic, result JSON
   shape, or candidate-discovery behavior — this task only adds UI navigation
   on top of existing data.
