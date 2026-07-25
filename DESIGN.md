---
name: Find-duplicates (browser review)
description: A flat, hairline-ruled review console where one registered ink stage delivers the sharpness verdict and a single blue means the file you keep.
colors:
  primary: "#024ad8"
  primary-bright: "#296ef9"
  primary-deep: "#0e3191"
  primary-soft: "#c9e0fc"
  on-primary: "#ffffff"
  ink: "#1a1a1a"
  ink-soft: "#292929"
  ink-deep: "#000000"
  on-ink: "#ffffff"
  paper: "#ffffff"
  cloud: "#f7f7f7"
  fog: "#e8e8e8"
  steel: "#c2c2c2"
  graphite: "#636363"
  charcoal: "#3d3d3d"
  hairline: "#e8e8e8"
  hairline-strong: "#c2c2c2"
  bloom-coral: "#ff5050"
  bloom-deep: "#b3262b"
  bloom-wine: "#5a1313"
  storm-mist: "#8ebdce"
  storm-sea: "#7fadbe"
  storm-deep: "#356373"
  error: "#b3262b"
typography:
  display-xxl:
    fontFamily: "system-ui, sans-serif"
    fontSize: "72px"
    fontWeight: 500
    lineHeight: 1
    letterSpacing: "-0.02em"
    fontVariant: "tabular-nums"
  display-lg:
    fontFamily: "system-ui, sans-serif"
    fontSize: "44px"
    fontWeight: 500
    lineHeight: 1
  display-md:
    fontFamily: "system-ui, sans-serif"
    fontSize: "32px"
    fontWeight: 500
    lineHeight: 1
    fontVariant: "tabular-nums"
  display-sm:
    fontFamily: "system-ui, sans-serif"
    fontSize: "24px"
    fontWeight: 500
    lineHeight: 1.17
  display-xs:
    fontFamily: "system-ui, sans-serif"
    fontSize: "20px"
    fontWeight: 500
    lineHeight: 1
  body-md:
    fontFamily: "system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.38
  body-emphasis:
    fontFamily: "system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 500
    lineHeight: 1.38
  caption-md:
    fontFamily: "system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  caption-bold:
    fontFamily: "system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 700
    lineHeight: 1.3
  caption-sm:
    fontFamily: "system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.33
  button-md:
    fontFamily: "system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: "0.7px"
  button-sm:
    fontFamily: "system-ui, sans-serif"
    fontSize: "12.6px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.09em"
rounded:
  none: "0px"
spacing:
  hair: "4px"
  tight: "8px"
  snug: "10px"
  base: "12px"
  cozy: "14px"
  gutter-sm: "16px"
  gutter: "20px"
  bay: "40px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    typography: "{typography.button-md}"
    rounded: "{rounded.none}"
    padding: "0 18px"
    height: "40px"
  button-primary-hover:
    backgroundColor: "{colors.primary-bright}"
    textColor: "{colors.on-primary}"
  button-primary-active:
    backgroundColor: "{colors.primary-deep}"
    textColor: "{colors.on-primary}"
  button-primary-disabled:
    backgroundColor: "{colors.fog}"
    textColor: "{colors.graphite}"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.button-md}"
    rounded: "{rounded.none}"
    padding: "0 18px"
    height: "40px"
  button-ghost-hover:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.on-ink}"
  button-quiet:
    backgroundColor: "transparent"
    textColor: "{colors.graphite}"
    typography: "{typography.button-md}"
    padding: "0 12px"
    height: "40px"
  button-quiet-hover:
    textColor: "{colors.primary}"
  input-text:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.body-md}"
    rounded: "{rounded.none}"
    padding: "0 12px"
    height: "40px"
  checkbox-checked:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.none}"
    size: "18px"
  queue-item:
    backgroundColor: "{colors.cloud}"
    textColor: "{colors.ink}"
    typography: "{typography.caption-md}"
    padding: "11px 16px"
  queue-item-active:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.on-ink}"
  candidate-tab:
    backgroundColor: "{colors.ink-soft}"
    textColor: "{colors.steel}"
    typography: "{typography.caption-bold}"
    padding: "10px 14px 0"
    width: "168px"
  candidate-tab-active:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
  stage:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.on-ink}"
    rounded: "{rounded.none}"
    height: "minmax(160px, 1fr)"
  notice-error:
    backgroundColor: "{colors.error}"
    textColor: "{colors.on-ink}"
    typography: "{typography.caption-md}"
    padding: "10px 20px"
  notice-dry:
    backgroundColor: "{colors.storm-deep}"
    textColor: "{colors.on-ink}"
    typography: "{typography.caption-md}"
    padding: "10px 20px"
  notice-done:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    typography: "{typography.caption-md}"
    padding: "10px 20px"
  toast:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.on-ink}"
    typography: "{typography.caption-md}"
    padding: "12px 16px"
---

# Design System: Find-duplicates (browser review)

This records the browser front end shipped in `static/` (HTML, CSS, vanilla JS,
no build step) and the `/api/*` fields it renders from. The Textual TUI is a
peer front end: it shares the status vocabulary (`Pending` / `Kept` / `Skipped`),
the close-call term, and the keyboard model, but none of this presentation.

## Overview

**Creative North Star: "The Registered Stage"**

Picking a keeper is a sharpness verdict, not a table lookup. The whole surface
is built around one ink-black stage that holds every candidate in a group at
identical framing, so flipping between them changes the pixels and nothing else.
The measurement table exists to corroborate a decision the eye already made; it
sits under the stage, opens only where the viewport can seat it without cutting
into the stage, and collapses to a single hairline header everywhere else. The
arrangement this refuses is the category default — a thumbnail grid beside a
spreadsheet.

The material is flat to the millimetre. There is no border-radius, no
box-shadow and no gradient anywhere in the stylesheet; the only occurrences of
those words are the comment at the top of `style.css` promising their absence.
Hierarchy is carried entirely by four flat grounds (paper, cloud, ink,
ink-soft), 1px hairlines, and type weight and size. White chrome sits over a
cloud queue rail; the stage and the candidate strip are the only dark regions,
and they are dark because a photo is being judged in them.

Colour is rationed and semantic. One saturated blue does the keeping. Coral
flags a decision that is genuinely close or a pick that has drifted from what is
on disk. Teal is dry-run and only dry-run. Red is a scan that failed. Nothing in
this palette is decorative.

**Key Characteristics:**
- Flat by commitment: zero radius, zero shadow, zero gradient
- Four grounds and a hairline vocabulary instead of elevation
- One saturated blue for the keeper, coral for close, teal for dry-run, red for failure
- The OS's own system font, with tabular figures on every comparable number
- The stage never animates; motion exists only on scan progress

## Colors

Neutral-dominant and near-monochrome, with three rationed signal hues that each
mean exactly one thing. Every `:root` value was pinned verbatim by the project
owner; none of it was sampled or derived from an existing asset.

### Primary
- **Signal Blue** (`{colors.primary}`): the keeper, and the controls and cells that
  produce a keeper. It fills Confirm keep, the confirmed dot in the queue, the
  kept segment of the review meter, the active candidate's index chip and score
  bar, the best value in a metric row, and the focus ring. It never decorates.
- **Bright Signal** (`{colors.primary-bright}`): the hover state of a primary
  button, the confirmed dot on an active (ink) row, and the scan progress fill.
- **Deep Signal** (`{colors.primary-deep}`): the pressed state of a primary or
  quiet button, and label text inside the tinted keeper column. Also aliased in
  `:root` as `--link-pressed`.
- **Soft Signal** (`{colors.primary-soft}`): the tint on the keeper's whole
  column in the measurement table, and the zoom readout on the ink stage.
- **On Primary** (`{colors.on-primary}`): text and the checkbox tick on blue.

### Secondary
- **Bloom Coral** (`{colors.bloom-coral}`): held in reserve for close-call
  emphasis on a dark ground; no component currently paints it.
- **Bloom Deep** (`{colors.bloom-deep}`): the ledger's close-call note, and the
  warning that a confirmed group's pick has drifted from what was actually
  moved. Same value as `{colors.error}`, deliberately kept as a separate name
  because the two roles are separate.

**The close call is stated once, where it can be acted on.** Most groups in a
real scan are close calls, so flagging them in the queue marked nearly every
row and the mark stopped meaning anything. The flag lives on the ledger header
of the group being looked at, plus the queue row's tooltip — never as a
standing mark down the rail.
- **Bloom Wine** (`{colors.bloom-wine}`): the hairline under an error notice.

### Tertiary
- **Storm Deep** (`{colors.storm-deep}`): the dry-run banner ground. Dry run is
  the only thing teal is allowed to say.
- **Storm Mist** (`{colors.storm-mist}`): the scan phase label on the ink stage.
- **Storm Sea** (`{colors.storm-sea}`): the score bar on an inactive candidate
  tab — a measured value that is not the current keeper.

### Neutral
- **Paper** (`{colors.paper}`): all chrome — command bar, ledger, decision bar,
  help sheet, inputs, and the active candidate tab. Also aliased as `--canvas`
  on `body`.
- **Cloud** (`{colors.cloud}`): the queue rail and the scan panel; the one step
  off white that says "this is the index, not the work".
- **Fog** (`{colors.fog}`): the empty track behind meters and score bars, the
  queue row hover, and disabled button strokes.
- **Steel** (`{colors.steel}`): text on the ink stage and the switcher, the
  skipped meter segment and skipped bar, and disabled ghost text.
- **Graphite** (`{colors.graphite}`): secondary text on paper — field labels,
  row headers, counts, quiet button rest state, and unmeasured (`n/a`) values.
- **Charcoal** (`{colors.charcoal}`): borders and hovers inside dark regions,
  and body copy in the help sheet.
- **Ink** (`{colors.ink}`): the stage ground, the active queue row, the toast,
  the skip link, and all primary text on paper.
- **Ink Soft** (`{colors.ink-soft}`): the candidate strip, one step off the
  stage so the strip reads as a separate surface without a rule doing the work.
- **Ink Deep** (`{colors.ink-deep}`): the hairlines that divide the dark
  regions, where a light hairline would glow.
- **Hairline** (`{colors.hairline}`) and **Hairline Strong**
  (`{colors.hairline-strong}`): the two-weight rule system — light for
  divisions inside a region, strong for the boundary between regions.
- **Error Red** (`{colors.error}`): a scan that failed, in the notice band and
  the error toast. Nothing else.

### Named Rules
**The Blue Means Keep Rule.** Blue is reserved for the keeper and for the
controls and cells that produce or mark one: Confirm, the confirmed dot, the
kept meter, the active candidate chip, the tinted keeper column, the winning
metric value, and the focus ring that reaches those controls. It is never used
to decorate, to divide, or to indicate mere activity — an active queue row goes
ink, not blue.

**The One Signal Per Hue Rule.** Teal is dry-run mode and nothing else. Red is a
failed scan and nothing else. Coral is a close call or a drifted pick and
nothing else. If a new state needs colour, it earns a new hue or goes neutral —
it does not borrow one of these three.

**The Shape-Not-Only-Colour Rule.** Every status that colour distinguishes also
differs in silhouette. In the queue, pending is a 10px open square, kept is a
filled 10px square, skipped is a 2px bar. Colour is the fast read; shape is the
guarantee.

## Typography

**Display Font:** system-ui
**Body Font:** system-ui
**Fallback stack:** sans-serif

`font: system-ui, sans-serif` resolves to whatever UI face the OS already has
loaded — San Francisco, Segoe UI, Roboto, or equivalent — so there is zero font
weight to ship, zero license to track, and no flash-of-unstyled-text to guard
against. `static/` stays HTML, CSS and vanilla JS with nothing in a `fonts/`
directory.

**Character:** Whatever the visiting OS's grotesque is, it is asked to do two
jobs at once: hold a 72px progress count on a black field without drama, and
set a 12px metric value that has to align digit-for-digit against five
siblings — a bar the system font stack's own hinting is left to clear on each
platform. The ramp is pinned verbatim as a set of CSS `font:` shorthands;
sizes are absolute px, never fluid.

### Hierarchy
- **Display XXL** (500, 72px/1, -0.02em, tabular): the live scan count on the
  ink stage. One use, and it is the loudest thing in the product.
- **Display LG** (500, 44px/1, tabular): the same count below 900px.
- **Display MD** (500, 32px/1, tabular): the number of groups left, at the top
  of the queue rail — the story's opening line.
- **Display SM** (500, 24px/1.17): the headline of an empty, error or
  nothing-selected stage message.
- **Display XS** (500, 20px/1): the help sheet title, and the queue count in the
  narrow layout.
- **Body MD** (400, 16px/1.38): the document default, stage message copy, and
  text input values.
- **Body Emphasis** (500, 16px/1.38): the scanned directory in the command bar,
  the kept filename in the decision sentence, and the score value in the ledger.
- **Caption MD** (400, 14px/1.5): queue rows, notices, ledger cells, the
  decision sentence, and help body copy. The workhorse.
- **Caption Bold** (700, 14px/1.3): the product mark (uppercase, 0.09em) and a
  candidate tab's filename.
- **Caption SM** (400, 12px/1.33): scan meta, tallies, candidate facts, and
  field labels (uppercase, 0.06em).
- **Button MD** (600, 14px/1.4, 0.7px, uppercase): every button label and the
  scan phase.
- **Button SM** (700, 12.6px/1, 0.06–0.09em, uppercase): the micro-label tier —
  stage HUD, table headers, ledger toggle, candidate tags and marks, notice
  tags, section headers in the help sheet.

### Named Rules
**The Tabular Comparison Rule.** Every number that a user compares against
another number carries `font-variant-numeric: tabular-nums` — metric cells,
scores, counts, meters, file facts, and both directory-meta figures. It is
applied per element, not globally: a number that is not compared does not need
it, and a number that is compared must never be allowed to drift.

**The Uppercase Micro-Label Rule.** Uppercase with letter-spacing between 0.06em
and 0.09em is reserved for labels under 14px: buttons, field labels, table
headers, HUD text, notice tags. Nothing at body size or larger is ever set in
caps.

## Layout

The page is a vertical flex column — command bar (56px), an optional scan panel,
an optional notice stack, then the app pane taking the rest. Flex rather than a
fixed grid on purpose: the panel and the notices come and go, and a
grid-template would drop the app pane into an implicit auto row the moment the
child count changed, collapsing the stage to its min-height.

The app pane is a two-column grid: a **queue rail of `--rail-w` (288px)** and
the review column. That one measure also sizes the command bar's first cell, so
the rail's right-hand hairline and the bar's first divider are a single
unbroken vertical rule rather than two verticals a few dozen pixels apart.
The review column is a four-row grid, `minmax(160px, 1fr) auto minmax(0, auto)
auto` — stage, candidate strip, ledger, decision bar. Short viewports squeeze
the stage to its 160px floor first and then the ledger (which owns its own
scroll, capped at `min(36vh, 360px)`); the candidate strip never gives ground,
because a clipped strip hides the filenames the decision is made on. Its
`overflow-x` zeroes its automatic minimum size, so its 100px floor is stated
explicitly.

Spacing rhythm is a 2px grid clustering at 8 / 10 / 12 / 14 / 16 / 20px, with
20px as the horizontal gutter for full-width chrome (command bar, notices,
ledger, decision bar) and 40px for the two full-bleed dark overlays. Nothing
uses a max-width container; the app fills the viewport and `body` never scrolls.
Prose is measured instead: 52ch on a stage message, 62ch in the help sheet.

**One breakpoint, at 900px.** Below it the rail rotates: the queue becomes a
horizontal strip above the review column, its rows lay out inline, and its
tally is dropped. The command bar wraps to two rows with the mark on
its own 44px line, the decision bar wraps its sentence above its actions, and
the dark overlays drop to 24px padding.

### Named Rules
**The Stage Floor Rule.** The stage is the only element allowed to take the
slack, and it is also the last to be sacrificed. Anything added to the review
column must fit in `auto` height or bring its own scroll — never push the stage
below 160px.

## Elevation & Depth

This system has no elevation. There are no shadows, no gradients, no
translucency over content, and no rounded corners; the sole `rgba()` in the
stylesheet is the help sheet's scrim (`rgba(26, 26, 26, 0.4)`), which darkens
the page behind a modal rather than lifting the modal off it.

Depth is done three ways. First, **flat grounds**: paper chrome, cloud rail,
ink stage, ink-soft strip — four values that say what kind of surface you are
looking at. Second, **two hairline weights**: `hairline` (#e8e8e8) divides
inside a region, `hairline-strong` (#c2c2c2) marks the boundary between regions,
and inside the dark regions `ink-deep` does the same job. Third, **type weight
and case**. A modal is not raised; it is a paper panel pinned to the right edge
with a 1px ink border.

Stickiness substitutes for layering in the measurement table: the header row,
the row-label column and the score row all pin themselves with `position:
sticky` and an opaque paper background, at z-indices 1–3, so the verdict row
stays on the ledger's bottom edge instead of scrolling out of sight.

### Named Rules
**The No-Lift Rule.** Nothing in this UI is raised. If a surface looks rounded,
shadowed or gradient-filled, it is a bug, not a style. State changes are shown
by swapping the ground (cloud → fog → ink) or by a hairline, never by elevation.

## Shapes

Zero radius everywhere (`{rounded.none}`) — buttons, inputs, checkboxes, tabs,
meters, chips, the modal panel and the toast are all true rectangles. Borders
are 1px and are the primary structural device; the ghost button, the checkbox,
the candidate index chip and the `n/a`-free metric grid are all defined by a
stroke rather than a fill.

Recurring geometry is deliberately small and literal: a 14×2px blue rule beside
the product mark, a 10px status square in the queue (2px tall when skipped), a
20px square index chip on a candidate tab, 3–6px bar tracks for scores and
meters, and a CSS-triangle disclosure caret on the ledger toggle that rotates
90° when open. The only non-rectangular shapes in the system are that caret and
the checkbox tick, both drawn from borders.

## Components

### Buttons
- **Shape:** true rectangle (0 radius), 40px tall, 1px transparent border so the
  ghost variant can take a stroke without shifting layout.
- **Primary:** signal blue ground, white label, 0 18px padding, uppercase Button
  MD. Hover goes bright blue; active goes deep blue; disabled goes fog on
  graphite. Used for exactly two things: Scan and Confirm keep.
- **Ghost:** transparent with a strong hairline stroke. Hover inverts to a solid
  ink block with white text. Used for Skip group.
- **Quiet:** no stroke, graphite label, 0 12px padding. Hover goes blue, active
  deep blue. Used for Help, Open full-res and Close.
- **Focus:** a 2px blue outline offset 1px, globally; inside the stage and the
  candidate strip the outline switches to white, because blue on ink is not
  visible enough to steer by.
- **Every button carries a `title` that says what it does and, where it has one,
  its shortcut.** Disabled buttons keep their title so the reason they are dim
  stays readable.

### Inputs / Fields
- **Style:** 40px tall, paper ground, 1px strong hairline, 0 12px padding, Body
  MD with tabular figures. Labels sit above in uppercase Caption SM graphite.
- **Focus:** the border turns blue and the outline offset drops to 0, so the
  ring lands on the border rather than outside it.
- **Checkbox:** an 18px appearance-none square with a strong hairline; checked
  fills blue and reveals a tick drawn from two 2px white borders rotated -45°.

### Navigation
- **Command bar:** a 56px paper strip divided by hairlines into three cells —
  the uppercase mark with its 14×2px blue rule, the scope button, and Help. The
  **scope** is the signature move: the scanned directory is simultaneously the
  label for what you are looking at and the control that changes it, one cell,
  not a path plus a button. It hovers cloud and inverts to ink when its panel is
  open, with its blue "Change" cue softening to `primary-soft` against ink.

### Queue rail (signature)
- 288px cloud column: a head block (groups left in Display MD, a 4px two-segment
  meter of kept and skipped, a tally line) over a scrolling list.
- Each row is a four-column grid — 2-digit index, status dot, filename, close
  flag — 11px 16px padding, divided by light hairlines. Hover is fog; active is
  a solid ink block with steel secondaries. Done rows drop their label to
  graphite.
- Status is shape first (see The Shape-Not-Only-Colour Rule), colour second.
- During a scan the list drops to 0.4 opacity and stops taking pointer events.

### Registered stage (signature)
- The ink field that the whole product is built to serve. Every candidate in the
  group is rendered as an absolutely positioned `<img>` in the same frame, laid
  out from the API's pixel dimensions, and switching candidates is an opacity
  swap with `transition: none`. The cross-fade is banned by name: it is exactly
  what would hide the difference being judged.
- Only the visible layer is exposed to assistive tech; the stacked siblings are
  `aria-hidden` with empty alt.
- Fit is contain by default. Clicking zooms to **inspect scale** — the group's
  largest file at true 1:1 — with every other candidate scaled to that same
  scene rectangle, so a smaller export renders visibly upscaled. Cursor tells
  the mode: `zoom-in`, `grab`, `grabbing`, `default` when nothing is zoomable.
- The HUD is two uppercase Button SM lines pinned to the top in steel, right
  side in `primary-soft`, `pointer-events: none`: group/file position on the
  left, zoom state and its instruction on the right.
- Scan state takes the same field: a storm-mist phase label, a 72px tabular
  count, and a 4px charcoal track at the bottom edge with a bright-blue fill.
- Empty, error and nothing-selected states render as a left-aligned Display SM
  headline with a 52ch steel paragraph, centred in the field. Nothing from the
  previous group survives into any of them.

### Candidate strip
- Ink-soft tabs separated by ink-deep hairlines, min 100px tall, flex-grow from
  a 168px basis but never below 152px — the strip scrolls rather than shrinking
  past the point where a filename stops being distinguishable.
- Each tab: a 20px bordered index chip, a two-line clamped filename (never
  single-line ellipsis — near-duplicate exports differ at the tail), a facts
  line, an optional uppercase tag, and a 3px score bar with its value.
- Active inverts to paper: chip fills blue, name goes ink, score bar goes blue.
  Inactive score bars are storm sea.

### Measurement ledger
- A real `<table>` with native semantics, on paper, Caption MD, tabular. Header
  row and label column are sticky; the score row is sticky to the bottom because
  it is the verdict the table builds toward, and it is separated by a 1px ink
  rule rather than a hairline.
- The keeper's entire column is tinted `primary-soft`; its header reads
  "N · keeping" and every column header is a button that re-picks.
- The best value in a scored row goes blue and bold; `n/a` goes graphite (not
  steel — steel on paper is ~1.8:1 and disappears entirely inside the tint) and
  carries a title naming the state without guessing its cause.
- Reference rows (dimensions, file size) are graphite, and the first scored row
  takes a strong-hairline top border to separate them.
- The header is a 40px hairline bar with an uppercase toggle and a caret. It
  opens by default only when the viewport is at least 940px tall; otherwise it
  ships collapsed to that one bar. Shortcut: M.

### Decision bar
- A paper strip with a strong hairline top: the consequence sentence on the
  left in Caption MD graphite (kept filename in Body Emphasis ink, drift warning
  in bloom-deep bold), actions right-aligned with 10px gaps.
- The sentence states the consequence in the present tense and names the
  destination. Under dry run it switches to the conditional voice ("Confirm
  would keep … and would move 5 files … dry run, so nothing actually moves"),
  because promising a move under a banner that says nothing will move is the
  wrong voice in the one place that describes a destructive action.
- The middle action is state-dependent: **Skip group** when pending, **Un-skip
  group** when skipped, **Undo keep** when confirmed — the last of which moves
  files back out of the destination.

### Notices and toast
- Notices are persistent full-width bands under the command bar, Caption MD with
  an uppercase Button SM tag: error red for a failed scan, storm deep for dry
  run, blue for "all reviewed". They are persistent by design — a scan failure,
  dry-run mode and a finished review must not time out.
- The toast is an ink block pinned bottom-left at 84px, clear of the decision
  bar, going error red on failure. It carries transient outcomes only.

### Help sheet
- A right-edge paper panel, `min(520px, 100%)` wide, 1px ink border, over a
  40%-ink scrim. Its head is a Display XS title and a quiet Close; its body is
  built from `/api/metrics-info` rather than hardcoded, so a change to metric
  weights or descriptions in the core reaches the UI without an edit here.

### Keyboard model
- **All bindings read `KeyboardEvent.code`, never `.key`.** Alternate layouts
  remap letter keys before the app sees them; this is a hard product constraint,
  established from a real failure, and it is shared with the TUI.
- Enter or C confirms; Delete, Backspace or S skips; 1–9 pick a candidate;
  arrows flip candidates and step groups; Shift+arrows pan the stage while
  inspecting at 1:1 (one tenth of the frame per press, so zoomed inspection is
  reachable without a pointer); Z toggles inspect; O opens full-res; M toggles
  the ledger; ? or F1 opens help; Escape closes.
- **Destructive keys ignore key repeat** (`e.repeat` returns early) and **yield
  to a focused control** (`button`, `a[href]`, `summary`), so holding Enter
  cannot walk through group after group and Enter on a focused button does that
  button's job. Arrows repeat on purpose.
- Focus moves to the stage when a group advances, so the next decision is
  already under the keyboard.

## Do's and Don'ts

### Do:
- **Do** keep every surface flat: 0 radius, no box-shadow, no gradient. Show
  state by swapping ground or adding a hairline.
- **Do** carry blue for the keeper and the controls that produce one, and let
  activity that is not a keeper go ink (`{colors.ink}`) instead.
- **Do** give every colour-coded status a second, non-colour signal — a
  silhouette, a word, or a position.
- **Do** set `font-variant-numeric: tabular-nums` on any number a user compares
  to another number.
- **Do** bind keys on `KeyboardEvent.code`, guard destructive keys against
  `e.repeat`, and let a focused control win the keystroke.
- **Do** keep uppercase and letter-spacing confined to labels of 14px and below.
- **Do** state the consequence of a destructive action in the sentence next to
  its button, and switch that sentence to the conditional voice under dry run.

### Don't:
- **Don't** animate the stage. Candidate switching stays `transition: none`; a
  cross-fade hides the exact difference being judged. Motion in this system is
  the scan progress fill (`width 180ms cubic-bezier(0.16, 1, 0.3, 1)`) and
  nothing else, and `prefers-reduced-motion` collapses all of it to 1ms.
- **Don't** let teal, error red or coral pick up a second meaning. A new state
  gets a new hue or goes neutral.
- **Don't** use steel for text on paper — it is ~1.8:1 and vanishes inside the
  tinted keeper column. Graphite is the floor for secondary text on light
  grounds.
- **Don't** truncate a filename at its tail. Near-duplicate exports differ at
  the end of the name; truncate in the middle or clamp to two lines.
- **Don't** add a font, an icon set, or a stylesheet from a CDN, and don't
  introduce a build step or a vendored font file. `static/` is plain HTML, CSS
  and vanilla JS; typography rides `system-ui` — whatever the OS already has.
- **Don't** push the review column past what `auto` rows can hold — the stage's
  160px floor and the candidate strip's 100px floor are both load-bearing.
- **Don't** let a toast cover the decision sentence, and don't demote a
  persistent condition (scan failure, dry run, review complete) to a toast.
