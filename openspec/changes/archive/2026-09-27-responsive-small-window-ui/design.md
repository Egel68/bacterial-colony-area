## Context

See `proposal.md` for motivation and `specs/responsive-user-interface/spec.md` for the user-visible contract. The UI is implemented with PyQt6 and currently uses multiple fixed-width/fixed-minimum widgets, static spacing/padding styles, and horizontal toolbars. The analysis window has a fixed-width control pane; the testing window places all groups in a non-scrollable vertical layout; the labeling window fixes its side panel and has a long single-row toolbar. Existing GUI tests are marked `gui` and use pytest-qt.

An offscreen PyQt6 geometry probe on the current branch (Qt-reported screen 800×800 logical pixels) confirmed the constraints are active rather than merely visual: `MainWindow` has a 900×500 minimum despite requesting an initial 700×450 size, while its file-open button collapses to zero height at 900×500 because the content size hint exceeds that height; `AnalysisWindow` has a 892×504 minimum (400px image minimum plus the 450px control scroll area and layout), and at 900×650 the bottom analysis action remains below the viewport; `TestingWindow` has a 591×958 minimum and a 677×1122 size hint, with no outer scroll area; `LabelingWindow` has a 2077px minimum width, largely from its one-row toolbar, and forces maximized state on show; and `LabelingSessionDialog` stays at least 460×420 after a 400×300 resize request. The analysis pane itself already scrolls vertically, but the outer window cannot fit an 800px-wide display. `ui/styles.py` and several window-local stylesheets also contain absolute pixel font sizes, paddings and minimum widths. These dimensions are environment-specific diagnostics, not universal pixel requirements. Existing GUI coverage exercises algorithm selection and the testing window, but has no compact-size geometry checks for the main window, labeling window or session dialog.

## Goals / Non-Goals

**Goals:**
- Provide consistent adaptive sizing across application-owned windows while keeping controls usable and text readable.
- Preserve access to all current actions at narrow and short client sizes by reflowing groups or making overflowing content scrollable.
- Exercise resize behavior through GUI regression tests at more than one window size.

**Non-Goals:**
- Change analysis, labeling, model, or testing behavior, algorithms, or data formats.
- Change operating-system-native file dialogs or add a new UI toolkit/dependency.
- Guarantee that every screen can display every control simultaneously at arbitrarily small dimensions; overflow must instead remain reachable.

## Decisions

### Use one adaptive sizing policy with a readable floor

Determine a scale from each top-level window's available client area relative to its chosen reference layout, clamp it to a defined readable range, and apply it consistently to typography, control dimensions, padding and layout spacing. Compute sizes in Qt logical coordinates; Qt already accounts for device pixel ratio, so multiplying logical dimensions by DPR again would double-scale the UI. Recompute from the original reference metrics on resize (never from already-scaled values) so enlarging a window restores the normal sizes. Use shared scale/metric helpers and shared scale-aware style rules; migrate local inline styles that currently hard-code font size, padding or minimum width so they cannot override the adaptive values. Reference layouts and scale bounds should be selected and documented from GUI measurements rather than duplicated as per-widget magic numbers.

**Alternative considered:** let Qt's platform font scaling alone resize the UI. This does not address fixed dimensions, hard-coded stylesheet padding, or single-row layouts and cannot meet the reachability requirements by itself.

### Reflow dense control groups; scroll overflowing content

Prefer responsive layouts that wrap or stack related button/control groups when width is constrained. Where complete content exceeds available height (notably testing sections, analysis controls, labeling controls, and session-dialog content), place the content in a scrollable region while keeping primary actions visible where practical. The labeling toolbar should be split into semantic groups and reflow or expose an explicit overflow path instead of imposing its entire one-row size as the minimum width. Preserve existing startup presentation (the labeling window currently opens maximized); ensure the user can restore and resize it. Preserve image canvases as the expanding area and keep their own scrolling/zoom behavior separate from control-panel scrolling; zooming the image must not force the restored top-level window to grow beyond the available display.

**Alternative considered:** enforce larger minimum window sizes. This prevents use on small displays rather than making the UI adapt, contrary to the agreed behavior.

### Centralize metrics and retain layout ownership

Use a shared UI sizing utility/policy rather than separately hard-coding scale calculations in each window. Keep application-wide colors/theme in the existing style module, and make dimension-bearing rules scale-aware per top-level window; remove or adapt local inline style declarations that would otherwise take precedence. Windows remain responsible for semantic grouping and responsive layout. Avoid applying geometrical scaling to the whole rendered window or image canvas, which would cause fuzzy text, distort image interaction coordinates, or bypass Qt layout calculations.

### Verify geometry and reachability, not pixel-perfect appearance

Use 1280×800 and 800×600 Qt logical client dimensions as normal/compact regression targets, plus a smaller dialog case. Add pytest-qt coverage for the main window, analysis, testing, labeling, and session dialog. Tests should check that after restoring a window, top-level windows do not insist on a size larger than the requested supported client area, primary controls are visible or reachable by scrolling/overflow, and interactive siblings do not geometrically overlap. Scrollable overflow counts as reachable only if the control is inside the scroll area's content and the scrollbar can bring it into the viewport. Also exercise a resize back to the normal size and image zoom in the labeling canvas. Keep assertions resilient to platform font/theme differences by checking scale relationships, bounds and reachability rather than pixel-perfect snapshots.

## Risks / Trade-offs

- [Scaling can make text too small or layouts too large] → Clamp scale to a readable range; reflow/scroll once the available area reaches the minimum.
- [Scroll areas can conflict with image zoom/painting or nested scrolling] → Limit scroll areas to control/content panes and explicitly test wheel behavior over image canvases.
- [Platform font metrics and OS scaling differ] → Use geometry/reachability assertions with Qt layout metrics and test common desktop sizes rather than brittle pixel-perfect snapshots.
- [A shared policy can trigger recursive resize/layout updates] → Update only when the computed scale changes and let Qt layouts recalculate widget geometry.
- [Per-widget stylesheets override the shared theme metrics] → Inventory and migrate dimension-bearing inline styles to shared adaptive rules, retaining only semantic color/role styling at widget level.
- [Nested image and control scrolling can consume each other's wheel events] → Keep independent viewports, prevent zoomed canvases from propagating minimum sizes to top-level windows, and cover both scroll paths in GUI tests.

## Migration Plan

No data migration is required. Update each application-owned window to use the shared adaptive policy and responsive layouts, then add GUI regression coverage. Rollback consists of reverting the UI layout/policy changes; no persisted user data or external interface is modified.
