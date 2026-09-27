## 1. Shared Adaptive Sizing

- [ ] 1.1 Add shared scale/metric helpers that calculate clamped sizes from stable reference metrics in Qt logical coordinates, recalculate on window resize, and restore reference sizes when the window grows; verify bounds, monotonic scaling, repeated resize stability, and no double-application of scale with unit tests.
- [ ] 1.2 Inventory and migrate absolute font sizes, paddings, layout margins/spacings, fixed/minimum widget sizes, and local inline styles across all five windows/dialogs to the shared adaptive policy; verify the adaptive stylesheet metrics are not overridden by local QSS and no target top-level window retains a minimum larger than its supported compact client area.

## 2. Responsive Application Windows

- [ ] 2.1 Adapt the main window's contradictory minimum/initial dimensions, margins, header, file-selector row, and primary action buttons; verify the window can use 800×600 and the open-file, analyze, labeling, and testing actions stay visible and operable.
- [ ] 2.2 Adapt the analysis window's fixed-width control pane, 400px image-label minimum, and control/image split; verify the window fits 800×600 and 640×480, each control remains reachable in the control scroll area, and the image can still expand.
- [ ] 2.3 Add a scrollable or reflowing structure to the testing window, whose current 591×958 minimum prevents fitting an 800×800 display; verify dataset selection, algorithm checkboxes, run/export actions, and both result tables are reachable at 800×600 without overlap.
- [ ] 2.4 Keep the labeling window's current maximized startup presentation but allow restoring/resizing despite the 2077px minimum width caused by its fixed file panel/one-row toolbar; adapt the Petri side panel and wrap or expose overflow for toolbar groups; verify a restored 800×600 window is possible and image navigation, painting, zoom, save, clear, and export remain reachable.
- [ ] 2.5 Make the labeling session dialog and its root-path/recent/action rows fit a compact area such as 400×300 or scroll without hiding create/open/cancel actions; verify long paths wrap or truncate accessibly and recent-session selection still works.

## 3. GUI Regression Verification

- [ ] 3.1 Add pytest-qt tests for main, analysis, testing, and labeling windows at 1280×800 and 800×600 Qt logical client sizes, and the session dialog at 400×300 and standard size; verify size requests are honored within platform chrome tolerance, primary controls are visible/reachable, no interactive siblings overlap, and scrollbars can reach overflow content.
- [ ] 3.2 Add focused regressions for restoring normal metrics after compact-to-large resize, labeling image zoom not growing the top-level window, and GUI-thread resizing during analysis; verify image coordinates/painting and analysis progress behavior remain intact.
- [ ] 3.3 Run the relevant GUI tests with `QT_QPA_PLATFORM=offscreen` and the broader test suite; verify all existing analysis, testing, session, and labeling behavior remains green.
