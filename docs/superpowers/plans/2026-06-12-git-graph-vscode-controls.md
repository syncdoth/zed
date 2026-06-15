# Git Graph VS Code-style Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four VS Code "Git Graph" behaviors to Zed's git graph — double-click a branch chip to check it out, toggle tags, toggle remote branches, and search/multi-select which branches to show.

**Architecture:** A `GraphRefFilter` is folded into `LogSource::All` so it flows through the graph's `LogSource`-keyed cache and reload path; toggling ref types changes the git log args (pruning commits, VS Code-style). Ref-type toggles persist in `settings.json`; the manual branch selection persists per-repo via the existing workspace-DB `LogSource` serialization. UI adds a "Branches" multi-select dropdown and a ⚙ "View" menu to the graph's top bar.

**Tech Stack:** Rust, GPUI, Zed `git`/`git_ui`/`settings_content` crates.

**Build/test commands:**
- Fast type-check: `cargo check -p git` and `cargo check -p git_ui`
- Lint: `./script/clippy`
- Targeted tests: `cargo test -p git_ui --lib git_graph::` and `cargo test -p git`

---

## File Structure

- `crates/git/src/repository.rs` — `GraphRefFilter` struct, `LogSource::All(GraphRefFilter)`, manual `Default for LogSource`, reworked `LogSource::get_args`.
- `crates/settings_content/src/settings_content.rs` — `GitGraphSettingsContent` nested under `GitPanelSettingsContent`.
- `assets/settings/default.json` — defaults for the three toggles.
- `crates/git_ui/src/git_panel_settings.rs` — resolved `GitGraphSettings` on `GitPanelSettings`.
- `crates/git_ui/src/git_graph.rs` — call-site updates, persistence serialization, initial filter + settings observer, double-click chip checkout, ⚙ View menu, top-bar wiring, tests.
- `crates/git_ui/src/git_graph_branch_filter.rs` — new "Branches" multi-select popover component.

---

## Task 1: `GraphRefFilter` and `LogSource` rework (git crate)

**Files:**
- Modify: `crates/git/src/repository.rs` (`LogSource` enum ~738-764)
- Test: `crates/git/src/repository.rs` (inline `#[cfg(test)]`)

- [ ] **Step 1: Write the failing test**

Add near the bottom of `repository.rs` (inside or appended to an existing `#[cfg(test)] mod tests`; if none exists in this file, create one):

```rust
#[cfg(test)]
mod log_source_tests {
    use super::*;

    #[test]
    fn all_filter_args() {
        let mk = |local, remote, tags, refs: Option<Vec<&str>>| GraphRefFilter {
            local_branches: local,
            remote_branches: remote,
            tags,
            selected_refs: refs
                .map(|r| r.into_iter().map(SharedString::from).collect::<Arc<[_]>>()),
        };

        // Everything on (default behavior, preserves old args).
        assert_eq!(
            LogSource::All(mk(true, true, true, None)).get_args().unwrap(),
            vec!["--ignore-missing", "--branches", "--remotes", "--tags", "HEAD"]
        );
        // Tags off.
        assert_eq!(
            LogSource::All(mk(true, true, false, None)).get_args().unwrap(),
            vec!["--ignore-missing", "--branches", "--remotes", "HEAD"]
        );
        // Local only.
        assert_eq!(
            LogSource::All(mk(true, false, true, None)).get_args().unwrap(),
            vec!["--ignore-missing", "--branches", "--tags", "HEAD"]
        );
        // Explicit selection replaces branch/remote flags, tags still honored.
        assert_eq!(
            LogSource::All(mk(true, true, true, Some(vec!["refs/heads/main"])))
                .get_args()
                .unwrap(),
            vec!["--ignore-missing", "refs/heads/main", "--tags", "HEAD"]
        );
        // Empty selection normalizes to show-all (never blank).
        assert_eq!(
            LogSource::All(mk(true, true, false, Some(vec![])))
                .get_args()
                .unwrap(),
            vec!["--ignore-missing", "--branches", "--remotes", "HEAD"]
        );
        // Default LogSource is All with everything on.
        assert_eq!(
            LogSource::default().get_args().unwrap(),
            vec!["--ignore-missing", "--branches", "--remotes", "--tags", "HEAD"]
        );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p git log_source_tests::all_filter_args 2>&1 | tail -20`
Expected: FAIL to compile (`GraphRefFilter` not found, `LogSource::All` is a unit variant).

- [ ] **Step 3: Implement `GraphRefFilter` and rework `LogSource`**

Ensure `use std::sync::Arc;` is present (it is, widely used in this file). Replace the `LogSource` enum and its `get_args` (currently ~738-764):

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphRefFilter {
    pub local_branches: bool,
    pub remote_branches: bool,
    pub tags: bool,
    /// `None` = show all refs governed by the bools above.
    /// `Some(refs)` = show only these explicit refs (an empty slice normalizes
    /// back to the bool-driven behavior so the graph never goes blank).
    pub selected_refs: Option<Arc<[SharedString]>>,
}

impl Default for GraphRefFilter {
    fn default() -> Self {
        Self {
            local_branches: true,
            remote_branches: true,
            tags: true,
            selected_refs: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LogSource {
    All(GraphRefFilter),
    Branch(SharedString),
    Sha(Oid),
    Path(RepoPath),
}

impl Default for LogSource {
    fn default() -> Self {
        LogSource::All(GraphRefFilter::default())
    }
}

impl LogSource {
    fn get_args(&self) -> Result<Vec<&str>> {
        match self {
            LogSource::All(filter) => {
                let mut args = vec!["--ignore-missing"];
                let selected = filter
                    .selected_refs
                    .as_deref()
                    .filter(|refs| !refs.is_empty());
                if let Some(refs) = selected {
                    args.extend(refs.iter().map(|r| r.as_str()));
                } else {
                    if filter.local_branches {
                        args.push("--branches");
                    }
                    if filter.remote_branches {
                        args.push("--remotes");
                    }
                }
                if filter.tags {
                    args.push("--tags");
                }
                args.push("HEAD");
                Ok(args)
            }
            LogSource::Branch(branch) => Ok(vec![branch.as_str()]),
            LogSource::Sha(oid) => Ok(vec![
                str::from_utf8(oid.as_bytes()).context("Failed to build str from sha")?,
            ]),
            LogSource::Path(path) => Ok(vec!["--follow", "--", path.as_unix_str()]),
        }
    }
}
```

Note: the old enum had `#[derive(... Default)]` with `#[default] All`. Remove the `Default` derive from the enum (we provide it manually) and delete the `#[default]` attribute.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p git log_source_tests::all_filter_args 2>&1 | tail -20`
Expected: PASS. (The `git` crate may surface unrelated downstream breakage only when `git_ui` compiles — that's Task 2.)

- [ ] **Step 5: Commit**

```bash
git add crates/git/src/repository.rs
git commit -m "git: Add GraphRefFilter to LogSource::All"
```

---

## Task 2: Fix `LogSource::All` call sites and persistence in git_ui

This task only restores compilation of `git_ui` after Task 1 (no new behavior). The graph keeps showing everything because all sites pass a default filter.

**Files:**
- Modify: `crates/git_ui/src/git_graph.rs` (call sites ~1124, 1148, 1176; serialization ~4356-4380; any `matches!(.., LogSource::All)` / `== LogSource::All`)

- [ ] **Step 1: Build to enumerate breakage**

Run: `cargo check -p git_ui 2>&1 | grep -A3 "LogSource::All\|expected.*All\|this enum variant takes" | head -60`
Expected: a list of sites that construct `LogSource::All` as a unit value or match it as a unit.

- [ ] **Step 2: Update construction sites**

For each site that builds `LogSource::All` (around lines 1124, 1148, 1176, 4358, 4367), change to carry a filter. For the constructors that just want the default-everything graph, use:

```rust
LogSource::All(GraphRefFilter::default())
```

Add the import at the top of `git_graph.rs` (find the existing `use git::...` group):

```rust
use git::repository::{GraphRefFilter, LogSource};
```

(Adjust to match the existing import path actually used for `LogSource` in this file — if it is imported as `git::LogSource`, then `use git::GraphRefFilter;`.)

For the line `RepositoryEvent::StashEntriesChanged if self.log_source == LogSource::All =>` (~1652) and any other `== LogSource::All` / `matches!(.., LogSource::All)`, change to a variant match:

```rust
RepositoryEvent::StashEntriesChanged if matches!(self.log_source, LogSource::All(_)) =>
```

- [ ] **Step 3: Update persistence serialization (~4356-4380)**

`serialize_log_source_type` matches `LogSource::All =>` — change to `LogSource::All(_) =>`.

`serialize_log_source_value` currently returns `None` for `All`. Persist the selected refs per-repo by serializing them (toggles come from settings, not here):

```rust
pub fn serialize_log_source_value(log_source: &LogSource) -> Option<String> {
    match log_source {
        LogSource::All(filter) => filter
            .selected_refs
            .as_deref()
            .filter(|refs| !refs.is_empty())
            .map(|refs| {
                refs.iter()
                    .map(|r| r.to_string())
                    .collect::<Vec<_>>()
                    .join("\n")
            }),
        LogSource::Branch(branch) => Some(branch.to_string()),
        // ...existing arms unchanged...
    }
}
```

In `deserialize_log_source` (find where `LOG_SOURCE_ALL` is handled, ~4205+), reconstruct the filter. Read the toggle defaults from settings at call time if available; for this task, restore selection only and keep bools at default (the live observer in Task 4 reconciles settings):

```rust
LOG_SOURCE_ALL => {
    let selected_refs = value
        .filter(|v| !v.is_empty())
        .map(|v| {
            v.split('\n')
                .map(SharedString::from)
                .collect::<std::sync::Arc<[_]>>()
        });
    LogSource::All(GraphRefFilter {
        selected_refs,
        ..GraphRefFilter::default()
    })
}
```

(Use the actual local variable names present in `deserialize_log_source`; match the existing `LOG_SOURCE_*` constant handling.)

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p git_ui 2>&1 | tail -20`
Expected: compiles (warnings OK).

- [ ] **Step 5: Commit**

```bash
git add crates/git_ui/src/git_graph.rs
git commit -m "git_ui: Thread GraphRefFilter through LogSource call sites"
```

---

## Task 3: Settings — `GitGraphSettingsContent` + defaults + resolution

**Files:**
- Modify: `crates/settings_content/src/settings_content.rs` (`GitPanelSettingsContent` ~638)
- Modify: `assets/settings/default.json` (`git_panel` block ~943)
- Modify: `crates/git_ui/src/git_panel_settings.rs` (`GitPanelSettings` struct + `from_settings`)

- [ ] **Step 1: Add the content struct**

In `settings_content.rs`, add a field to `GitPanelSettingsContent` (after `commit_title_max_length`):

```rust
    /// Git graph display options.
    pub git_graph: Option<GitGraphSettingsContent>,
```

And define the struct nearby (mirror the derives on `GitPanelSettingsContent` — find them just above line 638; typically `#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema, PartialEq)]` plus any `#[serde(...)]`):

```rust
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct GitGraphSettingsContent {
    /// Show local branches in the git graph.
    ///
    /// Default: true
    pub show_local_branches: Option<bool>,
    /// Show remote branches in the git graph.
    ///
    /// Default: true
    pub show_remote_branches: Option<bool>,
    /// Show tags in the git graph.
    ///
    /// Default: true
    pub show_tags: Option<bool>,
}
```

- [ ] **Step 2: Add defaults to `default.json`**

In `assets/settings/default.json`, inside the `"git_panel": { ... }` object (~943-980), add:

```jsonc
    "git_graph": {
      "show_local_branches": true,
      "show_remote_branches": true,
      "show_tags": true
    },
```

- [ ] **Step 3: Resolve into runtime settings**

In `git_panel_settings.rs`, add a runtime struct and field. Near `ScrollbarSettings`:

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GitGraphSettings {
    pub show_local_branches: bool,
    pub show_remote_branches: bool,
    pub show_tags: bool,
}
```

Add to `GitPanelSettings`:

```rust
    pub git_graph: GitGraphSettings,
```

In `from_settings`, after the existing fields:

```rust
            git_graph: {
                let git_graph = git_panel.git_graph.clone().unwrap_or_default();
                GitGraphSettings {
                    show_local_branches: git_graph.show_local_branches.unwrap_or(true),
                    show_remote_branches: git_graph.show_remote_branches.unwrap_or(true),
                    show_tags: git_graph.show_tags.unwrap_or(true),
                }
            },
```

- [ ] **Step 4: Verify**

Run: `cargo check -p git_ui 2>&1 | tail -20`
Expected: compiles. Settings JSON schema regenerates automatically; if a settings test fails on schema, run `cargo test -p settings 2>&1 | tail -20` and update as that crate's tests instruct.

- [ ] **Step 5: Commit**

```bash
git add crates/settings_content/src/settings_content.rs assets/settings/default.json crates/git_ui/src/git_panel_settings.rs
git commit -m "git_ui: Add git_graph branch/tag visibility settings"
```

---

## Task 4: Resolve initial filter from settings + live settings observer

**Files:**
- Modify: `crates/git_ui/src/git_graph.rs` (`GitGraph::new` ~1443-1557; add helper + observer)

- [ ] **Step 1: Add a helper that builds a filter from settings, preserving selection**

Add a method on `GitGraph` (near other small helpers):

```rust
fn filter_from_settings(selected_refs: Option<Arc<[SharedString]>>, cx: &App) -> GraphRefFilter {
    let settings = GitPanelSettings::get_global(cx).git_graph;
    GraphRefFilter {
        local_branches: settings.show_local_branches,
        remote_branches: settings.show_remote_branches,
        tags: settings.show_tags,
        selected_refs,
    }
}
```

Ensure `use crate::git_panel_settings::{GitPanelSettings, GitGraphSettings};` (or the existing path) and `use std::sync::Arc;` are imported.

- [ ] **Step 2: Apply settings to the initial `All` filter in `new`**

In `GitGraph::new` (~1453 `let log_source = log_source.unwrap_or_default();`), reconcile an `All` log source with current settings while preserving any persisted selection:

```rust
let log_source = log_source.unwrap_or_default();
let log_source = match log_source {
    LogSource::All(filter) => {
        LogSource::All(Self::filter_from_settings(filter.selected_refs, cx))
    }
    other => other,
};
```

- [ ] **Step 3: Add a `SettingsStore` observer that reconciles the live filter**

In `new`, where other subscriptions are pushed (the constructor builds `_subscriptions`), add (mirror `git_panel.rs`'s `observe_global_in::<SettingsStore>`):

```rust
cx.observe_global_in::<SettingsStore>(window, |this, _window, cx| {
    let LogSource::All(filter) = &this.log_source else {
        return;
    };
    let next = Self::filter_from_settings(filter.selected_refs.clone(), cx);
    if &next != filter {
        this.set_log_source(LogSource::All(next), cx);
    }
})
.detach();
```

- [ ] **Step 4: Add `set_log_source` (single reload entry point)**

If no such method exists, add one that updates the field, persists, and reloads. Reuse the existing reload path the constructor/`graph_data` calls use (the code around ~1579-1668 reloads when `source == &self.log_source`):

```rust
fn set_log_source(&mut self, log_source: LogSource, cx: &mut Context<Self>) {
    if self.log_source == log_source {
        return;
    }
    self.log_source = log_source;
    self.selected_entry_idx = None;
    if let Some(item) = self.workspace_item_handle.as_ref() {
        // Trigger workspace serialization so per-repo selection persists.
        item.update(cx, |_, cx| cx.notify()).ok();
    }
    self.reload_graph(cx); // use the file's existing reload helper name
    cx.notify();
}
```

Inspect the file for the actual reload helper (search `get_graph_data`/`graph_data(` usage in the constructor and the `RepositoryEvent` handler) and call that; if reload is inline, extract it into `reload_graph`. Match the actual persistence trigger used elsewhere (`serialize`/workspace item). Do not invent APIs — wire to what exists.

- [ ] **Step 5: Verify + manual sanity**

Run: `cargo check -p git_ui 2>&1 | tail -20`
Expected: compiles.

- [ ] **Step 6: Commit**

```bash
git add crates/git_ui/src/git_graph.rs
git commit -m "git_ui: Resolve git graph filter from settings with live updates"
```

---

## Task 5: Double-click branch chip → checkout

**Files:**
- Modify: `crates/git_ui/src/git_graph.rs` (`render_ref_chip` ~1730; add `checkout_ref`; tests ~6759 region)

- [ ] **Step 1: Write the failing test**

Model it on `test_global_git_command_task_runs_from_ref_context_menu` (~6759). Add a test that calls a new `GitGraph::checkout_ref` for a branch ref and asserts the repo's branch changed. Place near that test:

```rust
#[gpui::test]
async fn test_double_click_branch_chip_checks_out(cx: &mut TestAppContext) {
    // Reuse the same fixture setup as test_global_git_command_task_runs_from_ref_context_menu:
    // a repo with branches "main" (HEAD) and "feature-x" on some commit.
    let (git_graph, repo, cx) = setup_git_graph_with_branches(cx).await; // see note below

    git_graph.update_in(cx, |git_graph, window, cx| {
        git_graph.checkout_ref("feature-x".into(), window, cx);
    });
    cx.run_until_parked();

    let head = repo.read_with(cx, |repo, _| repo.snapshot().branch.clone());
    assert_eq!(head.unwrap().name(), "feature-x");
}
```

If no shared `setup_git_graph_with_branches` helper exists, inline the fixture used by the existing ref-context-menu test (copy its repo/branch setup) rather than referencing a non-existent helper.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p git_ui git_graph::tests::test_double_click_branch_chip_checks_out 2>&1 | tail -20`
Expected: FAIL (`checkout_ref` not found).

- [ ] **Step 3: Implement `checkout_ref`**

Add to `GitGraph` (model error handling on `branch_picker.rs` ~1306-1351 which uses `repo.change_branch`):

```rust
fn checkout_ref(&mut self, ref_name: SharedString, _window: &mut Window, cx: &mut Context<Self>) {
    let Some(repo) = self.get_repository(cx) else {
        return;
    };
    // Strip refs/heads/ and refs/remotes/ so we pass the branch name git expects;
    // for remote refs we pass through as-is to match VS Code (git may detach).
    let name = ref_name
        .strip_prefix("refs/heads/")
        .map(SharedString::from)
        .unwrap_or_else(|| ref_name.clone());
    cx.spawn(async move |this, cx| {
        let task = repo.update(cx, |repo, _| repo.change_branch(name.to_string()))?;
        if let Err(error) = task.await {
            this.update(cx, |_this, cx| {
                // Surface to the user; reuse the workspace notification path used elsewhere.
                log::error!("git graph checkout failed: {error:#}");
                cx.emit(/* existing error event, or workspace.show_error */);
            })
            .ok();
        }
        anyhow::Ok(())
    })
    .detach_and_log_err(cx);
}
```

Match the actual `change_branch` return type (it returns `BoxFuture<Result<()>>` per `repository.rs:827`; `repo.update` may return the future directly — adjust the `await`/`?` shape to the real signature). For error surfacing, use whatever the file already does for user-facing errors (search `notify_err`, `show_error`, or `workspace`). Do not silently drop the error.

- [ ] **Step 4: Wire double-click on the chip**

In `render_ref_chip` (~1730), for chips that represent a branch (local `refs/heads/*` or remote `refs/remotes/*`, not tags), add an `on_click` that checks out on double-click and stops propagation:

```rust
.on_click(cx.listener({
    let ref_name = ref_name.clone();
    move |this, event: &ClickEvent, window, cx| {
        if event.click_count() >= 2 {
            cx.stop_propagation();
            this.checkout_ref(ref_name.clone(), window, cx);
        }
    }
}))
```

Only attach this for branch refs — leave tag chips without it. Determine branch-ness from the decoration/ref data already available where the chip is built (`ref_name_from_decoration` ~1694 and the ref kind there).

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p git_ui git_graph::tests::test_double_click_branch_chip_checks_out 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/git_ui/src/git_graph.rs
git commit -m "git_ui: Check out branch on double-clicking its graph chip"
```

---

## Task 6: ⚙ View menu — tag/remote/local toggles

**Files:**
- Modify: `crates/git_ui/src/git_graph.rs` (`render_search_bar` ~2631; add `render_view_menu` + toggle writer)

- [ ] **Step 1: Add a settings-writing helper**

```rust
fn toggle_graph_setting(
    &mut self,
    field: GraphSettingField,
    cx: &mut Context<Self>,
) {
    let fs = self.fs.clone(); // confirm GitGraph holds `fs`; if not, get via project/workspace
    settings::update_settings_file(fs, cx, move |settings, _| {
        let git_panel = settings.git_panel.get_or_insert_with(Default::default);
        let git_graph = git_panel.git_graph.get_or_insert_with(Default::default);
        match field {
            GraphSettingField::Local => {
                let v = git_graph.show_local_branches.unwrap_or(true);
                git_graph.show_local_branches = Some(!v);
            }
            GraphSettingField::Remote => {
                let v = git_graph.show_remote_branches.unwrap_or(true);
                git_graph.show_remote_branches = Some(!v);
            }
            GraphSettingField::Tags => {
                let v = git_graph.show_tags.unwrap_or(true);
                git_graph.show_tags = Some(!v);
            }
        }
    });
}
```

Add `enum GraphSettingField { Local, Remote, Tags }` (private). The `SettingsStore` observer from Task 4 will pick up the change and reload the graph. If `GitGraph` has no `fs` field, obtain `<dyn Fs>::global(cx)` like `solo_diff_view.rs:483` does.

- [ ] **Step 2: Add `render_view_menu` (gear button + ContextMenu)**

Use `PopoverMenu` + `ContextMenu` with `toggle_entry` (search the codebase for `toggle_entry` usage to match the exact API). Each entry reads current `GitPanelSettings::get_global(cx).git_graph` for its checked state and calls `toggle_graph_setting` on toggle. Build an `IconButton::new("git-graph-view-menu", IconName::Settings)` (or `Sliders`/`Eye` — pick an existing icon) as the trigger.

```rust
fn render_view_menu(&self, cx: &mut Context<Self>) -> impl IntoElement {
    let settings = GitPanelSettings::get_global(cx).git_graph;
    let this = cx.entity();
    PopoverMenu::new("git-graph-view-menu")
        .trigger(IconButton::new("git-graph-view-menu-trigger", IconName::Settings)
            .shape(ui::IconButtonShape::Square)
            .icon_size(IconSize::Small)
            .tooltip(Tooltip::text("View options")))
        .menu(move |window, cx| {
            let this = this.clone();
            Some(ContextMenu::build(window, cx, |menu, _window, _cx| {
                menu.toggleable_entry("Show local branches", settings.show_local_branches, IconPosition::Start, None, {
                    let this = this.clone();
                    move |_w, cx| this.update(cx, |this, cx| this.toggle_graph_setting(GraphSettingField::Local, cx))
                })
                .toggleable_entry("Show remote branches", settings.show_remote_branches, IconPosition::Start, None, {
                    let this = this.clone();
                    move |_w, cx| this.update(cx, |this, cx| this.toggle_graph_setting(GraphSettingField::Remote, cx))
                })
                .toggleable_entry("Show tags", settings.show_tags, IconPosition::Start, None, {
                    let this = this.clone();
                    move |_w, cx| this.update(cx, |this, cx| this.toggle_graph_setting(GraphSettingField::Tags, cx))
                })
            }))
        })
}
```

Adjust `toggleable_entry` to the real `ContextMenu` method signature (search `fn toggleable_entry` / `fn toggle_entry` in `crates/ui`). If the signature differs, adapt the closure arity accordingly.

- [ ] **Step 3: Mount the gear button in the top bar**

In `render_search_bar` (~2678 cluster of right-aligned controls), add `.child(self.render_view_menu(cx))` to the right-hand `h_flex`.

- [ ] **Step 4: Verify**

Run: `cargo check -p git_ui 2>&1 | tail -20`
Expected: compiles. Manually: toggling "Show tags" should rewrite `settings.json` and reload the graph without tag-only commits.

- [ ] **Step 5: Commit**

```bash
git add crates/git_ui/src/git_graph.rs
git commit -m "git_ui: Add git graph view menu for tag/remote/local toggles"
```

---

## Task 7: "Branches" multi-select dropdown (new component)

**Files:**
- Create: `crates/git_ui/src/git_graph_branch_filter.rs`
- Modify: `crates/git_ui/src/git_ui.rs` (add `mod git_graph_branch_filter;` — confirm crate root filename)
- Modify: `crates/git_ui/src/git_graph.rs` (top-bar trigger + apply selection)

- [ ] **Step 1: Create the popover component**

Create `crates/git_ui/src/git_graph_branch_filter.rs`. It owns a filter `Editor`, the fetched branch list, and the pending selection; it emits the chosen `selected_refs` back to `GitGraph`. Model list/fuzzy-match on `branch_picker.rs` (`render_match`, `BranchEntry`), but multi-select with checkboxes.

```rust
use std::sync::Arc;
use git::repository::Branch;
use gpui::{prelude::*, Context, Entity, FocusHandle, Window};
use ui::prelude::*;

pub struct BranchFilterSelection {
    /// `None` = show all.
    pub selected_refs: Option<Arc<[SharedString]>>,
}

pub struct GitGraphBranchFilter {
    branches: Vec<Branch>,
    selected: std::collections::HashSet<SharedString>, // ref_names; empty = show all
    query: Entity<editor::Editor>,
    focus_handle: FocusHandle,
}

impl GitGraphBranchFilter {
    pub fn new(
        branches: Vec<Branch>,
        selected_refs: Option<Arc<[SharedString]>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let selected = selected_refs
            .map(|refs| refs.iter().cloned().collect())
            .unwrap_or_default();
        let query = cx.new(|cx| editor::Editor::single_line(window, cx));
        Self { branches, selected, query, focus_handle: cx.focus_handle() }
    }

    pub fn selection(&self) -> BranchFilterSelection {
        let selected_refs = if self.selected.is_empty() {
            None
        } else {
            Some(self.selected.iter().cloned().collect::<Arc<[_]>>())
        };
        BranchFilterSelection { selected_refs }
    }

    fn toggle(&mut self, ref_name: SharedString, cx: &mut Context<Self>) {
        if !self.selected.remove(&ref_name) {
            self.selected.insert(ref_name);
        }
        cx.emit(BranchFilterEvent::Changed);
        cx.notify();
    }

    fn show_all(&mut self, cx: &mut Context<Self>) {
        self.selected.clear();
        cx.emit(BranchFilterEvent::Changed);
        cx.notify();
    }
}

pub enum BranchFilterEvent {
    Changed,
}
impl EventEmitter<BranchFilterEvent> for GitGraphBranchFilter {}

impl Render for GitGraphBranchFilter {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let query = self.query.read(cx).text(cx).to_lowercase();
        let show_all_selected = self.selected.is_empty();
        v_flex()
            .w_72()
            .child(self.query.clone())
            .child(
                CheckboxWithLabel::new("git-graph-show-all", Label::new("Show all"),
                    show_all_selected.into(),
                    cx.listener(|this, _, _, cx| this.show_all(cx)))
            )
            .children(self.branches.iter().filter(|b| {
                query.is_empty() || b.name().to_lowercase().contains(&query)
            }).map(|branch| {
                let ref_name = branch.ref_name.clone();
                let checked = self.selected.contains(&ref_name);
                CheckboxWithLabel::new(
                    SharedString::from(format!("git-graph-branch-{}", ref_name)),
                    Label::new(branch.name().to_string()),
                    checked.into(),
                    cx.listener({
                        let ref_name = ref_name.clone();
                        move |this, _, _, cx| this.toggle(ref_name.clone(), cx)
                    }),
                )
            }))
    }
}
```

Adapt `CheckboxWithLabel`/`Checkbox` to the real `crates/ui` API (search `CheckboxWithLabel::new`), and `editor::Editor::single_line` to its real signature (see how `branch_picker.rs` builds its query editor). Make the list scrollable with `.max_h(...).overflow_y_scroll()` or `uniform_list` if the branch count is large.

- [ ] **Step 2: Register the module**

In the crate root (`crates/git_ui/src/git_ui.rs`), add:

```rust
mod git_graph_branch_filter;
```

Run: `cargo check -p git_ui 2>&1 | tail -20` — fix API mismatches surfaced here.

- [ ] **Step 3: Add the "Branches" trigger to the top bar**

In `git_graph.rs`, add a `branch_filter: Option<Entity<GitGraphBranchFilter>>` field (or lazily create on open), and a `render_branch_dropdown` using `PopoverMenu` whose trigger is a `Button::new("git-graph-branches", label)` where `label` reflects state:

```rust
fn branch_dropdown_label(&self) -> SharedString {
    let LogSource::All(filter) = &self.log_source else { return "All".into() };
    match filter.selected_refs.as_deref().filter(|r| !r.is_empty()) {
        None => "All".into(),
        Some([one]) => Branch::short_name(one), // strip refs/heads|remotes prefix
        Some(many) => format!("{} branches", many.len()).into(),
    }
}
```

Fetch branches via the repository's `branches()` (already used by `branch_picker`); load them when the popover opens. On `BranchFilterEvent::Changed`, read `selection()` and call `set_log_source(LogSource::All(GraphRefFilter { selected_refs, ..current }))` (preserving current toggle bools).

- [ ] **Step 4: Mount in `render_search_bar`**

Add `.child(self.render_branch_dropdown(window, cx))` between the search field and the view menu, matching Layout B order: search → Branches → ⚙.

- [ ] **Step 5: Verify**

Run: `cargo check -p git_ui 2>&1 | tail -20` then `./script/clippy 2>&1 | tail -30`
Expected: compiles, lint clean. Manual: selecting two branches prunes the graph to those; "Show all" restores.

- [ ] **Step 6: Commit**

```bash
git add crates/git_ui/src/git_graph_branch_filter.rs crates/git_ui/src/git_ui.rs crates/git_ui/src/git_graph.rs
git commit -m "git_ui: Add searchable branch multi-select to git graph"
```

---

## Task 8: Persistence round-trip test + final verification

**Files:**
- Modify: `crates/git_ui/src/git_graph.rs` (tests)

- [ ] **Step 1: Write the persistence round-trip test**

```rust
#[test]
fn test_selected_refs_round_trip() {
    use git::repository::{GraphRefFilter, LogSource};
    let refs: std::sync::Arc<[SharedString]> =
        vec![SharedString::from("refs/heads/main"), SharedString::from("refs/heads/dev")].into();
    let source = LogSource::All(GraphRefFilter { selected_refs: Some(refs.clone()), ..Default::default() });

    let value = persistence::serialize_log_source_value(&source);
    let ty = persistence::serialize_log_source_type(&source);
    let restored = persistence::deserialize_log_source(&fake_state(ty, value)); // build via existing helper

    match restored {
        LogSource::All(filter) => assert_eq!(filter.selected_refs.as_deref().unwrap(), &*refs),
        other => panic!("expected All, got {other:?}"),
    }
}
```

Use the actual `deserialize_log_source` input shape (it takes a serialized state struct, ~4205). If constructing that struct in a test is awkward, instead unit-test the encode/decode of the selection string directly (split/join on `\n`).

- [ ] **Step 2: Run the test**

Run: `cargo test -p git_ui git_graph 2>&1 | tail -30`
Expected: PASS (all git_graph tests).

- [ ] **Step 3: Full crate test + lint**

Run: `cargo test -p git -p git_ui 2>&1 | tail -40` then `./script/clippy 2>&1 | tail -40`
Expected: PASS, no new clippy warnings in touched files.

- [ ] **Step 4: Commit**

```bash
git add crates/git_ui/src/git_graph.rs
git commit -m "git_ui: Test git graph branch-selection persistence round-trip"
```

---

## Self-Review notes

- **Spec coverage:** Feature 1 → Task 5; Feature 2 (tags) → Tasks 1/3/6; Feature 3 (remotes) → Tasks 1/3/6; Feature 4 (search/select) → Task 7. Prune semantics → Task 1 `get_args`. Settings persistence → Task 3; per-repo selection persistence → Tasks 2 & 8. Initial/live filter → Task 4.
- **Known adaptation points (verify against real APIs during implementation, do not invent):** exact `LogSource`/`GraphRefFilter` import paths; `change_branch` future shape; `ContextMenu::toggleable_entry` signature; `CheckboxWithLabel`/`Editor::single_line` signatures; the file's existing reload helper and user-facing error path; `deserialize_log_source` state struct. Each is called out inline in its task.
```
