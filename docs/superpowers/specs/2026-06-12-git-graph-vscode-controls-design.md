# Git Graph — VS Code-style controls

**Date:** 2026-06-12
**Status:** Approved design, ready for implementation planning

## Goal

Bring four behaviors from the VS Code "Git Graph" extension to Zed's git graph
(`crates/git_ui/src/git_graph.rs`):

1. Double-clicking a branch ref checks out that branch (instead of opening the
   commit view).
2. Show/hide tags.
3. Show/hide remote branches (local-only mode).
4. Search and manually select which branches to show.

## Decisions (locked)

- **Filter semantics:** prune commits via git log args (VS Code-style), not
  chip-hiding. Toggling a ref type off removes commits only reachable through it.
- **Toolbar layout:** Layout B — a "Branches: All ▾" dropdown for the
  search/multi-select picker, plus a separate ⚙ "View" menu for the
  tag/remote toggles. Search box stays where it is.
- **Double-click target:** branch *chip* checks out; double-clicking a bare
  commit row keeps today's commit-view behavior.
- **Remote chip checkout:** pass the ref straight to checkout and surface
  whatever git does (matches VS Code; may land in detached HEAD on older git).
- **Persistence:**
  - Ref-type toggles (local/remote/tags) persist globally in `settings.json`.
  - The manually selected branch list persists **per-repo** via the existing
    workspace-DB `LogSource` serialization (it is repo-specific, so it does not
    belong in global settings).

## Architecture

### 1. Ref filter folded into `LogSource::All` (`crates/git/src/repository.rs`)

The graph caches `GraphData` and persists view state keyed by `LogSource`
(see `git_graph.rs` `graph.log_source == log_source` cache check at ~line 1223,
and `serialize_log_source_*`). The filter therefore must live inside `LogSource`
so it participates in the cache key and reloads when changed.

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphRefFilter {
    pub local_branches: bool,
    pub remote_branches: bool,
    pub tags: bool,
    /// None = "show all" governed by the bools above.
    /// Some(refs) = show only these explicit refs (still subject to `tags`).
    pub selected_refs: Option<Arc<[SharedString]>>,
}

impl Default for GraphRefFilter {
    fn default() -> Self {
        Self { local_branches: true, remote_branches: true, tags: true, selected_refs: None }
    }
}

pub enum LogSource {
    All(GraphRefFilter),
    Branch(SharedString),
    Sha(Oid),
    Path(RepoPath),
}
```

`LogSource` currently derives `Default` with `#[default] All`. Since `#[default]`
cannot annotate a struct variant, replace it with a manual
`impl Default for LogSource { fn default() -> Self { LogSource::All(GraphRefFilter::default()) } }`.
Keep the existing `Clone, Debug, PartialEq, Eq, Hash` derives (all field types
support them; `Arc<[SharedString]>` is `Eq + Hash`).

`get_args` (currently `git_graph` args `--ignore-missing --branches --remotes
--tags HEAD`) becomes, for the `All(filter)` arm:

```
--ignore-missing
if let Some(refs) = &filter.selected_refs {   // empty slice treated as None → show all
    push each ref.as_str()
} else {
    if filter.local_branches  { push "--branches" }
    if filter.remote_branches { push "--remotes" }
}
if filter.tags { push "--tags" }
HEAD
```

Returned `Vec<&str>` borrows from `self`, which is fine (the refs live in the
filter inside `self`). An empty `selected_refs` slice is normalized to the
bool-driven path so the graph never goes blank.

**Call-site updates:** the ~6 `LogSource::All` constructions in `git_graph.rs`
(approx lines 1124, 1148, 1176, plus `serialize_log_source_type`/`value` and
`deserialize_log_source`) become `LogSource::All(filter)` / pattern
`LogSource::All(_)`.

### 2. Settings — persistence for the ref-type toggles

Add a nested block to `GitPanelSettingsContent`
(`crates/settings_content/src/settings_content.rs`):

```rust
pub git_graph: Option<GitGraphSettingsContent>,

#[derive(... same derives as siblings ...)]
pub struct GitGraphSettingsContent {
    /// Show local branches in the git graph. Default: true
    pub show_local_branches: Option<bool>,
    /// Show remote branches in the git graph. Default: true
    pub show_remote_branches: Option<bool>,
    /// Show tags in the git graph. Default: true
    pub show_tags: Option<bool>,
}
```

- Defaults in `assets/settings/default.json` under `"git_panel"`:
  ```jsonc
  "git_graph": { "show_local_branches": true, "show_remote_branches": true, "show_tags": true }
  ```
- Resolve into a runtime struct in `crates/git_ui/src/git_panel_settings.rs`
  (add a `GitGraphSettings` field on `GitPanelSettings`, unwrapping the content
  the same way the existing fields do).

The ⚙ View-menu toggles call `update_settings_file` (pattern already used in
`git_panel.rs` for `tree_view`/`sort_by_path`), so they persist globally and
form the default filter for newly opened graphs.

### 3. Initial filter resolution & live updates (`git_graph.rs`)

When a `GitGraph` opens in `All` mode:
`filter = GraphRefFilter { from GitGraphSettings toggles, selected_refs: from per-repo persisted state }`.

- A `SettingsStore` observer (mirroring `git_panel.rs`'s
  `observe_global_in::<SettingsStore>`) updates the view's filter bools when
  settings change, swaps `self.log_source`, and triggers a reload.
- Changing branch selection updates `self.log_source` and persists the selected
  refs per-repo through the extended `serialize_log_source_value`.

### 4. Double-click chip → checkout (`git_graph.rs`)

Branch chips render in `render_ref_chip` (~line 1730); left-clicks currently
fall through to the row's `on_click` → `handle_entry_click`. Add an `on_click`
on the chip element:

- **click_count ≥ 2** on a branch chip (local `refs/heads/*` or remote
  `refs/remotes/*`): call `repo.change_branch(name)` (the API `branch_picker.rs`
  already uses at ~line 1330) with the chip's ref, and **stop propagation** so
  the commit view does not also open.
- **single click:** unchanged (do not stop propagation → row selects as today).
- **tag chips:** no checkout behavior.

Checkout runs in a `cx.spawn`; errors are surfaced to the UI (workspace toast /
`notify_err`) rather than swallowed, consistent with `branch_picker` and the
repo's no-silent-`let _ =` rule. The existing `handle_entry_click`
double-click → `open_commit_view` path remains for bare commit rows.

### 5. "Branches" dropdown — search + multi-select (feature 4)

New file `crates/git_ui/src/git_graph_branch_filter.rs` (distinct logical
component). An anchored popover triggered by the **"Branches: All ▾"** button:

- a filter `Editor` with fuzzy matching (reuse the matching approach in
  `branch_picker.rs`);
- a "Show all" entry → sets `selected_refs = None`;
- a scrollable checkbox list of local + remote branches sourced from
  `repo.branches()` (`BranchesScanResult`), where the list respects the
  show-local / show-remote toggles for what is offered;
- multi-select toggles update `selected_refs` on the view's filter.

The trigger button label reflects state: `"All"`, a single branch name, or
`"N branches"`.

### 6. ⚙ View menu (features 2 & 3)

A `ContextMenu` from a gear `IconButton` with three toggle entries — Show local
branches / Show remote branches / Show tags — each writing the corresponding
setting via `update_settings_file`.

### 7. Top-bar wiring (`render_search_bar`, ~line 2631)

Add the **Branches** dropdown button and the ⚙ button to the right of the
existing search controls (Layout B). No change to the search editor itself.

## Testing

In `git_graph.rs` tests (and `repository.rs` where appropriate):

- `GraphRefFilter` → `get_args` for each combination: all on; tags off; remotes
  off; local-only; explicit `selected_refs`; empty `selected_refs` normalizes to
  show-all.
- Double-click on a branch chip invokes `change_branch` with the right ref
  (mirror the existing `test_global_git_command_task_runs_from_ref_context_menu`
  harness); single-click does not check out.
- Toggling a View-menu setting updates the live filter / `log_source` and
  triggers a reload.
- Per-repo round-trip: serialize a `selected_refs` selection and deserialize it
  back into the filter.

## Out of scope

- Detached-HEAD checkout from bare commit rows (only chips check out).
- Tag checkout.
- Reordering/styling beyond adding the two top-bar controls.
- New branch operations (create/delete/merge) from the graph.

## Affected files

- `crates/git/src/repository.rs` — `GraphRefFilter`, `LogSource::All` rework,
  `get_args`, manual `Default`.
- `crates/settings_content/src/settings_content.rs` — `GitGraphSettingsContent`.
- `assets/settings/default.json` — defaults.
- `crates/git_ui/src/git_panel_settings.rs` — resolved `GitGraphSettings`.
- `crates/git_ui/src/git_graph.rs` — chip checkout, top-bar controls, View menu,
  filter resolution + observer, `LogSource` call sites, persistence
  serialization, tests.
- `crates/git_ui/src/git_graph_branch_filter.rs` — new branch picker popover.

## Release Notes

- Added VS Code-style git graph controls: double-click a branch to check it out,
  toggle tags and remote branches, and search/select which branches to show.
