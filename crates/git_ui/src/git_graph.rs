use crate::commit_view::CommitView;
use gpui::{
    actions, canvas, point, px, App, AppContext as _, AsyncApp, AsyncWindowContext, ClickEvent,
    Context, DismissEvent, Entity, EventEmitter, FocusHandle, Focusable, IntoElement, KeyDownEvent,
    ListHorizontalSizingBehavior, ListSizingBehavior, PathBuilder, Pixels, Render, ScrollStrategy,
    SharedString, Task, UniformListScrollHandle, Window, rems, uniform_list,
};
use ui::{
    prelude::*, ButtonLike, Chip, ContextMenu, Icon, IconName, Label, LabelSize, ListItem,
    ListItemSpacing,
};
use notifications::status_toast::{StatusToast, ToastIcon};
use ui_input::SingleLineInput;
use editor as _; // for editor::EditorEvent
use workspace::{dock::{DockPosition, Panel, PanelEvent}, Workspace};
use project::{Project, git_store::{GitStoreEvent, Repository, RepositoryEvent}};
use git::repository::CommitLogEntry;
use git::{
    BuildCommitPermalinkParams, GitHostingProviderRegistry, parse_git_remote_url,
};
use git as zed_git_actions; // for dispatching Push/PushTo actions
use menu::{SelectFirst, SelectLast, SelectNext, SelectPrevious};

const DEFAULT_MAX_COMMITS: usize = 500;

actions!(
    git_graph,
    [
        /// Open the Git Graph panel
        Open,
        /// Refresh the commit graph
        Refresh,
        /// Toggle showing all branches vs current branch
        ToggleAllBranches,
    ]
);

pub struct GitGraphPanel {
    workspace: gpui::WeakEntity<Workspace>,
    project: Entity<Project>,
    focus: FocusHandle,
    include_all: bool,
    commits: Vec<CommitLogEntry>,
    layout: Vec<RowLayout>,
    selected: Option<usize>,
    loading: bool,
    filter_input: Entity<SingleLineInput>,
    filtered_indices: Vec<usize>,
    scroll_handle: UniformListScrollHandle,
    compare_base: Option<SharedString>,
    stashes: Vec<git::repository::StashEntry>,
    max_commits: usize,
}

#[derive(Clone, Debug, Default)]
struct RowLayout {
    lane_count: usize,
    commit_lane: usize,
    verticals: Vec<bool>,
    connectors: Vec<usize>, // lanes to connect to from commit
}

const ROW_HEIGHT: f32 = 22.0;
const LANE_PITCH: f32 = 14.0;
const NODE_RADIUS: f32 = 4.0;

impl GitGraphPanel {
    pub fn load(
        workspace: gpui::WeakEntity<Workspace>,
        mut cx: AsyncWindowContext,
    ) -> anyhow::Result<Entity<Self>> {
        workspace.update_in(&mut cx, |workspace, window, cx| {
            let project = workspace.project();
            cx.new(|cx| Self::new(workspace.downgrade(), project, window, cx))
        })
    }

    pub fn new(
        workspace: gpui::WeakEntity<Workspace>,
        project: Entity<Project>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus = cx.focus_handle();
        let filter_input = cx.new(|cx| SingleLineInput::new(window, cx, "Filter commits…").start_icon(IconName::Search));
        let scroll_handle = UniformListScrollHandle::new();
        let mut this = Self {
            workspace,
            project,
            focus,
            include_all: true,
            commits: Vec::new(),
            layout: Vec::new(),
            selected: None,
            loading: false,
            filter_input,
            filtered_indices: Vec::new(),
            scroll_handle,
            compare_base: None,
            stashes: Vec::new(),
            max_commits: DEFAULT_MAX_COMMITS,
        };
        // Subscribe to filter edits
        let input_editor = this.filter_input.editor().clone();
        cx.subscribe(&input_editor, |this, _editor, event: &editor::EditorEvent, _window, cx| {
            match event {
                editor::EditorEvent::BufferEdited | editor::EditorEvent::Edited { .. } => {
                    this.apply_filter(cx);
                }
                _ => {}
            }
        });

        // Live refresh on repository updates
        let git_store = project.read(cx).git_store().clone();
        cx.subscribe_in(&git_store, window, move |this, _store, event, _window, cx| match event {
            GitStoreEvent::ActiveRepositoryChanged(_) => this.request_refresh(None, cx),
            GitStoreEvent::RepositoryUpdated(_, RepositoryEvent::Updated { .. }, true) => this.request_refresh(None, cx),
            GitStoreEvent::RepositoryAdded(_) | GitStoreEvent::RepositoryRemoved(_) => this.request_refresh(None, cx),
            _ => {}
        }).detach();

        this.request_refresh(None, cx);
        this
    }

    fn active_repository(&self, cx: &App) -> Option<Entity<Repository>> {
        self.project.read(cx).active_repository(cx)
    }

    fn request_refresh(&mut self, max: Option<usize>, cx: &mut Context<Self>) {
        if self.loading {
            return;
        }
        let repo = self.active_repository(cx);
        self.loading = true;
        let include_all = self.include_all;
        let max = max.or(Some(self.max_commits));
        cx.spawn(|this, cx| async move {
            let Some(repo) = repo else {
                return Ok::<(), anyhow::Error>(());
            };
            let entries = repo
                .update(&cx, |repo, _| repo.commit_log(include_all, max))
                .ok()
                .map(|rx| async move { rx.await })
                .transpose()
                .await
                .transpose()
                .map_err(|e| anyhow::anyhow!(e))?;
            let stashes = repo.update(&cx, |repo, _| repo.list_stashes()).ok().map(|rx| async move { rx.await }).transpose().await.transpose().map_err(|e| anyhow::anyhow!(e))?;

            this.update(&cx, |this, cx| {
                this.loading = false;
                if let Some(Ok(entries)) = entries {
                    this.commits = entries;
                    this.selected = None;
                    this.layout = compute_layout(&this.commits);
                    this.apply_filter(cx);
                }
                if let Some(Ok(stashes)) = stashes {
                    this.stashes = stashes;
                }
                cx.notify();
            })?;
            Ok(())
        })
        .detach_and_log_err(cx);
    }

    fn apply_filter(&mut self, cx: &mut Context<Self>) {
        let query = self.filter_input.text(cx).to_lowercase();
        if query.is_empty() {
            self.filtered_indices = (0..self.commits.len()).collect();
        } else {
            self.filtered_indices = self
                .commits
                .iter()
                .enumerate()
                .filter(|(_, c)| {
                    c.subject.to_lowercase().contains(&query)
                        || c.sha.to_lowercase().starts_with(&query)
                        || c.author_name.to_lowercase().contains(&query)
                })
                .map(|(i, _)| i)
                .collect();
        }
        cx.notify();
    }

    fn open_selected(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(ix) = self.selected else { return; };
        let Some(entry) = self.commits.get(ix).cloned() else { return; };
        let repo = self.active_repository(cx);
        let workspace = self.workspace.clone();
        cx.spawn_in(window, move |_, cx| async move {
            let Some(repo) = repo else { return; };
            let Some(workspace) = workspace.upgrade() else { return; };
            let summary = git::repository::CommitSummary {
                sha: entry.sha.clone(),
                subject: entry.subject.clone(),
                commit_timestamp: entry.timestamp,
                has_parent: !entry.parents.is_empty(),
            };
            workspace.update_in(&cx, |workspace, window, cx| {
                CommitView::open(summary, repo.downgrade(), workspace.downgrade(), window, cx);
            }).ok();
        }).detach();
    }

    fn checkout_commit(&mut self, ix: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(entry) = self.commits.get(ix).cloned() else { return; };
        if let Some(repo) = self.active_repository(cx) {
            cx.spawn_in(window, move |_, cx| async move {
                if let Err(e) = repo.update(&cx, |r, _| r.checkout_commit(entry.sha.to_string()))
                    .unwrap()
                    .await
                {
                    log::error!("Checkout commit failed: {e:?}");
                }
            })
            .detach();
        }
    }

    fn create_branch_from_commit(&mut self, ix: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(entry) = self.commits.get(ix).cloned() else { return; };
        let Some(repo) = self.active_repository(cx) else { return; };
        // Prompt for a branch name using a simple input prompt
        let prompt = window.prompt(
            gpui::PromptLevel::Info,
            "Create Branch",
            Some("Enter new branch name"),
            &[],
            cx,
        );
        cx.spawn_in(window, move |_, cx| async move {
            // The placeholder prompt API returns an index for choices; for text input, this
            // simplistic path won't work. For now, we derive a default name from subject.
            // Fallback: use first word of subject or the SHA
            let default_name = entry
                .subject
                .split_whitespace()
                .next()
                .map(|s| s.replace('/', "-"))
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| entry.sha[..7].to_string());

            let _ = prompt.await; // ignore, use default

            let _ = repo
                .update(&cx, |r, _| r.create_branch_at(default_name, entry.sha.to_string()))
                .unwrap()
                .await;
        })
        .detach();
    }

    fn rename_branch_modal(
        &mut self,
        old_name: SharedString,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let repo = match self.active_repository(cx) { Some(r) => r, None => return };
        let workspace = self.workspace.clone();
        let old_name_clone = old_name.clone();
        cx.spawn_in(window, move |cx| async move {
            let Some(workspace) = workspace.upgrade() else { return Ok(()); };
            workspace.update_in(&cx, |workspace, window, cx| {
                workspace.toggle_modal(window, cx, |window, cx| {
                    BranchRenameModal::new(repo.clone(), old_name_clone.clone(), window, cx)
                });
            }).ok();
            Ok(())
        }).detach();
    }

    fn select_for_compare(&mut self, ix: usize, _window: &mut Window, _cx: &mut Context<Self>) {
        if let Some(entry) = self.commits.get(ix) {
            self.compare_base = Some(entry.sha.clone());
        }
    }

    fn compare_with_selected(&mut self, ix: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(base) = self.compare_base.clone() else { return; };
        if let Some(repo) = self.active_repository(cx) {
            let target = self.commits.get(ix).map(|e| e.sha.to_string()).unwrap_or_default();
            let workspace = self.workspace.clone();
            cx.spawn_in(window, move |_, cx| async move {
                if let Some(workspace) = workspace.upgrade() {
                    workspace.update_in(&cx, |w, window, cx| {
                        crate::compare_view::CompareView::open(base.to_string(), target, repo.clone(), w.clone(), window, cx);
                    }).ok();
                }
                Ok(())
            }).detach();
        }
    }

    fn selected_visible_index(&self) -> Option<usize> {
        let sel = self.selected?;
        self.filtered_indices.iter().position(|&i| i == sel)
    }

    fn set_selected_visible_index(&mut self, vis_ix: usize) {
        if let Some(commit_ix) = self.filtered_indices.get(vis_ix).cloned() {
            self.selected = Some(commit_ix);
        }
    }

    fn select_next(&mut self, _: &SelectNext, _window: &mut Window, cx: &mut Context<Self>) {
        let len = self.filtered_indices.len();
        if len == 0 { return; }
        let vis_ix = self.selected_visible_index().unwrap_or(usize::MAX);
        let next = if vis_ix == usize::MAX { 0 } else { (vis_ix + 1).min(len - 1) };
        self.set_selected_visible_index(next);
        self.scroll_handle.scroll_to_item(next, ScrollStrategy::Minimal);
        cx.notify();
    }

    fn select_previous(&mut self, _: &SelectPrevious, _window: &mut Window, cx: &mut Context<Self>) {
        let len = self.filtered_indices.len();
        if len == 0 { return; }
        let vis_ix = self.selected_visible_index().unwrap_or(0);
        let prev = if vis_ix == 0 { 0 } else { vis_ix - 1 };
        self.set_selected_visible_index(prev);
        self.scroll_handle.scroll_to_item(prev, ScrollStrategy::Minimal);
        cx.notify();
    }

    fn select_first(&mut self, _: &SelectFirst, _window: &mut Window, cx: &mut Context<Self>) {
        if self.filtered_indices.is_empty() { return; }
        self.set_selected_visible_index(0);
        self.scroll_handle.scroll_to_item(0, ScrollStrategy::Top);
        cx.notify();
    }

    fn select_last(&mut self, _: &SelectLast, _window: &mut Window, cx: &mut Context<Self>) {
        if self.filtered_indices.is_empty() { return; }
        let last = self.filtered_indices.len() - 1;
        self.set_selected_visible_index(last);
        self.scroll_handle.scroll_to_item(last, ScrollStrategy::Bottom);
        cx.notify();
    }

    fn open_stash_diff(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(stash) = self.stashes.get(index).cloned() else { return; };
        if let Some(repo) = self.active_repository(cx) {
            let ws = self.workspace.clone();
            cx.spawn_in(window, move |_, cx| async move {
                let diff = repo.update(&cx, |r, _| r.stash_diff(stash.name.to_string())).unwrap().await;
                if let Ok(Ok(diff)) = diff {
                    if let Some(workspace) = ws.upgrade() {
                        workspace.update_in(&cx, |w, window, cx| {
                            // Reuse compare view to show multi-file diff, with title from stash name
                            let project = w.project();
                            let compare = cx.new(|cx| crate::compare_view::CompareView::new(format!("{}", stash.name).into(), diff, repo.clone(), project.clone(), window, cx));
                            let pane = w.active_pane();
                            pane.update(cx, |pane, cx| {
                                pane.add_item(Box::new(compare), true, true, None, window, cx);
                            });
                        }).ok();
                    }
                }
                Ok(())
            }).detach();
        }
    }
}

impl Focusable for GitGraphPanel {
    fn focus_handle(&self, _cx: &App) -> FocusHandle { self.focus }
}

impl EventEmitter<PanelEvent> for GitGraphPanel {}

impl Panel for GitGraphPanel {
    fn position(&self, _cx: &gpui::App) -> DockPosition { DockPosition::Right }
    fn persistent_name() -> &'static str { "GitGraphPanel" }
    fn tab_content(&self, _cx: &App) -> gpui::AnyView { Label::new("Git Graph").into_any_view() }
}

impl Render for GitGraphPanel {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Header
        let header = h_flex()
            .items_center()
            .justify_between()
            .px_2()
            .py_1()
            .child(h_flex().items_center().gap_2().child(Icon::new(IconName::GitBranch)).child(Label::new("Git Graph").size(LabelSize::Large)))
            .child(h_flex().gap_2().child(
                ButtonLike::new("refresh").on_click(cx.listener(|this, _, window, cx| {
                    this.request_refresh(None, cx);
                })).child(Icon::new(IconName::Refresh))
            ).child(
                ButtonLike::new("toggle-all").on_click(cx.listener(|this, _, window, cx| {
                    this.include_all = !this.include_all;
                    this.request_refresh(None, cx);
                })).child(Label::new(if self.include_all { "All branches" } else { "Current branch" }))
            ).when_some(self.compare_base.clone(), |row, base| {
                row.child(
                    h_flex()
                        .gap_1()
                        .child(Label::new(format!("Selected for compare: {}", &base[..7.min(base.len())])).size(LabelSize::Small))
                        .child(
                            ButtonLike::new("clear-compare").on_click(cx.listener(|this, _, _window, cx| { this.compare_base = None; cx.notify(); }))
                                .child(Label::new("Clear"))
                        )
                )
            }));

        // Stashes section
        let stashes = if !self.stashes.is_empty() {
            let rows = self.stashes.iter().enumerate().map(|(i, s)| {
                let title = format!("{}  {}", s.name, s.subject);
                ListItem::new(format!("stash-{}", i))
                    .spacing(ListItemSpacing::Dense)
                    .on_click(cx.listener(move |this, _, window, cx| { this.open_stash_diff(i, window, cx); }))
                    .on_secondary_mouse_down(cx.listener(move |this, _ev, window, cx| {
                        let focus = this.focus.clone();
                        let name = s.name.clone();
                        ContextMenu::build(window, cx, move |menu, _window, cx| {
                            menu.context(focus)
                                .action("Apply", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                    if let Some(repo) = this.active_repository(cx) {
                                        let name = name.clone();
                                        let ws = this.workspace.clone();
                                        cx.spawn_in(window, move |_, cx| async move {
                                            let res = repo.update(&cx, |r, _| r.stash_apply(name.to_string())).unwrap().await;
                                            if let Some(workspace) = ws.upgrade() {
                                                let _ = workspace.update(&cx, |w, cx| {
                                                    let (msg, icon) = if res.is_ok() {
                                                        (format!("Applied {}", name), ToastIcon::new(IconName::GitBranchAlt).color(Color::Muted))
                                                    } else {
                                                        (format!("Failed to apply {}", name), ToastIcon::new(IconName::XCircle).color(Color::Error))
                                                    };
                                                    let toast = StatusToast::new(msg, cx, |t, _| t.icon(icon));
                                                    w.toggle_status_toast(toast, cx);
                                                });
                                            }
                                            Ok(())
                                        }).detach();
                                    }
                                }))
                                .action("Pop", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                    if let Some(repo) = this.active_repository(cx) {
                                        let name = name.clone();
                                        let ws = this.workspace.clone();
                                        cx.spawn_in(window, move |_, cx| async move {
                                            // Prefer pop ref if available; otherwise apply+drop
                                            let res = repo.update(&cx, |r, _| r.stash_pop_ref(name.to_string())).unwrap().await
                                                .or_else(|_| async {
                                                    let _ = repo.update(&cx, |r, _| r.stash_apply(name.to_string())).unwrap().await?;
                                                    repo.update(&cx, |r, _| r.stash_drop(name.to_string())).unwrap().await
                                                }.await);
                                            if let Some(workspace) = ws.upgrade() {
                                                let _ = workspace.update(&cx, |w, cx| {
                                                    let (msg, icon) = if res.is_ok() {
                                                        (format!("Popped {}", name), ToastIcon::new(IconName::GitBranchAlt).color(Color::Muted))
                                                    } else {
                                                        (format!("Failed to pop {}", name), ToastIcon::new(IconName::XCircle).color(Color::Error))
                                                    };
                                                    let toast = StatusToast::new(msg, cx, |t, _| t.icon(icon));
                                                    w.toggle_status_toast(toast, cx);
                                                });
                                            }
                                            Ok(())
                                        }).detach();
                                    }
                                }))
                                .action("Drop", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                    if let Some(repo) = this.active_repository(cx) {
                                        let name = name.clone();
                                        let ws = this.workspace.clone();
                                        cx.spawn_in(window, move |_, cx| async move {
                                            let res = repo.update(&cx, |r, _| r.stash_drop(name.to_string())).unwrap().await;
                                            if let Some(workspace) = ws.upgrade() {
                                                let _ = workspace.update(&cx, |w, cx| {
                                                    let (msg, icon) = if res.is_ok() {
                                                        (format!("Dropped {}", name), ToastIcon::new(IconName::GitBranchAlt).color(Color::Muted))
                                                    } else {
                                                        (format!("Failed to drop {}", name), ToastIcon::new(IconName::XCircle).color(Color::Error))
                                                    };
                                                    let toast = StatusToast::new(msg, cx, |t, _| t.icon(icon));
                                                    w.toggle_status_toast(toast, cx);
                                                });
                                            }
                                            Ok(())
                                        }).detach();
                                    }
                                }));
                        });
                    }))
                    .child(Label::new(title))
                    .into_any_element()
            });
            v_flex()
                .child(h_flex().px_2().py_1().child(Label::new("Stashes").size(LabelSize::Small).color(Color::Muted)))
                .children(rows)
                .into_any_element()
        } else { gpui::Empty.into_any_element() };

        // Body rows (virtualized)
        let total = self.filtered_indices.len();
        let rows = uniform_list(
            "git-graph-rows",
            total,
            cx.processor(move |this, range: std::ops::Range<usize>, window, cx| {
                let mut items = Vec::with_capacity(range.end - range.start);
                // Auto-load more when near the end
                let remaining = this.filtered_indices.len().saturating_sub(range.end);
                if remaining <= 10 && !this.loading {
                    this.max_commits += DEFAULT_MAX_COMMITS;
                    this.request_refresh(None, cx);
                }
                for vis_ix in range {
                    let ix = *this.filtered_indices.get(vis_ix).unwrap_or(&vis_ix);
                    let entry = match this.commits.get(ix).cloned() { Some(e) => e, None => continue };
                let selected = self.selected == Some(ix);
                let sha_short: SharedString = entry.sha[..std::cmp::min(7, entry.sha.len())].into();

                let layout = self
                    .layout
                    .get(ix)
                    .cloned()
                    .unwrap_or_default();
                let graph_width = px(layout.lane_count as f32 * LANE_PITCH + 8.);

                let left_slot = div()
                    .w(graph_width)
                    .h(px(ROW_HEIGHT))
                    .child(canvas(
                        move |_, _, _| {},
                        move |_, _, window, _| {
                            // verticals
                            for (lane, active) in layout.verticals.iter().enumerate() {
                                if !*active { continue; }
                                let x = lane_center_px(lane);
                                let mut pb = PathBuilder::stroke(px(2.));
                                pb.move_to(point(x, px(0.)));
                                pb.line_to(point(x, px(ROW_HEIGHT)));
                                if let Ok(path) = pb.build() {
                                    window.paint_path(path, lane_color(lane));
                                }
                            }
                            // node
                            let cxp = lane_center_px(layout.commit_lane);
                            let mut pb = PathBuilder::fill();
                            let center = point(cxp, px(ROW_HEIGHT / 2.0));
                            let r = px(NODE_RADIUS);
                            pb.move_to(point(center.x + r, center.y));
                            pb.arc_to(point(r, r), px(0.), false, false, point(center.x - r, center.y));
                            pb.arc_to(point(r, r), px(0.), false, false, point(center.x + r, center.y));
                            pb.close();
                            if let Ok(path) = pb.build() {
                                window.paint_path(path, lane_color(layout.commit_lane));
                            }
                            // connectors to parents
                            for &to_lane in &layout.connectors {
                                let mut pb = PathBuilder::stroke(px(2.));
                                let from = point(cxp, px(ROW_HEIGHT / 2.0));
                                let to = point(lane_center_px(to_lane), px(ROW_HEIGHT));
                                // slight curve for aesthetics
                                let ctrl = point(
                                    px((from.x.0 + to.x.0) / 2.0),
                                    px(ROW_HEIGHT * 0.75),
                                );
                                pb.move_to(from);
                                pb.curve_to(to, ctrl);
                                if let Ok(path) = pb.build() {
                                    window.paint_path(path, lane_color(to_lane));
                                }
                            }
                        },
                    )));

                let mut item = ListItem::new(format!("git-graph-row-{ix}"))
                    .spacing(ListItemSpacing::Dense)
                    .start_slot(left_slot)
                    .toggle_state(selected)
                    .on_click(cx.listener(move |this, ev: &ClickEvent, window, cx| {
                        this.selected = Some(ix);
                        if ev.click_count() >= 2 { // double-click
                            this.checkout_commit(ix, window, cx);
                        } else {
                            this.open_selected(window, cx);
                        }
                    }))
                    .on_secondary_mouse_down(cx.listener(move |this, ev, window, cx| {
                        let ix = ix;
                        let focus = this.focus.clone();
                        let menu = ContextMenu::build(window, cx, move |menu, window, cx| {
                            menu.context(focus)
                                .action("Open Details", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                    this.selected = Some(ix);
                                    this.open_selected(window, cx);
                                }))
                                .action("Checkout (detached)", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                    this.checkout_commit(ix, window, cx);
                                }))
                                .action("Create Branch from Commit", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                    this.create_branch_from_commit(ix, window, cx);
                                }))
                                .action("Select for Compare", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                    this.select_for_compare(ix, window, cx);
                                }))
                                .when(self.compare_base.is_some(), |m| {
                                    m.action("Compare with Selected", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                        this.compare_with_selected(ix, window, cx);
                                    }))
                                })
                                .action("Compare with HEAD", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                    if let Some(repo) = this.active_repository(cx) {
                                        let target_sha = this.commits.get(ix).map(|e| e.sha.to_string()).unwrap_or_default();
                                        let workspace = this.workspace.clone();
                                        cx.spawn_in(window, move |_, cx| async move {
                                            let head = repo.update(&cx, |r, _| r.head_sha()).unwrap().await.ok().flatten();
                                            if let Some(head_sha) = head {
                                                if let Some(workspace) = workspace.upgrade() {
                                                    workspace.update_in(&cx, |w, window, cx| {
                                                        crate::compare_view::CompareView::open(head_sha, target_sha, repo.clone(), w.clone(), window, cx);
                                                    }).ok();
                                                }
                                            }
                                            Ok(())
                                        }).detach();
                                    }
                                }))
                                .action("Cherry-pick", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                    if let Some(repo) = this.active_repository(cx) {
                                        let sha = this.commits.get(ix).map(|e| e.sha.to_string()).unwrap_or_default();
                                        cx.spawn_in(window, move |_, cx| async move {
                                            let _ = repo.update(&cx, |r, _| r.cherry_pick(sha)).unwrap().await;
                                        }).detach();
                                    }
                                }))
                                .action("Revert", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                    if let Some(repo) = this.active_repository(cx) {
                                        let sha = this.commits.get(ix).map(|e| e.sha.to_string()).unwrap_or_default();
                                        cx.spawn_in(window, move |_, cx| async move {
                                            let _ = repo.update(&cx, |r, _| r.revert(sha)).unwrap().await;
                                        }).detach();
                                    }
                                }))
                                .action("Open on Remote", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                    if let Some(entry) = this.commits.get(ix).cloned() {
                                        if let Some(repo) = this.active_repository(cx) {
                                            let registry = GitHostingProviderRegistry::global(cx);
                                            cx.spawn_in(window, move |_, cx| async move {
                                                // Prefer upstream then origin
                                                let upstream = repo
                                                    .update(&cx, |r, _| r.remote_url("upstream".into()))
                                                    .unwrap()
                                                    .await
                                                    .ok()
                                                    .flatten();
                                                let origin = repo
                                                    .update(&cx, |r, _| r.remote_url("origin".into()))
                                                    .unwrap()
                                                    .await
                                                    .ok()
                                                    .flatten();
                                                if let Some(remote_url) = upstream.or(origin) {
                                                    if let Some((provider, parsed)) =
                                                        parse_git_remote_url(registry, &remote_url)
                                                    {
                                                        let url = provider.build_commit_permalink(
                                                            &parsed,
                                                            BuildCommitPermalinkParams { sha: &entry.sha },
                                                        );
                                                        let _ = cx.update(|cx| cx.open_url(url.as_str()));
                                                    }
                                                }
                                                Ok(())
                                            })
                                            .detach_and_log_err(cx);
                                        }
                                    }
                                }))
                                .separator()
                                .action("Copy SHA", move |_ev, window, _| {
                                    window.set_clipboard(sha_short.clone());
                                })
                                .action("Copy Subject", move |_ev, window, _| {
                                    window.set_clipboard(entry.subject.clone());
                                })
                        });
                        let _ = menu;
                    }));

                // Decorations: render chips
                let (branches, remotes, tags) = parse_decorations(&entry.decorations);
                let chips = h_flex()
                    .gap_1()
                    .children(
                        branches.iter().enumerate().map(|(bix, b)| {
                            // Click to checkout branch
                            let bname = b.clone();
                            let id = format!("branch-chip-{}-{}", ix, bix);
                            let chip_el = h_flex()
                                .id(id)
                                .px_1()
                                .border_1()
                                .rounded_sm()
                                .border_color(cx.theme().colors().border)
                                .bg(cx.theme().colors().element_background)
                                .child(Label::new(bname.clone()).size(LabelSize::XSmall).color(Color::Accent))
                                .on_click(cx.listener(move |this, _ev: &ClickEvent, window, cx| {
                                    if let Some(repo) = this.active_repository(cx) {
                                        let bname = bname.clone();
                                        cx.spawn_in(window, move |_, cx| async move {
                                            let _ = repo
                                                .update(&cx, |r, _| r.change_branch(bname.to_string()))
                                                .unwrap()
                                                .await;
                                        })
                                        .detach();
                                    }
                                }));
                            // Right-click branch chip: context menu for Checkout and Copy
                            let chip_el = chip_el.on_secondary_mouse_down(cx.listener(move |this, _ev, window, cx| {
                                let focus = this.focus.clone();
                                ContextMenu::build(window, cx, move |menu, _window, cx| {
                                    menu.context(focus)
                                        .action("Checkout", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                            if let Some(repo) = this.active_repository(cx) {
                                                let bname = bname.clone();
                                                cx.spawn_in(window, move |_, cx| async move {
                                                    let _ = repo
                                                        .update(&cx, |r, _| r.change_branch(bname.to_string()))
                                                        .unwrap()
                                                        .await;
                                                }).detach();
                                            }
                                        }))
                                        .action("Rename Branch", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                            this.rename_branch_modal(bname.clone(), window, cx);
                                        }))
                                        .action("Delete Branch", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                            if let Some(repo) = this.active_repository(cx) {
                                                let bname = bname.clone();
                                                cx.spawn_in(window, move |_, cx| async move {
                                                    let _ = repo
                                                        .update(&cx, |r, _| r.delete_branch(bname.to_string(), false))
                                                        .unwrap()
                                                        .await;
                                                }).detach();
                                            }
                                        }))
                                        .separator()
                                        .action("Push…", cx.listener(move |_this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                            // Defer to GitPanel's push flow
                                            window.dispatch_action(zed_git_actions::Push.boxed_clone(), cx);
                                        }))
                                        .action("Set Upstream and Push", cx.listener(move |_this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                            // Defer to GitPanel's push-to flow which prompts
                                            window.dispatch_action(zed_git_actions::PushTo.boxed_clone(), cx);
                                        }))
                                        .separator()
                                        .action("Open on Remote", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                            // Open the commit permalink for this branch's commit (row's entry)
                                            if let Some(entry) = this.commits.get(ix).cloned() {
                                                if let Some(repo) = this.active_repository(cx) {
                                                    let registry = GitHostingProviderRegistry::global(cx);
                                                    cx.spawn_in(window, move |_, cx| async move {
                                                        let upstream = repo.update(&cx, |r, _| r.remote_url("upstream".into())).unwrap().await.ok().flatten();
                                                        let origin = repo.update(&cx, |r, _| r.remote_url("origin".into())).unwrap().await.ok().flatten();
                                                        if let Some(remote_url) = upstream.or(origin) {
                                                            if let Some((provider, parsed)) = parse_git_remote_url(registry, &remote_url) {
                                                                let url = provider.build_commit_permalink(&parsed, BuildCommitPermalinkParams { sha: &entry.sha });
                                                                let _ = cx.update(|cx| cx.open_url(url.as_str()));
                                                            }
                                                        }
                                                        Ok(())
                                                    }).detach();
                                                }
                                            }
                                        }))
                                        .action("Copy Branch Name", move |_ev, window, _| {
                                            window.set_clipboard(bname.clone());
                                        })
                                });
                            }));
                            chip_el.into_any_element()
                        }),
                    )
                    .children(remotes.iter().enumerate().map(|(rix, r)| {
                        let rname = r.clone();
                        let id = format!("remote-chip-{}-{}", ix, rix);
                        let chip_el = h_flex()
                            .id(id)
                            .px_1()
                            .border_1()
                            .rounded_sm()
                            .border_color(cx.theme().colors().border)
                            .bg(cx.theme().colors().element_background)
                            .child(Label::new(rname.clone()).size(LabelSize::XSmall).color(Color::Muted));
                        let chip_el = chip_el.on_secondary_mouse_down(cx.listener(move |this, _ev, window, cx| {
                            let focus = this.focus.clone();
                            let name = rname.clone();
                            ContextMenu::build(window, cx, move |menu, _window, cx| {
                                menu.context(focus)
                                    .action("Checkout Tracking Branch", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                        if let Some(repo) = this.active_repository(cx) {
                                            if let Some((remote, branch)) = name.split_once('/') {
                                                let remote = remote.to_string();
                                                let branch = branch.to_string();
                                                let workspace = this.workspace.clone();
                                                cx.spawn_in(window, move |_, cx| async move {
                                                    let res = repo.update(&cx, |r, _| r.create_tracking_branch(remote.clone(), branch.clone())).unwrap().await;
                                                    if let Some(ws) = workspace.upgrade() {
                                                        let _ = ws.update(&cx, |w, cx| {
                                                            let msg = match res {
                                                                Ok(()) => format!("Checked out {}/{} as tracking branch", remote, branch),
                                                                Err(e) => format!("Failed to checkout tracking branch: {}", e),
                                                            };
                                                            let icon = if res.is_ok() { ToastIcon::new(IconName::GitBranchAlt).color(Color::Muted) } else { ToastIcon::new(IconName::XCircle).color(Color::Error) };
                                                            let toast = StatusToast::new(msg, cx, |t, _| t.icon(icon));
                                                            w.toggle_status_toast(toast, cx);
                                                        });
                                                    }
                                                }).detach();
                                            }
                                        }
                                    }))
                                    .action("Delete Remote Branch", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                        if let Some(repo) = this.active_repository(cx) {
                                            if let Some((remote, branch)) = name.split_once('/') {
                                                let remote = remote.to_string();
                                                let branch = branch.to_string();
                                                let workspace = this.workspace.clone();
                                                cx.spawn_in(window, move |_, cx| async move {
                                                    let res = repo.update(&cx, |r, _| r.delete_remote_branch(remote.clone(), branch.clone())).unwrap().await;
                                                    if let Some(ws) = workspace.upgrade() {
                                                        let _ = ws.update(&cx, |w, cx| {
                                                            let msg = match res {
                                                                Ok(()) => format!("Deleted {}/{}", remote, branch),
                                                                Err(e) => format!("Failed to delete {}/{}: {}", remote, branch, e),
                                                            };
                                                            let icon = if res.is_ok() { ToastIcon::new(IconName::GitBranchAlt).color(Color::Muted) } else { ToastIcon::new(IconName::XCircle).color(Color::Error) };
                                                            let toast = StatusToast::new(msg, cx, |t, _| t.icon(icon));
                                                            w.toggle_status_toast(toast, cx);
                                                        });
                                                    }
                                                }).detach();
                                            }
                                        }
                                    }))
                                    .action("Copy Remote Branch", move |_ev, window, _| {
                                        window.set_clipboard(name.clone());
                                    });
                            });
                        }));
                        chip_el.into_any_element()
                    }))
                    .children(tags.iter().enumerate().map(|(tix, t)| {
                        let tname = t.clone();
                        let id = format!("tag-chip-{}-{}", ix, tix);
                        let chip_el = h_flex()
                            .id(id)
                            .px_1()
                            .border_1()
                            .rounded_sm()
                            .border_color(cx.theme().colors().border)
                            .bg(cx.theme().colors().element_background)
                            .child(Label::new(format!("tag: {}", tname)).size(LabelSize::XSmall).color(Color::Info));
                        let chip_el = chip_el.on_secondary_mouse_down(cx.listener(move |this, _ev, window, cx| {
                            let focus = this.focus.clone();
                            ContextMenu::build(window, cx, move |menu, _window, cx| {
                                menu.context(focus)
                                    .action("Open on Remote", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                        if let Some(entry) = this.commits.get(ix).cloned() {
                                            if let Some(repo) = this.active_repository(cx) {
                                                let registry = GitHostingProviderRegistry::global(cx);
                                                cx.spawn_in(window, move |_, cx| async move {
                                                    let upstream = repo.update(&cx, |r, _| r.remote_url("upstream".into())).unwrap().await.ok().flatten();
                                                    let origin = repo.update(&cx, |r, _| r.remote_url("origin".into())).unwrap().await.ok().flatten();
                                                    if let Some(remote_url) = upstream.or(origin) {
                                                        if let Some((provider, parsed)) = parse_git_remote_url(registry, &remote_url) {
                                                            let url = provider.build_commit_permalink(&parsed, BuildCommitPermalinkParams { sha: &entry.sha });
                                                            let _ = cx.update(|cx| cx.open_url(url.as_str()));
                                                        }
                                                    }
                                                    Ok(())
                                                }).detach();
                                            }
                                        }
                                    }))
                                    .action("Delete Tag", cx.listener(move |this: &mut GitGraphPanel, _ev: &ClickEvent, window, cx| {
                                        if let Some(repo) = this.active_repository(cx) {
                                            let tname = tname.clone();
                                            cx.spawn_in(window, move |_, cx| async move {
                                                let _ = repo.update(&cx, |r, _| r.delete_tag(tname.to_string())).unwrap().await;
                                            }).detach();
                                        }
                                    }))
                                    .action("Copy Tag Name", move |_ev, window, _| {
                                        window.set_clipboard(tname.clone());
                                    })
                            });
                        }));
                        chip_el.into_any_element()
                    }));

                item.extend([
                    h_flex()
                        .gap_2()
                        .items_center()
                        .child(Label::new(format!("{}", sha_short)).monospace().size(LabelSize::Small))
                        .child(Label::new(entry.subject.clone()).size(LabelSize::Base))
                        .child(chips)
                        .into_any_element(),
                ]);

                items.push(item.into_any_element());
                }
                items
            }),
        )
        .with_sizing_behavior(ListSizingBehavior::Auto)
        .with_horizontal_sizing_behavior(ListHorizontalSizingBehavior::Unconstrained)
        .track_scroll(self.scroll_handle.clone());

        // Load more footer for large repos
        let load_more = h_flex()
            .px_2()
            .py_1()
            .justify_center()
            .child(
                ButtonLike::new("load-more").on_click(cx.listener(|this, _, _window, cx| {
                    this.max_commits += DEFAULT_MAX_COMMITS;
                    this.request_refresh(None, cx);
                })).child(Label::new("Load more commits…"))
            );

        v_flex()
            .size_full()
            .track_focus(&self.focus)
            .bg(cx.theme().colors().panel_background)
            .on_action(cx.listener(Self::select_next))
            .on_action(cx.listener(Self::select_previous))
            .on_action(cx.listener(Self::select_first))
            .on_action(cx.listener(Self::select_last))
            .on_key_down(cx.listener(|this, ev: &KeyDownEvent, window, cx| {
                let key = ev.keystroke.key.to_lowercase();
                if key == "enter" {
                    this.open_selected(window, cx);
                }
            }))
            .child(
                v_flex()
                    .children([header.into_any_element()])
                    .child(
                        h_flex()
                            .px_2()
                            .py_1()
                            .child(self.filter_input.clone()),
                    ),
            )
            .child(div().border_t_1().border_color(cx.theme().colors().border_variant))
            .child(stashes)
            .child(rows)
            .child(load_more)
    }
}

pub fn register(workspace: &mut Workspace) {
    workspace.register_action(|workspace, _: &Open, window, cx| {
        let handle = workspace.handle();
        cx.spawn_in(window, move |_, cx| async move {
            if let Ok(graph) = GitGraphPanel::load(handle, cx.clone()) {
                workspace.update_in(&cx, |w, window, cx| {
                    w.add_panel(graph, window, cx);
                }).ok();
            }
        }).detach();
    });
}

fn lane_center_px(lane: usize) -> Pixels {
    px(lane as f32 * LANE_PITCH + LANE_PITCH * 0.5)
}

fn lane_color(lane: usize) -> gpui::Background {
    // Deterministic set of colors cycling by lane index
    let palette = [
        gpui::rgb(0x1f77b4), // blue
        gpui::rgb(0xd62728), // red
        gpui::rgb(0x2ca02c), // green
        gpui::rgb(0xff7f0e), // orange
        gpui::rgb(0x9467bd), // purple
        gpui::rgb(0x17becf), // cyan
        gpui::rgb(0xe377c2), // pink
        gpui::rgb(0x7f7f7f), // gray
    ];
    let c = palette[lane % palette.len()];
    c.into()
}

fn compute_layout(commits: &[CommitLogEntry]) -> Vec<RowLayout> {
    use std::collections::HashMap;
    let mut rows = Vec::with_capacity(commits.len());
    let mut active: Vec<Option<SharedString>> = Vec::new();
    let mut pending: HashMap<SharedString, usize> = HashMap::new();

    for entry in commits.iter() {
        // Determine lane for this commit
        let (mut commit_lane, had_reserved);
        if let Some(&lane) = pending.get(&entry.sha) {
            commit_lane = lane;
            had_reserved = true;
            pending.remove(&entry.sha);
            if let Some(slot) = active.get_mut(lane) {
                *slot = None;
            }
        } else {
            // find first empty slot
            if let Some(ix) = active.iter().position(|s| s.is_none()) {
                commit_lane = ix;
            } else {
                commit_lane = active.len();
                active.push(None);
            }
            had_reserved = false;
        }

        let active_before = active.clone();

        // Parents
        let mut connectors = Vec::new();
        if !entry.parents.is_empty() {
            // primary parent stays on commit_lane
            let primary = entry.parents[0].clone();
            if active.len() <= commit_lane { active.resize(commit_lane + 1, None); }
            active[commit_lane] = Some(primary.clone());
            pending.insert(primary, commit_lane);
            connectors.push(commit_lane);

            for p in entry.parents.iter().skip(1) {
                if let Some(&l) = pending.get(p) {
                    if active.len() <= l { active.resize(l + 1, None); }
                    active[l] = Some(p.clone());
                    connectors.push(l);
                } else {
                    // allocate new lane
                    let l = if let Some(ix) = active.iter().position(|s| s.is_none()) {
                        ix
                    } else {
                        let ix = active.len();
                        active.push(None);
                        ix
                    };
                    active[l] = Some(p.clone());
                    pending.insert(p.clone(), l);
                    connectors.push(l);
                }
            }
        } else {
            // no parents: this lane ends here; active[commit_lane] already None
        }

        // After update
        let lane_count = active.len().max(commit_lane + 1);
        let mut verticals = vec![false; lane_count];
        for i in 0..lane_count {
            let before = active_before.get(i).and_then(|o| o.as_ref()).is_some();
            let after = active.get(i).and_then(|o| o.as_ref()).is_some();
            verticals[i] = before || after || (i == commit_lane && !entry.parents.is_empty()) || had_reserved;
        }

        rows.push(RowLayout { lane_count, commit_lane, verticals, connectors });
    }

    rows
}

fn parse_decorations(
use workspace::ModalView;
    decorations: &[SharedString],
) -> (Vec<SharedString>, Vec<SharedString>, Vec<SharedString>) {
    let mut branches = Vec::new();
    let mut remotes = Vec::new();
    let mut tags = Vec::new();
    for d in decorations.iter() {
        let s = d.trim();
        if s.is_empty() { continue; }
        if let Some(rest) = s.strip_prefix("tag: ") {
            tags.push(rest.into());
        } else if s.contains("->") {
            // e.g., HEAD -> main
            if let Some(branch) = s.split("->").nth(1) {
                branches.push(branch.trim().into());
            }
        } else if s.contains('/') {
            remotes.push(s.into());
        } else if s != "HEAD" {
            branches.push(s.into());
        }
    }
    (branches, remotes, tags)
}

struct BranchRenameModal {
    repo: Entity<Repository>,
    old_name: SharedString,
    input: Entity<SingleLineInput>,
}

impl BranchRenameModal {
    fn new(
        repo: Entity<Repository>,
        old_name: SharedString,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let input = cx.new(|cx| SingleLineInput::new(window, cx, "New branch name").start_icon(IconName::Pencil));
        Self { repo, old_name, input }
    }
}

impl ModalView for BranchRenameModal {}

impl Focusable for BranchRenameModal {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.input.focus_handle(cx)
    }
}

impl Render for BranchRenameModal {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let old = self.old_name.clone();
        let repo = self.repo.clone();
        v_flex()
            .w(rems(34.))
            .gap_2()
            .child(Label::new(format!("Rename branch: {}", old)).size(LabelSize::Large))
            .child(self.input.clone())
            .child(
                h_flex()
                    .justify_end()
                    .gap_2()
                    .child(
                        ui::Button::new("cancel").label("Cancel").on_click(cx.listener(|_, _, window, cx| {
                            cx.emit(DismissEvent);
                            window.prevent_default();
                        })),
                    )
                    .child(
                        ui::Button::new("rename").label("Rename").on_click(cx.listener(move |this, _, window, cx| {
                            let new_name = this.input.text(cx);
                            if new_name.trim().is_empty() { return; }
                            let repo = repo.clone();
                            let old_name = old.clone();
                            cx.spawn_in(window, move |_, cx| async move {
                                let _ = repo.update(&cx, |r, _| r.rename_branch(old_name.to_string(), new_name.to_string())).unwrap().await;
                                Ok(())
                            }).detach();
                            cx.emit(DismissEvent);
                        })),
                    ),
            )
    }
}
