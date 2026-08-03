use std::{collections::HashSet, ops::Range, sync::Arc};

use editor::{Editor, EditorEvent};
use git::repository::Branch;
use gpui::{
    App, Context, DismissEvent, Entity, EventEmitter, FocusHandle, Focusable, Subscription,
    WeakEntity, Window, uniform_list,
};
use menu::Cancel;
use ui::{Checkbox, Divider, ToggleState, prelude::*};
use util::ResultExt as _;

use crate::git_graph::GitGraph;

/// The multi-select branch picker shown in the git graph "Branches" dropdown.
/// Selecting branches narrows the graph to those refs; "Show all" clears the
/// selection. Toggling a row applies immediately via the owning [`GitGraph`].
pub struct GitGraphBranchFilter {
    git_graph: WeakEntity<GitGraph>,
    branches: Vec<BranchEntry>,
    /// Canonical ref names currently selected. Empty means "show all".
    selected: HashSet<SharedString>,
    show_local_branches: bool,
    show_remote_branches: bool,
    query_editor: Entity<Editor>,
    _subscriptions: Vec<Subscription>,
}

struct BranchEntry {
    ref_name: SharedString,
    name: SharedString,
    name_lowercase: String,
    is_remote: bool,
}

impl GitGraphBranchFilter {
    pub fn new(
        git_graph: WeakEntity<GitGraph>,
        branches: Vec<Branch>,
        selected: HashSet<SharedString>,
        show_local_branches: bool,
        show_remote_branches: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let branches = branches
            .into_iter()
            .map(|branch| {
                let name = SharedString::from(branch.name().to_string());
                let is_remote = branch.is_remote();
                BranchEntry {
                    ref_name: branch.ref_name,
                    name_lowercase: name.to_lowercase(),
                    name,
                    is_remote,
                }
            })
            .collect();

        let query_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_placeholder_text("Filter branches…", window, cx);
            editor
        });
        let _subscriptions = vec![cx.subscribe(&query_editor, |_, _, _: &EditorEvent, cx| {
            cx.notify();
        })];

        Self {
            git_graph,
            branches,
            selected,
            show_local_branches,
            show_remote_branches,
            query_editor,
            _subscriptions,
        }
    }

    pub(crate) fn normalize_selected_refs(
        branches: &[Branch],
        selected: &HashSet<SharedString>,
    ) -> HashSet<SharedString> {
        selected
            .iter()
            .filter_map(|selected_ref| {
                branches
                    .iter()
                    .find(|branch| branch.ref_name.as_ref() == selected_ref.as_ref())
                    .map(|branch| branch.ref_name.clone())
                    .or_else(|| {
                        let mut matching_branches = branches
                            .iter()
                            .filter(|branch| branch.name() == selected_ref.as_ref());
                        let branch = matching_branches.next()?;
                        matching_branches
                            .next()
                            .is_none()
                            .then(|| branch.ref_name.clone())
                    })
                    .or_else(|| {
                        // Keep canonical refs even when the branch is gone;
                        // `git log --ignore-missing` tolerates stale refs, and
                        // dropping them can silently widen the filter to "show all".
                        selected_ref
                            .starts_with("refs/")
                            .then(|| selected_ref.clone())
                    })
            })
            .collect()
    }

    fn apply(&self, cx: &mut App) {
        let selected_refs = if self.selected.is_empty() {
            None
        } else {
            Some(
                self.selected
                    .iter()
                    .cloned()
                    .collect::<Arc<[SharedString]>>(),
            )
        };
        self.git_graph
            .update(cx, |git_graph, cx| {
                git_graph.apply_branch_selection(selected_refs, cx);
            })
            .log_err();
    }

    fn toggle(&mut self, ref_name: SharedString, cx: &mut Context<Self>) {
        if !self.selected.remove(&ref_name) {
            self.selected.insert(ref_name);
        }
        self.apply(cx);
        cx.notify();
    }

    fn show_all(&mut self, cx: &mut Context<Self>) {
        self.selected.clear();
        self.apply(cx);
        cx.notify();
    }
}

impl EventEmitter<DismissEvent> for GitGraphBranchFilter {}

impl Focusable for GitGraphBranchFilter {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.query_editor.focus_handle(cx)
    }
}

impl Render for GitGraphBranchFilter {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let query = self.query_editor.read(cx).text(cx).to_lowercase();
        let hover_bg = cx.theme().colors().element_hover;
        let border_color = cx.theme().colors().border_variant;

        let branch_indices = self
            .branches
            .iter()
            .enumerate()
            .filter(|(_, branch)| {
                (branch.is_remote && self.show_remote_branches
                    || !branch.is_remote && self.show_local_branches
                    || self.selected.contains(&branch.ref_name))
                    && (query.is_empty() || branch.name_lowercase.contains(&query))
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let branch_count = branch_indices.len();

        v_flex()
            .key_context("GitGraphBranchFilter")
            .on_action(cx.listener(|_, _: &Cancel, _, cx| cx.emit(DismissEvent)))
            .elevation_2(cx)
            .w_72()
            .overflow_hidden()
            .child(
                div()
                    .p_1p5()
                    .border_b_1()
                    .border_color(border_color)
                    .child(self.query_editor.clone()),
            )
            .child(
                h_flex()
                    .id("git-graph-show-all")
                    .w_full()
                    .px_2()
                    .py_0p5()
                    .gap_1p5()
                    .rounded_sm()
                    .cursor_pointer()
                    .hover(move |style| style.bg(hover_bg))
                    .child(Checkbox::new(
                        "git-graph-show-all-cb",
                        ToggleState::from(self.selected.is_empty()),
                    ))
                    .child(Label::new("Show all branches").size(LabelSize::Small))
                    .on_click(cx.listener(|this, _, _, cx| this.show_all(cx))),
            )
            .child(Divider::horizontal())
            .child(
                uniform_list(
                    "git-graph-branch-list",
                    branch_count,
                    cx.processor(move |this, range: Range<usize>, _window, cx| {
                        range
                            .filter_map(|index| {
                                let branch_index = *branch_indices.get(index)?;
                                let branch = this.branches.get(branch_index)?;
                                let ref_name = branch.ref_name.clone();
                                let checked = this.selected.contains(&ref_name);
                                let icon = if branch.is_remote {
                                    IconName::Server
                                } else {
                                    IconName::GitBranch
                                };

                                Some(
                                    h_flex()
                                        .id(("git-graph-branch", branch_index))
                                        .w_full()
                                        .px_2()
                                        .py_0p5()
                                        .gap_1p5()
                                        .rounded_sm()
                                        .cursor_pointer()
                                        .hover(move |style| style.bg(hover_bg))
                                        .child(Checkbox::new(
                                            ("git-graph-branch-cb", branch_index),
                                            ToggleState::from(checked),
                                        ))
                                        .child(
                                            Icon::new(icon)
                                                .size(IconSize::Small)
                                                .color(Color::Muted),
                                        )
                                        .child(
                                            Label::new(branch.name.clone())
                                                .size(LabelSize::Small)
                                                .truncate(),
                                        )
                                        .on_click(cx.listener(move |this, _, _, cx| {
                                            this.toggle(ref_name.clone(), cx)
                                        })),
                                )
                            })
                            .collect()
                    }),
                )
                .max_h_80()
                .py_0p5(),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn branch(ref_name: &str) -> Branch {
        Branch {
            is_head: false,
            ref_name: ref_name.into(),
            upstream: None,
            most_recent_commit: None,
        }
    }

    #[test]
    fn normalize_selected_refs_migrates_unambiguous_legacy_names() {
        let branches = vec![
            branch("refs/heads/main"),
            branch("refs/heads/origin/main"),
            branch("refs/remotes/origin/main"),
        ];
        let selected = HashSet::from([
            SharedString::from("main"),
            SharedString::from("origin/main"),
            SharedString::from("refs/remotes/origin/main"),
            SharedString::from("deleted"),
            SharedString::from("refs/heads/deleted"),
        ]);

        assert_eq!(
            GitGraphBranchFilter::normalize_selected_refs(&branches, &selected),
            HashSet::from([
                SharedString::from("refs/heads/main"),
                SharedString::from("refs/remotes/origin/main"),
                SharedString::from("refs/heads/deleted"),
            ])
        );
    }
}
