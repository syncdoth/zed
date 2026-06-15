use std::{collections::HashSet, sync::Arc};

use editor::{Editor, EditorEvent};
use git::repository::Branch;
use gpui::{
    App, Context, DismissEvent, Entity, EventEmitter, FocusHandle, Focusable, Subscription,
    WeakEntity, Window,
};
use ui::{Checkbox, Divider, ToggleState, prelude::*};

use crate::git_graph::GitGraph;

/// The multi-select branch picker shown in the git graph "Branches" dropdown.
/// Selecting branches narrows the graph to those refs; "Show all" clears the
/// selection. Toggling a row applies immediately via the owning [`GitGraph`].
pub struct GitGraphBranchFilter {
    git_graph: WeakEntity<GitGraph>,
    branches: Vec<BranchEntry>,
    /// Short ref names (e.g. `main`, `origin/main`) currently selected. Empty
    /// means "show all".
    selected: HashSet<SharedString>,
    query_editor: Entity<Editor>,
    focus_handle: FocusHandle,
    _subscriptions: Vec<Subscription>,
}

struct BranchEntry {
    name: SharedString,
    is_remote: bool,
}

impl GitGraphBranchFilter {
    pub fn new(
        git_graph: WeakEntity<GitGraph>,
        branches: Vec<Branch>,
        selected: HashSet<SharedString>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let branches = branches
            .into_iter()
            .map(|branch| BranchEntry {
                name: SharedString::from(branch.name().to_string()),
                is_remote: branch.is_remote(),
            })
            .collect();

        let query_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_placeholder_text("Filter branches…", window, cx);
            editor
        });
        let focus_handle = cx.focus_handle();
        let _subscriptions = vec![cx.subscribe(
            &query_editor,
            |_, _, _: &EditorEvent, cx| {
                cx.notify();
            },
        )];

        Self {
            git_graph,
            branches,
            selected,
            query_editor,
            focus_handle,
            _subscriptions,
        }
    }

    fn apply(&self, cx: &mut App) {
        let selected_refs = if self.selected.is_empty() {
            None
        } else {
            Some(self.selected.iter().cloned().collect::<Arc<[SharedString]>>())
        };
        self.git_graph
            .update(cx, |git_graph, cx| {
                git_graph.apply_branch_selection(selected_refs, cx);
            })
            .ok();
    }

    fn toggle(&mut self, name: SharedString, cx: &mut Context<Self>) {
        if !self.selected.remove(&name) {
            self.selected.insert(name);
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
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for GitGraphBranchFilter {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let query = self.query_editor.read(cx).text(cx).to_lowercase();
        let hover_bg = cx.theme().colors().element_hover;
        let border_color = cx.theme().colors().border_variant;

        // Pure row data, computed without `cx` so the element closures below can
        // borrow `cx` for their listeners without conflicting.
        let entries = self
            .branches
            .iter()
            .filter(|branch| query.is_empty() || branch.name.to_lowercase().contains(&query))
            .map(|branch| {
                (
                    branch.name.clone(),
                    branch.is_remote,
                    self.selected.contains(&branch.name),
                )
            })
            .collect::<Vec<_>>();

        v_flex()
            .key_context("GitGraphBranchFilter")
            .track_focus(&self.focus_handle)
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
                v_flex()
                    .id("git-graph-branch-list")
                    .max_h_80()
                    .overflow_y_scroll()
                    .py_0p5()
                    .children(entries.into_iter().map(|(name, is_remote, checked)| {
                        let icon = if is_remote {
                            IconName::Server
                        } else {
                            IconName::GitBranch
                        };
                        h_flex()
                            .id(SharedString::from(format!("git-graph-branch-{name}")))
                            .w_full()
                            .px_2()
                            .py_0p5()
                            .gap_1p5()
                            .rounded_sm()
                            .cursor_pointer()
                            .hover(move |style| style.bg(hover_bg))
                            .child(Checkbox::new(
                                SharedString::from(format!("git-graph-branch-cb-{name}")),
                                ToggleState::from(checked),
                            ))
                            .child(Icon::new(icon).size(IconSize::Small).color(Color::Muted))
                            .child(Label::new(name.clone()).size(LabelSize::Small).truncate())
                            .on_click(
                                cx.listener(move |this, _, _, cx| this.toggle(name.clone(), cx)),
                            )
                    })),
            )
    }
}
