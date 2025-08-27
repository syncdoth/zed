use anyhow::{Context as _, Result};
use buffer_diff::{BufferDiff, BufferDiffSnapshot};
use editor::{Editor, EditorEvent, MultiBuffer, SelectionEffects};
use git::repository::{CommitDiff, RepoPath};
use gpui::{AnyElement, AnyView, App, AppContext as _, AsyncApp, Context, Entity, EventEmitter, FocusHandle, Focusable, IntoElement, Render, SharedString, Task, Window};
use language::{Buffer, Capability, File, LanguageRegistry, LineEnding, Rope, TextBuffer};
use multi_buffer::PathKey;
use project::{Project, git_store::Repository};
use std::{any::{Any, TypeId}, path::Path, sync::Arc};
use ui::{Color, Icon, IconName, Label, LabelCommon as _, SharedString as UiSharedString};
use util::ResultExt;
use workspace::{Item, ItemHandle as _, ToolbarItemLocation, Workspace, item::{BreadcrumbText, ItemEvent, TabContentParams}};

pub struct CompareView {
    title: SharedString,
    editor: Entity<Editor>,
    multibuffer: Entity<MultiBuffer>,
}

impl CompareView {
    pub fn open(
        base: String,
        target: String,
        repo: Entity<Repository>,
        workspace: Entity<Workspace>,
        window: &mut Window,
        cx: &mut App,
    ) {
        let diff_task = repo.update(cx, |repo, _| repo.diff_between(base.clone(), target.clone())).ok();
        window.spawn(cx, async move |cx| {
            let diff = diff_task?.await.log_err()?.log_err()?;
            workspace.update_in(&cx, |workspace, window, cx| {
                let project = workspace.project();
                let compare_view = cx.new(|cx| {
                    CompareView::new(format!("{} ↔ {}", &target[..7.min(target.len())], &base[..7.min(base.len())]).into(), diff, repo.clone(), project, window, cx)
                });

                let pane = workspace.active_pane();
                pane.update(cx, |pane, cx| {
                    pane.add_item(Box::new(compare_view.clone()), true, true, None, window, cx);
                });
            }).ok();
            Ok(())
        }).detach();
    }

    pub fn new(
        title: SharedString,
        diff: CommitDiff,
        repository: Entity<Repository>,
        project: Entity<Project>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let language_registry = project.read(cx).languages().clone();
        let multibuffer = cx.new(|_| MultiBuffer::new(Capability::ReadOnly));
        let editor = cx.new(|cx| {
            let mut editor = Editor::for_multibuffer(multibuffer.clone(), Some(project.clone()), window, cx);
            editor.disable_inline_diagnostics();
            editor.set_expand_all_diff_hunks(cx);
            editor
        });

        let first_worktree_id = project.read(cx).worktrees(cx).next().map(|wt| wt.read(cx).id());

        cx.spawn(async move |this, cx| {
            for file in diff.files {
                let is_deleted = file.new_text.is_none();
                let new_text = file.new_text.unwrap_or_default();
                let old_text = file.old_text;
                let worktree_id = repository
                    .update(cx, |repository, cx| {
                        repository
                            .repo_path_to_project_path(&file.path, cx)
                            .map(|p| p.worktree_id)
                            .or(first_worktree_id)
                    })?
                    .context("project has no worktrees")?;
                let blob = Arc::new(CompareBlob { path: file.path.clone(), worktree_id, is_deleted }) as Arc<dyn language::File>;

                let buffer = build_buffer(new_text, blob, &language_registry, cx).await?;
                let buffer_diff = build_buffer_diff(old_text, &buffer, &language_registry, cx).await?;

                this.update(cx, |this, cx| {
                    this.multibuffer.update(cx, |multibuffer, cx| {
                        let snapshot = buffer.read(cx).snapshot();
                        let diff = buffer_diff.read(cx);
                        let diff_hunk_ranges = diff
                            .hunks_intersecting_range(language::Anchor::MIN..language::Anchor::MAX, &snapshot, cx)
                            .map(|h| h.buffer_range.to_point(&snapshot))
                            .collect::<Vec<_>>();
                        let path = snapshot.file().unwrap().path().clone();
                        multibuffer.set_excerpts_for_path(
                            PathKey::namespaced(1, path),
                            buffer,
                            diff_hunk_ranges,
                            editor::DEFAULT_MULTIBUFFER_CONTEXT,
                            cx,
                        );
                    });
                })?;
            }
            Ok(())
        }).detach();

        Self { title, editor, multibuffer }
    }
}

struct CompareBlob {
    path: RepoPath,
    worktree_id: project::WorktreeId,
    is_deleted: bool,
}

impl language::File for CompareBlob {
    fn path(&self) -> &Path { self.path.as_ref() }
    fn worktree_id(&self) -> project::WorktreeId { self.worktree_id }
    fn to_proto(&self, _: &App) -> language::proto::File { unimplemented!() }
    fn is_private(&self) -> bool { false }
}

async fn build_buffer(
    mut text: String,
    file: Arc<dyn File>,
    language_registry: &Arc<LanguageRegistry>,
    cx: &mut AsyncApp,
) -> Result<Entity<Buffer>> {
    let line_ending = LineEnding::detect(&text);
    LineEnding::normalize(&mut text);
    let text = Rope::from(text);
    let language = cx.update(|cx| language_registry.language_for_file(&file, Some(&text), cx))?;
    let language = if let Some(language) = language {
        language_registry.load_language(&language).await.ok().and_then(|e| e.log_err())
    } else { None };
    let buffer = cx.new(|cx| {
        let buffer = TextBuffer::new_normalized(0, cx.entity_id().as_non_zero_u64().into(), line_ending, text);
        let mut buffer = Buffer::build(buffer, Some(file), Capability::ReadWrite);
        buffer.set_language(language, cx);
        buffer
    })?;
    Ok(buffer)
}

async fn build_buffer_diff(
    mut old_text: Option<String>,
    buffer: &Entity<Buffer>,
    language_registry: &Arc<LanguageRegistry>,
    cx: &mut AsyncApp,
) -> Result<Entity<BufferDiff>> {
    if let Some(old_text) = &mut old_text { LineEnding::normalize(old_text); }
    let buffer = cx.update(|cx| buffer.read(cx).snapshot())?;
    let base_buffer = cx.update(|cx| {
        Buffer::build_snapshot(old_text.as_deref().unwrap_or("").into(), buffer.language().cloned(), Some(language_registry.clone()), cx)
    })?.await;
    let diff_snapshot = cx.update(|cx| {
        BufferDiffSnapshot::new_with_base_buffer(buffer.text.clone(), old_text.map(Arc::new), base_buffer, cx)
    })?.await;
    cx.new(|cx| { let mut diff = BufferDiff::new(&buffer.text, cx); diff.set_snapshot(diff_snapshot, &buffer.text, cx); diff })
}

impl EventEmitter<EditorEvent> for CompareView {}

impl Focusable for CompareView {
    fn focus_handle(&self, cx: &App) -> FocusHandle { self.editor.focus_handle(cx) }
}

impl Item for CompareView {
    fn tab_tooltip_text(&self, _: &App) -> Option<UiSharedString> { None }
    fn tab_content(&self, _index: usize, _cx: &App) -> AnyView {
        Label::new(self.title.clone()).into_any_view()
    }
    fn tab_content_text(&self, _: usize, _: &App) -> UiSharedString { UiSharedString::from(self.title.as_ref()) }
    fn toolbar_items(&self, _cx: &Context<'_, Self>) -> Vec<(ToolbarItemLocation, Vec<AnyElement>)> { Vec::new() }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn receive_pan(&mut self, _: f32, _: f32, _: &mut Window, _: &mut Context<Self>) {}
}

impl Render for CompareView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        self.editor.clone()
    }
}
