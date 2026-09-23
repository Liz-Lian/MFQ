"""检查会话操作契约；完整发送、取消与恢复由浏览器测试覆盖。"""
from pathlib import Path
from tests.studio_sources import read_studio_sources, read_studio_styles

ROOT = Path(__file__).resolve().parents[1]
CSS = read_studio_styles()
APP = read_studio_sources('App.tsx', 'features/chat', 'features/models', 'app')
GENERATION = read_studio_sources('features/chat/state/generationController.ts')


def test_simplified_chat_can_clear_and_replace_the_active_session() -> None:
    assert "async function clearActiveConversation()" in APP
    assert "const replacement = await api.createSession(active.model, active.mode)" in APP
    assert "await api.deleteSession(active.id)" in APP
    assert "setActiveId(replacement.id)" in APP
    assert "setMessages([])" in APP


def test_user_messages_can_be_edited_and_assistant_messages_regenerated() -> None:
    assert "async function saveEdit(message: Message)" in APP
    assert "setEditDraft({ messageId: message.id, text: parts.text })" in APP
    assert "const rewound = await api.rewindSession(" in APP
    assert "messages.slice(0, index)" in APP
    assert "await generate(rewound, parts, false)" in APP
    assert "expected_revision: session.revision" in APP
    assert 'tr("保存", "Save")' in APP
    assert 'tr("保存到新分支", "Save as branch")' not in APP
    assert ".message-editor" in CSS
    assert 'message.role === "assistant" && <button' in APP


def test_regenerate_rewinds_to_the_preceding_user_message() -> None:
    assert "async function regenerate(message: Message)" in APP
    actions = read_studio_sources('features/chat/hooks/useMessageActions.ts')
    assert '.slice(0, index)' in actions
    assert '.reverse()' in actions
    assert ".find((item) => item.role === 'user')" in actions
    assert "api.rewindSession(" in APP
    assert "user.id," in APP
    assert "await generate(rewound, user.parts, false)" in APP


def test_stop_generation_cancels_the_server_before_aborting_the_stream() -> None:
    stop = GENERATION.index("stop = async")
    cancel = GENERATION.index("await this.dependencies.cancel(run.sessionId, controller.signal)", stop)
    abort = GENERATION.index("run.controller.abort()", cancel)
    assert cancel < abort
    assert "tr('正在停止生成', 'Stopping generation')" in APP


def test_generation_does_not_inject_a_hidden_system_prompt() -> None:
    assert "LANGUAGE_CONSISTENCY_PROMPT" not in APP
    assert "Before answering, identify the language" not in APP
    assert "system_prompt: inference.effectiveSettings.systemPrompt.trim()" in APP


def test_media_attachments_are_previewed_uploaded_and_sent_as_typed_parts() -> None:
    assert "const [attachments, setAttachments]" in APP
    assert "api.uploadMedia(attachment.file)" in APP
    assert 'accept={attachmentAccept}' in APP
    assert 'type: attachment.kind' in APP
    assert 'mediaMetadata(attachment.file, attachment.kind)' in APP
    assert "<MediaPartView" in APP
    assert ".attachment-tray" in CSS


def test_generation_tracks_the_exact_target_message() -> None:
    assert "item.request_id === run.id" in GENERATION
    assert "this.active !== run" in GENERATION
    assert "response.output_message_id" in GENERATION
    assert "current.setMessages(messages)" in APP
    assert "messages.pop()" not in APP


def test_runtime_controls_wait_until_the_runtime_is_ready() -> None:
    assert 'isRuntimeReady(status.runtime_state)' in APP
    assert 'Promise.resolve<RuntimeModel[]>([])' in APP
    assert "instance.state !== 'ready'" in APP
    assert 'ready || location.pathname' in APP
