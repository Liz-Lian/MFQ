/** 聊天路由页面，仅组合会话导航、消息列表和输入区域；业务由聊天领域拥有。 */
import { lazy, Suspense, useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { ArrowDownIcon } from '@phosphor-icons/react';
import { useChat } from './ChatProvider';
import { useSettings } from '../settings/SettingsProvider';
import { useChatAutoScroll } from './hooks/useChatAutoScroll';
import { Icon } from '../../app/display';
import { formatNumber, errorMessage } from '../../app/formatters';
import { AudioClip } from '../voice/AudioClip';
import { renderMarkdown } from './MessageMarkdown';
import { StreamingMessage } from './components/StreamingMessage';
import { ChatComposer } from './components/ChatComposer';
import { ChatToolbar } from './components/ChatToolbar';
const SavedMessageList = lazy(() =>
  import('./SavedMessageList').then((module) => ({ default: module.SavedMessageList })),
);

/** 渲染聊天页，页面离开后生成控制器由外层 ChatProvider 保留。 */
export function ChatPage() {
  const chat = useChat();
  const navigate = useNavigate();
  const { tr } = useSettings();
  const {
    conversation,
    inference,
    voice,
    attachments: attachmentState,
    messageActions,
    generation,
    generationPhase,
    busy,
    recoveryNeeded,
    send,
    clearActiveConversation,
    voiceComponentBusy,
    installOrEnableVoiceOutput,
    mcpTools,
  } = chat;
  const {
    sessions,
    active,
    activeId,
    messages,
    responses,
    transitioning: sessionTransitioning,
    modelAvailable: selectedModelAvailable,
    conversationReady,
    error,
    setError,
  } = conversation;
  const {
    availableModelNames,
    selectedModel: model,
    effectiveSettings,
    selectedModelLoading,
    voiceComponent,
    jobs,
  } = inference;
  const { attachments, selectAttachments: selectAttachments, removeAttachment } = attachmentState;
  const { editDraft, setEditDraft, saveEdit, copyMessage, regenerate, executeToolCalls } =
    messageActions;
  const { liveVoice, voiceMessages } = voice;
  const currentVoiceMessages = voiceMessages.filter((message) => message.sessionId === activeId);
  const attachmentAccept = inference.attachmentAccept;
  const live = generation.getSnapshot().live;
  const stopGeneration = generation.stop;
  const voiceComponentJob = jobs.find(
    (job) =>
      job.kind === 'component.voice_output.install' &&
      ['queued', 'running', 'cancelling'].includes(job.status),
  );
  const needsVoiceOutputComponent =
    active?.mode === 'full_duplex' &&
    Boolean(inference.capabilities?.model_capabilities.features.audio_output) &&
    !inference.realtimeAvailable;
  const [chatSessionsOpen, setChatSessionsOpen] = useState(
    () => window.matchMedia('(min-width: 681px)').matches,
  );
  const {
    scrollerRef: messageScrollerRef,
    following,
    handleScroll: handleMessageScroll,
    scrollToBottom,
  } = useChatAutoScroll(activeId, true);
  useEffect(() => {
    const viewport = window.matchMedia('(max-width: 680px)');
    const close = () => {
      if (viewport.matches) setChatSessionsOpen(false);
    };
    viewport.addEventListener('change', close);
    return () => viewport.removeEventListener('change', close);
  }, []);
  /** 选择历史会话时折叠移动端导航，避免遮挡消息。 */
  function selectSession(id: string) {
    if (busy) return;
    conversation.selectSession(id);
    if (window.matchMedia('(max-width: 680px)').matches) setChatSessionsOpen(false);
  }
  /** 创建空会话后收起移动端导航。 */
  async function createSession() {
    await conversation.createSession(active?.mode);
    if (window.matchMedia('(max-width: 680px)').matches) setChatSessionsOpen(false);
  }
  /** 模型加载操作由模型页面负责，聊天页只提供入口。 */
  function chooseModelDirectory() {
    navigate('/models');
  }
  /** 禁止生成期间改变选中模型。 */
  function selectModel(value: string) {
    if (!busy) inference.setSelectedModel(value);
  }
  return (
    <section className={'chat-view ' + (chatSessionsOpen ? 'sessions-open' : 'sessions-collapsed')}>
      <aside
        className={'chat-session-sidebar' + (chatSessionsOpen ? ' open' : '')}
        aria-label={tr('会话列表', 'Conversations')}
      >
        <div className="chat-session-sidebar-header">
          <strong>{tr('对话', 'Chats')}</strong>
          <button
            aria-label={tr('新建会话', 'New chat')}
            className="chat-icon-button"
            disabled={busy || sessionTransitioning || !selectedModelAvailable}
            onClick={() => void createSession()}
            title={tr('新建会话', 'New chat')}
            type="button"
          >
            <Icon name="plus" size={15} />
          </button>
        </div>
        <div className="chat-session-list">
          {sessions.length ? (
            sessions.map((session) => (
              <button
                aria-current={session.id === activeId ? 'page' : undefined}
                className={session.id === activeId ? 'active' : ''}
                disabled={busy || sessionTransitioning}
                key={session.id}
                onClick={() => selectSession(session.id)}
                title={session.title || tr('未命名会话', 'Untitled chat')}
                type="button"
              >
                <strong>{session.title || tr('未命名会话', 'Untitled chat')}</strong>
                <small>{session.model}</small>
              </button>
            ))
          ) : (
            <p>{tr('暂无会话', 'No conversations yet')}</p>
          )}
        </div>
      </aside>
      <header className="chat-screen-header">
        <div className="chat-screen-title">
          <button
            aria-expanded={chatSessionsOpen}
            aria-label={
              chatSessionsOpen
                ? tr('收起会话列表', 'Collapse conversations')
                : tr('展开会话列表', 'Expand conversations')
            }
            className="chat-sidebar-toggle"
            onClick={() => setChatSessionsOpen((open) => !open)}
            title={
              chatSessionsOpen
                ? tr('收起会话列表', 'Collapse conversations')
                : tr('展开会话列表', 'Expand conversations')
            }
            type="button"
          >
            <span aria-hidden="true">{chatSessionsOpen ? '‹' : '›'}</span>
          </button>
          <h1>{active?.title || tr('对话', 'Chat')}</h1>
        </div>
        <div className="chat-screen-actions">
          <div className="chat-model-summary">
            {availableModelNames.length > 1 ? (
              <select
                aria-label={tr('对话模型', 'Chat model')}
                disabled={busy || sessionTransitioning}
                onChange={(event) => selectModel(event.target.value)}
                value={model}
              >
                {availableModelNames.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            ) : (
              <strong>{model || tr('尚未加载模型', 'No model loaded')}</strong>
            )}
            <small>
              {tr(
                '最多 ' + formatNumber(effectiveSettings.maxTokens) + ' tokens',
                formatNumber(effectiveSettings.maxTokens) + ' max tokens',
              )}{' '}
              · {tr('温度', 'temperature')} {formatNumber(effectiveSettings.temperature, 2)} ·{' '}
              {tr('流式', 'streaming')}
            </small>
          </div>
          <span className={'runtime-status-pill ' + (conversationReady ? 'running' : 'stopped')}>
            <i />
            {conversationReady
              ? tr('就绪', 'Ready')
              : selectedModelLoading
                ? tr('加载中', 'Loading')
                : tr('空闲', 'Idle')}
          </span>
          <button
            aria-label={tr('清空对话', 'Clear conversation')}
            className="chat-icon-button"
            disabled={
              !conversationReady || busy || (!messages.length && !currentVoiceMessages.length)
            }
            onClick={() => void clearActiveConversation()}
            title={tr('清空对话', 'Clear conversation')}
            type="button"
          >
            <Icon name="trash" size={14} />
          </button>
        </div>
      </header>
      <div className="message-scroller" onScroll={handleMessageScroll} ref={messageScrollerRef}>
        <div className="message-list">
          {!messages.length && !currentVoiceMessages.length && !live && (
            <div className="welcome">
              <Icon name="chat" size={34} />
              {!selectedModelAvailable ? (
                <>
                  <h1>
                    {selectedModelLoading
                      ? tr('模型加载中', 'Model loading')
                      : tr('尚未加载模型', 'No model loaded')}
                  </h1>
                  <p>
                    {selectedModelLoading
                      ? tr(
                          '加载完成后即可开始对话。',
                          'Chat becomes available as soon as loading completes.',
                        )
                      : tr(
                          '选择本地检查点后即可开始对话。',
                          'Choose a local checkpoint to use the inference playground.',
                        )}
                  </p>
                  {!selectedModelLoading && (
                    <button
                      className="open-model-primary"
                      disabled={busy}
                      onClick={() => void chooseModelDirectory()}
                      type="button"
                    >
                      <Icon name="folder" />
                      {tr('选择模型', 'Choose model')}
                    </button>
                  )}
                </>
              ) : !active ? (
                <>
                  <h1>{tr('开始对话', 'Start a conversation')}</h1>
                  <p>
                    {tr('连接到当前配置的 MFQ 服务。', 'Connected to your configured MFQ service.')}
                  </p>
                  <button
                    className="open-model-primary"
                    disabled={busy}
                    onClick={() => void createSession()}
                    type="button"
                  >
                    {tr('开始对话', 'Start chat')}
                  </button>
                </>
              ) : (
                <>
                  <h1>{tr('开始对话', 'Start a conversation')}</h1>
                  <p>
                    {tr('连接到当前配置的 MFQ 服务。', 'Connected to your configured MFQ service.')}
                  </p>
                </>
              )}
            </div>
          )}
          <Suspense fallback={<p role="status">{tr('正在加载消息…', 'Loading messages…')}</p>}>
            <SavedMessageList
              messages={messages}
              responses={responses}
              mcpTools={mcpTools}
              busy={busy}
              tr={tr}
              editDraft={editDraft}
              setEditDraft={setEditDraft}
              actions={{ saveEdit, copyMessage, regenerate, executeToolCalls }}
            />
          </Suspense>
          {currentVoiceMessages.map((message) => (
            <article className={`message message-${message.role}`} key={message.id}>
              <div className="message-body">
                {message.text && renderMarkdown(message.text, false, message.role === 'assistant')}
                {message.audioId && <AudioClip audioId={message.audioId} />}
              </div>
            </article>
          ))}
          {liveVoice?.sessionId === activeId && liveVoice.text && (
            <article className="message message-assistant live-message">
              <div className="message-body">{renderMarkdown(liveVoice.text, true, true)}</div>
            </article>
          )}
          <StreamingMessage controller={generation} sessionId={activeId} tr={tr} />
        </div>
      </div>
      {!following && (
        <button
          className="chat-scroll-bottom"
          onClick={scrollToBottom}
          aria-label={tr('回到底部', 'Scroll to bottom')}
          type="button"
        >
          <ArrowDownIcon size={16} aria-hidden="true" />
        </button>
      )}
      {error && (
        <div className="error-banner" role="alert">
          <span>{error}</span>
          <button onClick={() => setError(null)} type="button">
            ×
          </button>
        </div>
      )}
      <div className="composer-region">
        {needsVoiceOutputComponent && voiceComponent && (
          <div className="voice-component-banner">
            <div>
              <strong>
                {voiceComponent.ready
                  ? tr('语音组件已下载', 'Voice component downloaded')
                  : tr('此模型还缺少语音输出组件', 'This model needs the voice output component')}
              </strong>
              <span>
                {voiceComponentJob
                  ? tr(
                      `正在下载并校验 · ${formatNumber(voiceComponentJob.progress * 100)}%`,
                      `Downloading and verifying · ${formatNumber(voiceComponentJob.progress * 100)}%`,
                    )
                  : voiceComponent.error
                    ? voiceComponent.error
                    : tr(
                        'Token2Wav 独立安装，不会重复占用每个模型的空间。',
                        'Token2Wav is installed once and shared by all compatible models.',
                      )}
              </span>
            </div>
            {voiceComponentJob && <progress max={1} value={voiceComponentJob.progress} />}
            <button
              disabled={voiceComponentBusy || Boolean(voiceComponentJob)}
              onClick={() => void installOrEnableVoiceOutput()}
              type="button"
            >
              {voiceComponentJob
                ? tr('正在下载…', 'Downloading…')
                : voiceComponent.ready
                  ? tr('启用语音输出', 'Enable voice output')
                  : tr(
                      `下载组件 · ${formatNumber(voiceComponent.total_bytes / 1e9, 2)} GB`,
                      `Download · ${formatNumber(voiceComponent.total_bytes / 1e9, 2)} GB`,
                    )}
            </button>
          </div>
        )}
        <ChatComposer
          sessionId={activeId ?? 'new'}
          ready={conversationReady}
          busy={busy}
          recoveryNeeded={recoveryNeeded}
          phase={generationPhase}
          placeholder={
            conversationReady
              ? tr('向模型发送消息', 'Message MFQ')
              : selectedModelAvailable
                ? tr('正在加载会话', 'Loading conversation')
                : tr('请先加载模型', 'Load a model first')
          }
          attachments={attachments}
          attachmentAccept={attachmentAccept}
          tr={tr}
          onSend={send}
          onStop={stopGeneration}
          onSelectAttachments={selectAttachments}
          onRemoveAttachment={removeAttachment}
          onError={(cause) => setError(errorMessage(cause))}
          toolbar={<ChatToolbar />}
        />
        <p>
          {tr(
            '模型输出可能存在错误，请核对重要信息。',
            'Model output may be inaccurate. Verify important information.',
          )}
        </p>
      </div>
    </section>
  );
}
