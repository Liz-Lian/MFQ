/** 组合聊天领域的会话、消息操作、附件和语音生命周期，跨页面保留进行中的生成。 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { useLocation } from 'react-router';
import {
  api,
  type ContentPart,
  type McpToolResource,
  type Session,
  type SessionMode,
} from '../../api';
import { studioConfirm } from '../../studio';
import { useRuntime } from '../../app/RuntimeProvider';
import { errorMessage } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { useConversationSessions } from './hooks/useConversationSessions';
import { useChatGeneration } from './hooks/useChatGeneration';
import { useChatAttachments } from './hooks/useChatAttachments';
import { useChatInference } from './hooks/useChatInference';
import { useMessageActions } from './hooks/useMessageActions';
import { useVoiceConversation } from '../voice/useVoiceConversation';
import { isGenerationBusy } from './state/generationController';

/** 按聊天访问惰性加载数据，生成和语音控制器不会因为切换其他页面而丢失。 */
function useChatDomain() {
  const location = useLocation();
  const [visited, setVisited] = useState(location.pathname === '/chat');
  useEffect(() => {
    if (location.pathname === '/chat') setVisited(true);
  }, [location.pathname]);
  const conversationRef = useRef<ReturnType<typeof useConversationSessions> | null>(null);
  const { controller: generation, phase: generationPhase } = useChatGeneration({
    onSessionState: (id, state, revision) =>
      conversationRef.current?.setSessions((current) =>
        current.map((session) => (session.id === id ? { ...session, state, revision } : session)),
      ),
    onSynchronized: ({ session, messages, responses }) => {
      const current = conversationRef.current;
      if (!current) return;
      current.setSessions((items) =>
        items.map((item) => (item.id === session.id ? session : item)),
      );
      if (current.activeIdRef.current !== session.id) return;
      current.setMessages(messages);
      current.setResponses(
        Object.fromEntries(
          responses
            .filter((response) => response.output_message_id)
            .map((response) => [response.output_message_id!, response]),
        ),
      );
    },
  });
  const conversation = useConversationSessions(
    visited || location.pathname === '/chat',
    isGenerationBusy(generationPhase),
  );
  conversationRef.current = conversation;
  const { active, activeId, activeIdRef, setSessions, setMessages, setResponses, setError } =
    conversation;
  const { settings, tr } = useSettings();
  const runtimeContext = useRuntime();
  const { connectionRevision, ready, refreshRuntime, voiceComponent } = runtimeContext;
  const inference = useChatInference(active?.mode ?? 'text');
  const voice = useVoiceConversation(activeId, connectionRevision, setError);
  useEffect(() => {
    const controller = voice.voiceRef.current;
    if (controller && !controller.active)
      void controller.setFullDuplex(active?.mode === 'full_duplex');
  }, [active?.mode, connectionRevision, voice.voiceRef]);
  const attachments = useChatAttachments(activeId, setError);
  const [operationBusy, setBusy] = useState(false);
  const [voiceComponentBusy, setVoiceComponentBusy] = useState(false);
  const [mcpTools, setMcpTools] = useState<McpToolResource[]>([]);
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const revisionRef = useRef(connectionRevision);
  revisionRef.current = connectionRevision;
  const busy = operationBusy || isGenerationBusy(generationPhase);
  const recoveryNeeded = generation.getSnapshot().recoveryNeeded;
  useEffect(() => {
    generation.reset();
    setBusy(false);
  }, [activeId, connectionRevision, generation]);
  useEffect(() => {
    if (!visited || !ready) return;
    let current = true;
    const refresh = () => {
      void api
        .mcpTools()
        .then((result) => {
          if (current) setMcpTools(result.data);
        })
        .catch((cause) => {
          if (current) setError(errorMessage(cause));
        });
    };
    refresh();
    window.addEventListener('mfq:tools-changed', refresh);
    return () => {
      current = false;
      window.removeEventListener('mfq:tools-changed', refresh);
    };
  }, [visited, ready, connectionRevision, setError]);

  /** 为当前语音连接生成实时配置，使用解析后的模型默认设置。 */
  function realtimeSessionConfig(sessionId: string) {
    const value = inference.effectiveSettings;
    return {
      sessionId,
      systemPrompt: value.systemPrompt.trim(),
      temperature: value.temperature,
      topP: value.topP,
      topK: value.topK,
      repetitionPenalty: value.repetitionPenalty,
    };
  }

  /** 发起文本或工具结果生成，UI 快照与请求身份由独立控制器管理。 */
  async function generate(
    session: Session,
    input: ContentPart[],
    optimistic = true,
    role: 'user' | 'tool' = 'user',
  ) {
    if (
      activeIdRef.current !== session.id ||
      isGenerationBusy(generation.getPhase()) ||
      generation.getSnapshot().recoveryNeeded
    )
      return;
    setError(null);
    if (optimistic)
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'user',
          parts: input,
          parent_id: current.at(-1)?.id ?? null,
          created_at: new Date().toISOString(),
        },
      ]);
    await generation.start(session.id, {
      request_id: crypto.randomUUID(),
      expected_revision: session.revision,
      input,
      input_role: role,
      sampling: inference.sampling,
      system_prompt: inference.effectiveSettings.systemPrompt.trim(),
      include_reasoning_history: !inference.effectiveSettings.excludeReasoning,
      tools: mcpTools
        .filter((tool) => selectedTools.includes(tool.qualified_name))
        .map((tool) => ({
          type: 'function' as const,
          function: {
            name: tool.qualified_name,
            description: tool.description,
            parameters: tool.input_schema,
          },
        })),
      tool_choice: selectedTools.length ? 'auto' : 'none',
      stream: true,
    });
    void refreshRuntime();
  }

  /** 准备附件或语音输入后发送；服务切换后丢弃旧操作的返回，不清除新草稿。 */
  async function send(text: string, accepted: () => void) {
    if (!active || !conversation.conversationReady || busy || recoveryNeeded) return;
    const revision = connectionRevision;
    const isCurrent = () => activeIdRef.current === active.id && revisionRef.current === revision;
    setBusy(true);
    setError(null);
    const controller = voice.voiceRef.current;
    const resumeCapture = Boolean(controller?.capturing);
    try {
      if (
        active.mode !== 'text' &&
        inference.realtimeAvailable &&
        controller &&
        !attachments.attachments.length
      ) {
        if (!text) return;
        await controller.submitText(text, realtimeSessionConfig(active.id));
        if (!isCurrent()) return;
        accepted();
        voice.setVoiceMessages((current) => [
          ...current,
          {
            id: crypto.randomUUID(),
            sessionId: active.id,
            role: 'user',
            text,
            created_at: new Date().toISOString(),
          },
        ]);
        return;
      }
      if (active.mode !== 'text' && controller?.active) await controller.stop();
      const parts = await attachments.uploadAttachments();
      if (!isCurrent()) return;
      if (text) parts.push({ type: 'text', text });
      if (!parts.length) return;
      accepted();
      attachments.clearAttachments();
      setBusy(false);
      await generate(active, parts);
    } catch (cause) {
      if (isCurrent()) setError(errorMessage(cause));
    } finally {
      if (isCurrent()) {
        setBusy(false);
        if (resumeCapture && controller && inference.realtimeAvailable) {
          await controller
            .start(realtimeSessionConfig(active.id))
            .catch((cause) => setError(errorMessage(cause)));
        }
      }
    }
  }

  const messageActions = useMessageActions({
    conversation,
    blocked: busy || recoveryNeeded || !conversation.conversationReady,
    generate,
    setBusy,
  });

  /** 用户确认后用新会话替换旧会话，同时移除对应语音历史。 */
  async function clearActiveConversation() {
    if (
      !active ||
      busy ||
      recoveryNeeded ||
      !(await studioConfirm(tr('清空当前对话？', 'Clear this conversation?')))
    )
      return;
    setBusy(true);
    try {
      const replacement = await api.createSession(active.model, active.mode);
      await api.deleteSession(active.id);
      if (activeIdRef.current !== active.id) return;
      setSessions((current) => [
        replacement,
        ...current.filter((session) => session.id !== active.id),
      ]);
      voice.setVoiceMessages((current) =>
        current.filter((message) => message.sessionId !== active.id),
      );
      conversation.setActiveId(replacement.id);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  /** 切换会话交互模式前停止音频，成功后更新服务返回的会话版本。 */
  async function selectInteractionMode(mode: SessionMode) {
    if (!active || busy || active.mode === mode) return;
    setBusy(true);
    try {
      await voice.voiceRef.current?.stop();
      const updated = await api.updateSession(active.id, { mode });
      setSessions((current) =>
        current.map((session) => (session.id === updated.id ? updated : session)),
      );
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  /** 根据当前会话切换麦克风采集，异常交给聊天错误区域展示。 */
  async function toggleVoice() {
    if (!active || active.mode === 'text' || !inference.realtimeAvailable || busy) return;
    await voice.voiceRef.current
      ?.toggleCapture(realtimeSessionConfig(active.id))
      .catch((cause) => setError(errorMessage(cause)));
  }

  /** 显式下载或启用语音组件，提交后刷新共享任务状态。 */
  async function installOrEnableVoiceOutput() {
    if (voiceComponentBusy) return;
    setVoiceComponentBusy(true);
    try {
      if (voiceComponent?.ready) {
        const result = await api.activateVoiceOutputComponent();
        if (!result.active)
          throw new Error(result.error || result.reason || 'Voice output activation failed');
      } else await api.installVoiceOutputComponent();
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setVoiceComponentBusy(false);
    }
  }

  return {
    conversation,
    inference,
    voice,
    attachments,
    messageActions,
    generation,
    generationPhase,
    busy,
    recoveryNeeded,
    send,
    clearActiveConversation,
    selectInteractionMode,
    toggleVoice,
    voiceComponentBusy,
    installOrEnableVoiceOutput,
    mcpTools,
    selectedTools,
    setSelectedTools,
  };
}

const ChatContext = createContext<ReturnType<typeof useChatDomain> | null>(null);

/** 维持聊天领域实例，页面卸载不会取消正在进行的文本生成。 */
export function ChatProvider({ children }: { children: ReactNode }) {
  const value = useChatDomain();
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

/** 从聊天页面或工具栏读取聊天领域接口。 */
export function useChat() {
  const value = useContext(ChatContext);
  if (!value) throw new Error('ChatProvider is missing');
  return value;
}
