/** 管理会话列表、选中会话与消息历史，所有异步回写绑定当前连接和会话。 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  api,
  type Session,
  type Message,
  type ResponseResource,
  type SessionMode,
} from '../../../api';
import { useRuntime } from '../../../app/RuntimeProvider';
import { errorMessage } from '../../../app/formatters';

/** 首次打开聊天才加载会话，切换时取消旧历史请求，跨页面保留已加载状态。 */
export function useConversationSessions(enabled: boolean, generationBusy: boolean) {
  const { ready, connectionRevision, selectedModel, setSelectedModel, models, instances } =
    useRuntime();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [responses, setResponses] = useState<Record<string, ResponseResource>>({});
  const [historyLoadedId, setHistoryLoadedId] = useState<string | null>(null);
  const [transitioning, setTransitioning] = useState(false);
  const [importRevision, setImportRevision] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const version = useRef(0);
  const activeIdRef = useRef(activeId);
  activeIdRef.current = activeId;
  const active = useMemo(
    () => sessions.find((session) => session.id === activeId) ?? null,
    [sessions, activeId],
  );
  const modelAvailable =
    models.some((model) => model.id === selectedModel) ||
    instances.some(
      (instance) => instance.model === selectedModel && ['ready', 'busy'].includes(instance.state),
    );
  useEffect(() => {
    const refresh = () => setImportRevision((current) => current + 1);
    window.addEventListener('mfq:sessions-imported', refresh);
    return () => window.removeEventListener('mfq:sessions-imported', refresh);
  }, []);

  useEffect(() => {
    const request = ++version.current;
    setSessions([]);
    setActiveId(null);
    setMessages([]);
    setResponses({});
    setHistoryLoadedId(null);
    setError(null);
    if (!ready || !enabled) return;
    void api
      .listSessions()
      .then((next) => {
        if (request !== version.current) return;
        setSessions(next);
        const selected = next[0];
        setActiveId(selected?.id ?? null);
        if (selected) setSelectedModel(selected.model);
      })
      .catch((cause) => {
        if (request === version.current) setError(errorMessage(cause));
      });
    return () => {
      ++version.current;
    };
  }, [ready, enabled, connectionRevision, importRevision, setSelectedModel]);

  useEffect(() => {
    setHistoryLoadedId(null);
    setMessages([]);
    setResponses({});
    if (!activeId) return;
    const controller = new AbortController();
    void Promise.all([
      api.listMessages(activeId, controller.signal),
      api.listResponses(activeId, controller.signal),
    ])
      .then(([nextMessages, nextResponses]) => {
        if (controller.signal.aborted) return;
        setMessages(nextMessages);
        setResponses(
          Object.fromEntries(
            nextResponses
              .filter((response) => response.output_message_id)
              .map((response) => [response.output_message_id!, response]),
          ),
        );
        setHistoryLoadedId(activeId);
      })
      .catch((cause) => {
        if (!controller.signal.aborted) setError(errorMessage(cause));
      });
    return () => controller.abort();
  }, [activeId, connectionRevision]);

  useEffect(() => {
    if (!active || generationBusy || !modelAvailable || active.model === selectedModel) return;
    let current = true;
    setTransitioning(true);
    void api
      .forkSession(active.id, null, true, active.title, selectedModel)
      .then((replacement) => {
        if (!current) return;
        setSessions((existing) => [replacement, ...existing]);
        setActiveId(replacement.id);
      })
      .catch((cause) => {
        if (current) setError(errorMessage(cause));
      })
      .finally(() => {
        if (current) setTransitioning(false);
      });
    return () => {
      current = false;
      setTransitioning(false);
    };
  }, [active?.id, active?.model, active?.title, selectedModel, modelAvailable, generationBusy]);

  /** 切换会话及其绑定模型，历史请求在 effect 中按会话重建。 */
  const selectSession = useCallback(
    (id: string) => {
      const session = sessions.find((candidate) => candidate.id === id);
      if (!session || transitioning) return;
      setSelectedModel(session.model);
      setActiveId(id);
    },
    [sessions, transitioning, setSelectedModel],
  );

  /** 创建新的空会话，版本戳防止服务切换后的返回污染当前列表。 */
  const createSession = useCallback(
    async (mode: SessionMode = 'text') => {
      if (!selectedModel || transitioning) return;
      const request = version.current;
      setTransitioning(true);
      try {
        const created = await api.createSession(selectedModel, mode);
        if (request !== version.current) return;
        setSessions((current) => [created, ...current]);
        setActiveId(created.id);
      } catch (cause) {
        if (request === version.current) setError(errorMessage(cause));
      } finally {
        if (request === version.current) setTransitioning(false);
      }
    },
    [selectedModel, transitioning],
  );

  return {
    sessions,
    setSessions,
    active,
    activeId,
    setActiveId,
    activeIdRef,
    messages,
    setMessages,
    responses,
    setResponses,
    transitioning,
    error,
    setError,
    selectSession,
    createSession,
    conversationReady: Boolean(
      active && modelAvailable && active.model === selectedModel && historyLoadedId === activeId,
    ),
    modelAvailable,
  };
}
