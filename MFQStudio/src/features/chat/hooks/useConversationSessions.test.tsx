/** 验证会话模块的惰性加载、历史竞态隔离及生成期间模型切换保护。 */
import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { sessionsApi } from '../../../shared/api/resources/sessions';
import type { Session, Message } from '../../../shared/api/types';
import { useConversationStore } from '../state/conversationStore';
import { useConversationSessions } from './useConversationSessions';

const runtime = vi.hoisted(() => ({
  ready: true,
  connectionRevision: 1,
  selectedModel: 'model-a',
  setSelectedModel: vi.fn(),
  models: [{ id: 'model-a' }, { id: 'model-b' }],
  instances: [],
}));
vi.mock('../../../app/RuntimeProvider', () => ({ useRuntime: () => runtime }));
const first = { id: 'a', model: 'model-a', title: 'A', mode: 'text', revision: 0 } as Session;
const second = { ...first, id: 'b', title: 'B' };

beforeEach(() => {
  useConversationStore.getState().reset();
  runtime.ready = true;
  runtime.connectionRevision = 1;
  runtime.selectedModel = 'model-a';
  runtime.setSelectedModel.mockClear();
  vi.spyOn(sessionsApi, 'listSessions').mockResolvedValue([first, second]);
  vi.spyOn(sessionsApi, 'listMessages').mockResolvedValue([]);
  vi.spyOn(sessionsApi, 'listResponses').mockResolvedValue([]);
  vi.spyOn(sessionsApi, 'forkSession').mockResolvedValue({ ...first, id: 'fork', model: 'model-b' });
});

it('连接版本变化后丢弃旧列表请求并加载新连接的会话', async () => {
  let resolveOld!: (sessions: Session[]) => void;
  vi.mocked(sessionsApi.listSessions)
    .mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; }))
    .mockResolvedValueOnce([second]);
  const { result, rerender } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(sessionsApi.listSessions).toHaveBeenCalledOnce());
  runtime.connectionRevision = 2;
  rerender();
  await waitFor(() => expect(result.current.activeId).toBe('b'));
  await act(async () => resolveOld([first]));
  expect(result.current.sessions).toEqual([second]);
});

it('创建会话期间连接重置不会将旧创建结果写入新列表', async () => {
  let resolveCreate!: (session: Session) => void;
  vi.spyOn(sessionsApi, 'createSession').mockImplementationOnce(() =>
    new Promise((resolve) => { resolveCreate = resolve; }),
  );
  const { result, rerender } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  let creation!: Promise<void>;
  act(() => { creation = result.current.createSession(); });
  runtime.connectionRevision = 2;
  rerender();
  await waitFor(() => expect(sessionsApi.listSessions).toHaveBeenCalledTimes(2));
  await act(async () => { resolveCreate({ ...first, id: 'obsolete' }); await creation; });
  expect(result.current.sessions.some((session) => session.id === 'obsolete')).toBe(false);
});

it('未访问聊天不请求会话，访问后等待历史就绪才启用输入', async () => {
  const { result, rerender } = renderHook(
    ({ enabled }) => useConversationSessions(enabled, false),
    { initialProps: { enabled: false } },
  );
  expect(sessionsApi.listSessions).not.toHaveBeenCalled();
  expect(result.current.conversationReady).toBe(false);
  rerender({ enabled: true });
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  expect(sessionsApi.listSessions).toHaveBeenCalledOnce();
});

it('切换会话后迟到的旧历史不能覆盖当前消息', async () => {
  let resolveOld!: (messages: Message[]) => void;
  vi.mocked(sessionsApi.listMessages).mockImplementation((id) =>
    id === 'a'
      ? new Promise((resolve) => {
          resolveOld = resolve;
        })
      : Promise.resolve([
          { id: 'b-message', role: 'user', parts: [], parent_id: null, created_at: '' },
        ]),
  );
  const { result } = renderHook(() => useConversationSessions(true, false));
  await waitFor(() => expect(result.current.activeId).toBe('a'));
  await act(async () => result.current.selectSession('b'));
  await waitFor(() => expect(result.current.messages[0]?.id).toBe('b-message'));
  await act(async () =>
    resolveOld([{ id: 'a-message', role: 'user', parts: [], parent_id: null, created_at: '' }]),
  );
  expect(result.current.activeId).toBe('b');
  expect(result.current.messages[0]?.id).toBe('b-message');
});

it('后台生成期间不派生新模型会话，完成后再执行模型切换', async () => {
  const { result, rerender } = renderHook(({ busy }) => useConversationSessions(true, busy), {
    initialProps: { busy: true },
  });
  await waitFor(() => expect(result.current.activeId).toBe('a'));
  runtime.selectedModel = 'model-b';
  rerender({ busy: true });
  expect(sessionsApi.forkSession).not.toHaveBeenCalled();
  rerender({ busy: false });
  await waitFor(() => expect(result.current.activeId).toBe('fork'));
  expect(sessionsApi.forkSession).toHaveBeenCalledOnce();
});
