/** 验证会话模块的惰性加载、历史竞态隔离及生成期间模型切换保护。 */
import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { api, type Session, type Message } from '../../../api';
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
  runtime.ready = true;
  runtime.connectionRevision = 1;
  runtime.selectedModel = 'model-a';
  runtime.setSelectedModel.mockClear();
  vi.spyOn(api, 'listSessions').mockResolvedValue([first, second]);
  vi.spyOn(api, 'listMessages').mockResolvedValue([]);
  vi.spyOn(api, 'listResponses').mockResolvedValue([]);
  vi.spyOn(api, 'forkSession').mockResolvedValue({ ...first, id: 'fork', model: 'model-b' });
});

it('未访问聊天不请求会话，访问后等待历史就绪才启用输入', async () => {
  const { result, rerender } = renderHook(
    ({ enabled }) => useConversationSessions(enabled, false),
    { initialProps: { enabled: false } },
  );
  expect(api.listSessions).not.toHaveBeenCalled();
  expect(result.current.conversationReady).toBe(false);
  rerender({ enabled: true });
  await waitFor(() => expect(result.current.conversationReady).toBe(true));
  expect(api.listSessions).toHaveBeenCalledOnce();
});

it('切换会话后迟到的旧历史不能覆盖当前消息', async () => {
  let resolveOld!: (messages: Message[]) => void;
  vi.mocked(api.listMessages).mockImplementation((id) =>
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
  expect(api.forkSession).not.toHaveBeenCalled();
  rerender({ busy: false });
  await waitFor(() => expect(result.current.activeId).toBe('fork'));
  expect(api.forkSession).toHaveBeenCalledOnce();
});
