/** 验证编辑草稿在服务端确认回退后立即退出，失败时仍保留输入。 */
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { sessionsApi } from '../../../shared/api/resources/sessions';
import type { Message, Session } from '../../../shared/api/types';
import { useMessageActions } from './useMessageActions';
import type { useConversationSessions } from './useConversationSessions';

const session = { id: 'session-1', model: 'model-a', revision: 1 } as Session;
const message = {
  id: 'message-1',
  role: 'user',
  parts: [{ type: 'text', text: 'original' }],
} as Message;

afterEach(() => vi.restoreAllMocks());

it('回退成功时立即结束编辑，不等待生成流完成', async () => {
  vi.spyOn(sessionsApi, 'rewindSession').mockResolvedValue({ ...session, revision: 2 });
  const onCommitted = vi.fn();
  let finishGeneration!: () => void;
  const generate = vi.fn(
    () =>
      new Promise<void>((resolve) => {
        finishGeneration = resolve;
      }),
  );
  const conversation = {
    active: session,
    activeIdRef: { current: session.id },
    messages: [message],
    setMessages: vi.fn(),
    setSessions: vi.fn(),
    setResponses: vi.fn(),
    setError: vi.fn(),
  } as unknown as ReturnType<typeof useConversationSessions>;
  const { result } = renderHook(() =>
    useMessageActions({
      conversation,
      blocked: false,
      generate,
      setBusy: vi.fn(),
    }),
  );

  let pending!: Promise<boolean>;
  act(() => {
    pending = result.current.saveEdit(message, 'revised', onCommitted);
  });
  await waitFor(() => expect(onCommitted).toHaveBeenCalledOnce());
  expect(generate).toHaveBeenCalledOnce();
  await act(async () => {
    finishGeneration();
    expect(await pending).toBe(true);
  });
});

it('回退失败时不清除编辑草稿', async () => {
  vi.spyOn(sessionsApi, 'rewindSession').mockRejectedValue(new Error('rewind failed'));
  const onCommitted = vi.fn();
  const setError = vi.fn();
  const conversation = {
    active: session,
    activeIdRef: { current: session.id },
    messages: [message],
    setMessages: vi.fn(),
    setSessions: vi.fn(),
    setResponses: vi.fn(),
    setError,
  } as unknown as ReturnType<typeof useConversationSessions>;
  const { result } = renderHook(() =>
    useMessageActions({
      conversation,
      blocked: false,
      generate: vi.fn(),
      setBusy: vi.fn(),
    }),
  );

  await act(async () => {
    expect(await result.current.saveEdit(message, 'revised', onCommitted)).toBe(false);
  });
  expect(onCommitted).not.toHaveBeenCalled();
  expect(setError).toHaveBeenCalledWith('rewind failed');
});
