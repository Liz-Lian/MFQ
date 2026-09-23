/** 管理历史消息编辑、重新生成和确认后的工具执行，不持有全局运行时状态。 */
import { useState } from 'react';
import { api, type ContentPart, type Message, type Session } from '../../../api';
import { textParts, isMediaPart } from '../messageParts';
import type { useConversationSessions } from './useConversationSessions';
import type { EditDraft } from '../SavedMessageList';
import { errorMessage } from '../../../app/formatters';

interface MessageActionOptions {
  conversation: ReturnType<typeof useConversationSessions>;
  blocked: boolean;
  /** 根据回退后的会话版本继续生成；工具输入不得追加乐观用户消息。 */
  generate: (
    session: Session,
    input: ContentPart[],
    optimistic?: boolean,
    role?: 'user' | 'tool',
  ) => Promise<void>;
  /** 在历史写入期间锁定聊天操作，结束后由本 hook 释放。 */
  setBusy: (busy: boolean) => void;
}

/** 向消息列表提供明确的业务动作，并通过活动会话标识拒绝过期回写。 */
export function useMessageActions({
  conversation,
  blocked,
  generate,
  setBusy,
}: MessageActionOptions) {
  const [editDraft, setEditDraft] = useState<EditDraft | null>(null);
  const { active, activeIdRef, messages, setMessages, setSessions, setResponses, setError } =
    conversation;
  const current = (sessionId: string) => activeIdRef.current === sessionId;

  /** 编辑用户消息后回退历史，保留附件并继续生成。 */
  async function saveEdit(message: Message) {
    if (!active || !editDraft || blocked) return;
    const index = messages.findIndex((item) => item.id === message.id);
    const text = editDraft.text.trim();
    if (index < 0 || (!text && !message.parts.some(isMediaPart))) return;
    setBusy(true);
    try {
      const rewound = await api.rewindSession(active.id, active.revision, message.id, false);
      if (!current(active.id)) return;
      const parts: ContentPart[] = [
        ...(text ? [{ type: 'text' as const, text }] : []),
        ...message.parts.filter((part) => isMediaPart(part) || part.type === 'document'),
      ];
      const previous = messages.slice(0, index);
      setSessions((items) =>
        items.map((session) => (session.id === rewound.id ? rewound : session)),
      );
      setMessages([...previous, { ...message, parts }]);
      setResponses((items) =>
        Object.fromEntries(
          Object.entries(items).filter(([id]) => previous.some((item) => item.id === id)),
        ),
      );
      setEditDraft(null);
      await generate(rewound, parts, false);
    } catch (cause) {
      if (current(active.id)) setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  /** 从助手回答前的用户输入重新生成，删除后续历史并使用服务返回的版本号。 */
  async function regenerate(message: Message) {
    if (!active || blocked || message.role !== 'assistant') return;
    const index = messages.findIndex((item) => item.id === message.id);
    const user = messages
      .slice(0, index)
      .reverse()
      .find((item) => item.role === 'user');
    if (!user?.parts.length) return;
    setBusy(true);
    try {
      const rewound = await api.rewindSession(active.id, active.revision, user.id, false);
      if (!current(active.id)) return;
      const previous = messages.slice(0, messages.indexOf(user) + 1);
      setSessions((items) =>
        items.map((session) => (session.id === rewound.id ? rewound : session)),
      );
      setMessages(previous);
      setResponses((items) =>
        Object.fromEntries(
          Object.entries(items).filter(([id]) => previous.some((item) => item.id === id)),
        ),
      );
      await generate(rewound, user.parts, false);
    } catch (cause) {
      if (current(active.id)) setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  /** 复制消息正文与推理内容，失败时保留页面并展示可读错误。 */
  async function copyMessage(message: Message) {
    try {
      const parts = textParts(message);
      await navigator.clipboard.writeText(
        [parts.reasoning, parts.text].filter(Boolean).join('\n\n'),
      );
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  /** 仅在用户点击确认后执行工具，按会话版本提交工具结果并继续生成。 */
  async function executeToolCalls(message: Message) {
    if (!active || blocked) return;
    const calls = message.parts.filter(
      (part): part is Extract<ContentPart, { type: 'tool_call' }> => part.type === 'tool_call',
    );
    if (!calls.length) return;
    setBusy(true);
    try {
      const results = await Promise.all(
        calls.map(async (call) => ({
          call,
          result: await api.callMcpTool(call.name, call.arguments),
        })),
      );
      if (!current(active.id)) return;
      let updated = await api.getSession(active.id);
      const parts = results.map(({ call, result }): ContentPart => ({
        type: 'tool_result',
        call_id: call.call_id,
        result: result.structured_content ?? result.content,
        is_error: result.is_error,
      }));
      for (const part of parts.slice(0, -1)) {
        updated = (await api.appendMessage(updated.id, updated.revision, 'tool', [part])).session;
        if (!current(active.id)) return;
      }
      await generate(updated, [parts.at(-1)!], false, 'tool');
    } catch (cause) {
      if (current(active.id)) setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  return { editDraft, setEditDraft, saveEdit, regenerate, copyMessage, executeToolCalls };
}
