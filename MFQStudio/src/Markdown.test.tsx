/** 验证流式 Markdown 限频、最终内容刷新与不可信 HTML 净化。 */
import { act, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Markdown } from './Markdown';

afterEach(() => vi.useRealTimers());

describe('Markdown', () => {
  it('持续输入时按时间窗解析最新文本，结束时无需等待窗口', async () => {
    vi.useFakeTimers();
    const view = render(<Markdown text="first" live />);
    view.rerender(<Markdown text="second" live />);
    await act(() => vi.advanceTimersByTimeAsync(40));
    view.rerender(<Markdown text="third" live />);
    expect(screen.getByText('first')).toBeInTheDocument();
    await act(() => vi.advanceTimersByTimeAsync(40));
    expect(screen.getByText('third')).toBeInTheDocument();
    view.rerender(<Markdown text="final" />);
    expect(screen.getByText('final')).toBeInTheDocument();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('完整和流式输出都净化事件属性、脚本与危险链接', async () => {
    const text =
      '<img src="x" onerror="alert(1)"><script>alert(1)</script>[link](javascript:alert(1))';
    const view = render(<Markdown text={text} live />);
    expect(view.container.querySelector('script')).toBeNull();
    expect(view.container.querySelector('[onerror]')).toBeNull();
    expect(view.container.querySelector('a[href^="javascript:"]')).toBeNull();
    view.rerender(<Markdown text={text} />);
    expect(view.container.querySelector('[onerror]')).toBeNull();
  });

  it('未闭合代码块完成后有一个复制按钮，卸载时清理限频任务', async () => {
    vi.useFakeTimers();
    const view = render(<Markdown text={'```js\nconst a = 1;'} live />);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
    view.rerender(<Markdown text={'```js\nconst a = 1;\n```'} />);
    expect(screen.getAllByRole('button', { name: 'Copy' })).toHaveLength(1);
    view.rerender(<Markdown text="next" live />);
    view.unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});
