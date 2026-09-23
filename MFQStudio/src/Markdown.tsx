/** 渲染经过净化的 Markdown，流式阶段限频解析，完成后补充公式与复制操作。 */
import DOMPurify from 'dompurify';
import renderMathInElement from 'katex/contrib/auto-render';
import { marked } from 'marked';
import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { normalizeEscapedMarkdownLineBreaks } from './markdownText';
import 'katex/dist/katex.min.css';

interface MarkdownProps {
  text: string;
  live?: boolean;
  normalizeEscapedLineBreaks?: boolean;
}

/** 限制长回答的全量 Markdown 解析频率，并在结束生成时立即展示最终文本。 */
function useStreamingText(text: string, live: boolean): string {
  const [displayed, setDisplayed] = useState(text);
  const latest = useRef(text);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  latest.current = text;
  useEffect(() => {
    if (!live) {
      if (timer.current !== null) clearTimeout(timer.current);
      timer.current = null;
      setDisplayed(text);
    } else if (timer.current === null && text !== displayed) {
      timer.current = setTimeout(() => {
        timer.current = null;
        setDisplayed(latest.current);
      }, 80);
    }
  }, [text, live, displayed]);
  useEffect(
    () => () => {
      if (timer.current !== null) clearTimeout(timer.current);
    },
    [],
  );
  return live ? displayed : text;
}

/** 将模型文本安全渲染为富文本；历史消息使用 memo 避免无关增量更新重复执行。 */
export const Markdown = memo(function Markdown({
  text,
  live = false,
  normalizeEscapedLineBreaks = false,
}: MarkdownProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const displayedText = useStreamingText(text, live);
  const markdown = useMemo(
    () =>
      normalizeEscapedLineBreaks
        ? normalizeEscapedMarkdownLineBreaks(displayedText)
        : displayedText,
    [normalizeEscapedLineBreaks, displayedText],
  );
  const html = useMemo(
    () =>
      DOMPurify.sanitize(
        marked.parse(markdown, {
          breaks: true,
          gfm: true,
        }) as string,
      ),
    [markdown],
  );
  // 保持相同 HTML 的对象引用，避免限频状态更新覆盖公式和复制按钮的 DOM 增强。
  const markup = useMemo(() => ({ __html: html }), [html]);

  useEffect(() => {
    if (live || !rootRef.current) return;
    renderMathInElement(rootRef.current, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false },
        { left: '$', right: '$', display: false },
      ],
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
      strict: 'ignore',
      throwOnError: false,
      trust: false,
    });
    const buttons: HTMLButtonElement[] = [];
    const timers = new Set<ReturnType<typeof setTimeout>>();
    let disposed = false;
    for (const block of rootRef.current.querySelectorAll('pre')) {
      const source = block.querySelector('code')?.textContent || block.textContent || '';
      const button = document.createElement('button');
      button.className = 'code-copy';
      button.type = 'button';
      button.textContent = 'Copy';
      button.addEventListener('click', () => {
        void Promise.resolve()
          .then(() => navigator.clipboard.writeText(source))
          .then(() => {
            if (disposed) return;
            button.textContent = 'Copied';
            const timer = setTimeout(() => {
              button.textContent = 'Copy';
              timers.delete(timer);
            }, 1200);
            timers.add(timer);
          })
          .catch(() => {
            if (!disposed) button.textContent = 'Copy failed';
          });
      });
      block.append(button);
      buttons.push(button);
    }
    return () => {
      disposed = true;
      timers.forEach(clearTimeout);
      buttons.forEach((button) => button.remove());
    };
  }, [html, live]);

  return <div className="rich-text" dangerouslySetInnerHTML={markup} ref={rootRef} />;
});
