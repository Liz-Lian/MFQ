# MFQ Studio 前端

Web 与 Tauri 共用 React 前端。开发使用 pnpm，现有桌面打包的 npm 入口保持兼容；依赖变化后同时更新两份锁文件。

## 本地运行

```sh
pnpm install --frozen-lockfile
pnpm dev
```

Vite 默认监听 `127.0.0.1:5173`，将 `/api` 代理到 `127.0.0.1:8090`。实际推理需要启动 MFQ 服务；浏览器测试使用模拟接口，不需要模型、GPU 或凭据。页面使用 Hash 路由，例如 `/#/chat`，兼容桌面资源协议。

## 模块边界

| 目录 | 职责 |
| --- | --- |
| `src/app` | 应用展示、格式化、可折叠面板 |
| `src/features/chat` | 消息、附件、输入组件、生成生命周期与滚动 |
| `src/features/voice` | 语音控制器、PCM 编解码、连续重采样、片段存储 |
| `src/features/settings` | 设置页面、推理配置、预设转换 |
| `src/features/models`、`runtime`、`jobs` | 模型引用解析、实例选择、指标与任务表单规则 |
| `src/shared/api` | HTTP、SSE、协议校验、类型与八类资源 API |
| `src/shared/ui` | Dialog、Tooltip、Switch 的应用级 Radix 封装 |
| `src/shared/platform` | Web/Tauri 桥接 |

`api.ts`、`studio.ts`、`realtimeAudio.ts` 保留导入兼容入口。新业务优先使用领域模块；共享 API 层不能反向依赖页面或 React 状态。`App.tsx` 仍负责跨业务编排，后续迁移页面时应连同对应请求与状态一起移动，避免制造仅包装大型 hook 的新入口。

局部输入草稿按会话保存于内存 Zustand store，不写入浏览器持久化存储。根组件只订阅生成阶段，`StreamingMessage` 订阅增量快照，历史正文通过 memo 和稳定业务回调隔离。

## 生成生命周期

```text
submitting -> streaming -> syncing -> completed
                    |         |
                 stopping     failed -> 重新同步
                    |
                 syncing -> cancelled
```

- 每个请求保留自己的会话与请求标识；重置后旧流、旧同步和旧取消回调失效。
- SSE 校验版本、会话、连续序号、响应标识和业务终态。正常 EOF 不等于生成完成。
- 文本增量按 32ms 窗口批量提交，结束时强制刷新；Markdown 在流式阶段按 80ms 窗口解析，完成后立即刷新并补充公式、复制按钮。所有 HTML 仍经过 DOMPurify。
- 历史同步完成前保留临时回答并锁定发送；同步失败后保留收到的文本，恢复按钮只读取历史，不重放生成 POST。
- 取消请求最多等待 5 秒，历史同步最多等待 10 秒。无法确认服务端停止时保持恢复状态，防止产生重复生成。
- 滚动跟随由内容尺寸变化驱动，用户向上阅读后暂停。输入组件处理中文输入法组合状态，移动端会话侧栏默认关闭。

## 验证

```sh
pnpm test
pnpm typecheck
pnpm build
pnpm exec playwright install chromium --only-shell
pnpm test:e2e
```

从仓库根目录运行：

```sh
python -m pytest tests/test_mfq_studio_source.py -q
```

单元测试覆盖分块 SSE、协议错误、取消、历史恢复、旧请求隔离、草稿渲染边界、Markdown 净化以及音频分块等价性。浏览器测试覆盖桌面和移动端发送、取消、恢复、输入法及弹窗键盘焦点，截图与失败追踪输出到被忽略的 `artifacts/playwright`。

这些检查不替代真实模型推理、麦克风、扬声器及 Tauri 原生窗口联调。没有将合成事件测试描述为真实模型性能提升；长历史虚拟化及稳定 Markdown 分块缓存应在真实长会话压测证明必要后加入。
