# Studio 前端测试

在 `MFQStudio` 目录运行：

```sh
pnpm install --frozen-lockfile
pnpm test
pnpm typecheck
pnpm build
pnpm exec playwright install chromium --only-shell
pnpm test:e2e
```

`pnpm test:watch` 用于本地持续运行。测试使用 Vitest、jsdom 与 Testing Library；测试文件既可放在业务模块旁，也可放在本目录。统一配置自动清理 DOM、浏览器存储和全局桩对象。

## 覆盖边界

- `navigation.test.ts`：深链接、路由生成、尾部斜杠及未知地址回退。
- `markdownText.test.ts`：转义换行恢复与代码、JSON、数学文本保护。
- `eventStream.test.ts`：SSE 字节分片、换行、畸形 JSON、截断与取消清理。
- `streamResponse.test.ts`：会话、序号、响应标识、业务终态和错误传播。
- `src` 内的组件及状态测试：由对应业务模块维护，按用户可观察行为断言。

从仓库根目录执行 `python -m pytest tests/test_mfq_studio_source.py -q`，校验桌面打包及平台契约。这组历史测试仍包含源码断言，模块拆分时应更新真实实现路径；新交互优先用行为测试覆盖。

CI 同时运行 `test_webui_*`、对话模板、MiniCPM 语音和模型能力中引用前端源码的相关契约。共享读取器 `tests/studio_sources.py` 只收集实现文件，不把测试文本误计为实现。用户消息编辑与助手消息重新生成沿用本次界面基线的交互。

生产构建不纳入测试文件。`pnpm typecheck` 另外检查测试及测试配置，避免遗漏断言类型错误。CI 同时执行行为测试、类型检查、构建、桌面与移动端浏览器回归及 Python 契约测试。

项目保留桌面打包使用的 npm 入口。开发依赖变动时需要同步 `pnpm-lock.yaml` 与 `package-lock.json`，不手工修改锁文件。
