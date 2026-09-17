# TODO

- [ ] 统一模型输入接口：Metal 仍通过 `MfqContainer` 兼容层加载，而 CUDA 已直接消费 `ModelSource`。
- [x] 收敛模型路由：Server 不再读取 `ModelGraph` 或按架构分流，统一由推理 runtime 判断具体实现。
- [ ] 简化 CUDA 模型构造：配置解析、顶层权重和层构造已迁入 `backends/cuda/models/<architecture>/`；继续拆分 `Model` 中架构专用的 forward/cache 状态。
- [ ] 改进加载进度：Web 当前显示的是 Job 健康检查进度，不是实际权重或层加载进度。
