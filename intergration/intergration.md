# Tutti Intergration

Integration 层用于对接上层软件栈（vLLM/LLM runtime 等），向上管理应用暴露的设备内存，提供高性能裸设备 CPU/GPU 读写接口。

## v0.1 规范索引
- 人读规范: doc/Integration_API_Spec_v0.1.md
- AI 规范: docs-for-ai/Integration_API_Spec_v0.1_AI.md

v0.1 覆盖内容:
- 请求/响应模型
- 错误码语义
- 线程与并发语义
- 最小调用时序
- 验收清单

## 目标
- 提供稳定、可测试、可替换的上层 API 契约。
- 在不破坏 Tutti 现有行为的前提下，对接推理框架的数据装载与回写路径。
- 支持两种资源模式：
	- 进程内独立初始化模式。
	- 通过 NVMeService 获取共享队列资源的 attach 模式。

## 分层位置
- 上游调用方: 推理引擎、缓存管理器、KV 管理组件。
- Integration 层职责: API 适配、参数校验、错误语义归一化、异步完成回调。
- 下游依赖: Tutti Interface API + NVMeService Client API。

## 关键接口契约（草案）
- Init(config_path, device_id, options)
	- 职责: 初始化控制平面和数据平面上下文。
	- 输入: 配置路径、GPU 设备 ID、模式开关（独立或共享）。
	- 输出: context handle。
	- 错误语义: 返回结构化错误码（配置错误、设备不可用、权限不足、资源不足）。

- RegisterDeviceTensor(context, tensor_ptr, size, granularity)
	- 职责: 注册设备内存并建立 PRP 映射。
	- 并发语义: 线程安全；同一 tensor 多次注册幂等。

- ReadDeviceBlocks(context, file_ids, offsets, tensors, stream)
- WriteDeviceBlocks(context, file_ids, offsets, tensors, stream)
	- 职责: 高性能批量读写接口。
	- 并发语义: 支持多 stream 并发；同一 file segment 的写入冲突由调用方管理。
	- 性能注意事项: 建议按对齐粒度和批大小提交，避免小 IO 过多。

- AllocQueueLease(context, controller_index, queue_count)
- ReleaseQueueLease(context, lease_id)
	- 职责: 与 NVMeService 协同管理共享队列资源。
	- 错误语义: lease 过期、配额不足、服务不可达。

补充:
- 建议新增 LeaseHeartbeat(context, lease_id) 作为 shared 模式保活接口。
- Read/Write 接口默认为异步提交语义（提交成功不代表 IO 完成）。

## 当前状态
- 已有接口基础
	- Tutti 主 API 与 GPU/NVMe 数据通路已实现。
	- NVMeService 已支持 FsAllocQueues/FsReleaseQueues/LeaseHeartbeat。
- 待补齐
	- Integration 对外 API 的正式实现文件与示例。
	- Python 绑定层及对 vLLM 的最小可运行样例。

## 验证要求
- 功能验证
	- 单进程与多进程读写一致性。
	- 队列租约申请、续约、释放与超时回收。
- 性能验证
	- 吞吐、延迟、CPU 占用、GPU 利用率。
- 回归验证
	- 守护进程重启、进程异常退出、配置异常回退路径。
