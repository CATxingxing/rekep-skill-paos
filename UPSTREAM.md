# 上游参考实现

本工程模仿、但不照搬以下参考实现：

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/baiyu858/PhyAgentOS-rekep-real-plugin` |
| 形态 | 外部 HAL 插件 + 独立 ReKep 运行时（仿真线 + 真机线） |
| pinned commit | `c532a28`（2026-04-02） |
| 本地子模块 | `reference/rekep-real-plugin` |
| 许可证 | MIT（见子模块 `LICENSE`） |

## 复用范围（原样引用，不改）

仿真线算法模块位于 `reference/rekep-real-plugin/runtime/`：

- `keypoint_proposal.py` — DINOv2 关键点提议
- `constraint_generation.py` + `vlm_query/prompt_template.txt` — VLM 生成 ReKep 约束
- `subgoal_solver.py` / `path_solver.py` / `ik_solver.py` — 分层优化与 IK
- `utils.py` / `transform_utils.py` — 变换、插值、bounds 等
- `environment.py` / `og_scene_file_pen.json` — OmniGibson 仿真 env（本工程以接口隔离，后续可换回）

## 不采用的集成方式

参考实现的 HAL 插件集成（`PhyAgentOS_plugin.toml` + `driver.py` + `hal_watchdog` + `ACTION.md`）是 Forge 之前的旧协议，本工程**不采用**，改为标准 Forge Skill（manifest-v2 + Gateway Tool API + Dora Node）。
