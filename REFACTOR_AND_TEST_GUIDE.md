# ReKep PAOS Skill 重构、运行逻辑与测试指南

## 1. 重构结果

本仓库已从“源码目录、参考仓库、运行环境和生成物相互耦合”的实验工程，重构为可定位来源、可独立构建、可校验归档的 PAOS Skill。核心边界如下：

- `skill-src/rekep/` 严格只保留 `SKILL.md` 与 `skill.yaml`。
- `profiles/`、`assets/`、`nodes/`、`locks/`、`scripts/` 与 `tests/` 分层存放。
- 仿真 profile 只有一个外部 `mujoco_sim`，三个 ReKep 节点都不创建第二个模拟器。
- 感知结果、约束程序和执行结果使用版本化 JSON 契约；执行前必须核对 `session_id`、`scene_revision`、`observation_id`、`plan_id` 与 `plan_digest`。
- VLM 只能产生受限 JSON AST，不接受 Python 代码、模板规划或任意关节命令。
- 执行器一次只允许一个下游 Action，上一段必须返回终态 `succeeded/SUCCESS` 才会发送下一段。
- 运行记录按内容 ID 不可变保存；不存在 `latest` 文件作为节点间通信通道。
- 旧 `reference/`、旧源码树、旧内嵌 profile、证据输出和已提交构建产物已从目标结构移除。

## 2. 已有 Forge Node 的复用结论

已先核对以下官方源：

- GitLab：<https://gitlab.ex-ai.cn/PhyAgentOS/framework>。当前网页抓取不可达，匿名 Git 请求要求凭据，因此它只作为有凭据时的候选镜像，不能作为默认无人值守构建源。
- GitHub：<https://github.com/Forgelab-Robotics>。官方组织明确提供可复用的 Gateway、MuJoCo、Motion 和 Controls 仓库。

本 Skill 复用 5 个已有节点，不重复实现或注册它们：

| 节点 | 官方仓库 | 锁定提交 | 源码版本 |
|---|---|---|---|
| `gateway` | `adapter-forge-gateway` | `49caba94d641aff4664fbdf9d4c8b1402add1ce6` | 1.1.0 |
| `mujoco_sim` | `operator-simulator-mujoco` | `0a84a1a81807b467338e696032a38b084cb35747` | 1.1.1 |
| `motion_server` | `operator-motion` | `24c8a430715f697eea8431bf169c21b8cbb5eb92` | 0.1.0 |
| `joint_trajectory_controller` | `operator-controls` | `cde73dc2a6222d81b01ffc32f69f12cb12e42760` | 0.1.0 |
| `gripper_action_controller` | `operator-controls` | `cde73dc2a6222d81b01ffc32f69f12cb12e42760` | 0.1.0 |

三个自研节点 `rekep_perception`、`rekep_planner`、`rekep_executor` 保留，因为官方通用节点不包含本 Skill 特有的 DINOv2 关键点提取、观测绑定约束生成和 ReKep 约束求解逻辑。

## 3. glibc 兼容重打包方案

不要直接复用在更高 glibc 环境中生成的可执行文件。仓库改为“拉取锁定源码 → 使用官方构建脚本 → 检查 ELF 所需 GLIBC 符号 → 生成单入口确定性 tar.gz → 写入本机构建锁”。

官方四个仓库均提供 `scripts/build_pyinstaller.sh`，并提供基于 `ubuntu:20.04` 的构建 workflow；Ubuntu 20.04 的 glibc 基线是 2.31，低于本服务器的 2.35。

推荐构建方式：

```bash
# 1. 拉取并 checkout 到 locks/nodes.lock.json 的精确提交
uv run python scripts/fetch_nodes.py

# 2. 推荐：复现官方 ubuntu:20.04 基线，并拒绝高于 GLIBC_2.31 的产物
uv run python scripts/rebuild_reused_nodes.py \
  --mode container \
  --container-image ubuntu:20.04 \
  --max-glibc 2.31
```

如果 Docker 不可用，可直接在本服务器构建；这保证产物不会要求高于服务器的 glibc 2.35：

```bash
uv run python scripts/rebuild_reused_nodes.py \
  --mode local \
  --max-glibc 2.35
```

输出为：

```text
dist/nodes/<artifact-id>.tar.gz
.build/reused-nodes.lock.json
```

每个 tar.gz 只能包含一个位于归档根目录、名称与 `entrypoint` 相同且可执行的文件。生成锁记录源码 URL、提交、版本、SHA-256、构建方式、构建镜像、构建环境 glibc 基线和外层 PyInstaller bootloader 所需 GLIBC 符号。one-file 内嵌 `.so` 被压缩，不能仅凭外层 `readelf` 推断其完整需求；兼容性门禁因此以“构建环境不高于目标环境 + 目标环境独立启动”共同判定。`package_skill.py` 只接受该生成锁，不再信任下载来的旧二进制摘要。

可单独审计 ELF 版本：

```bash
readelf --version-info .build/upstream-sources/gateway/dist/gateway \
  | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1
```

## 4. 运行数据流

```text
PAOS Agent
   │ Tool invoke/status/result/control
   ▼
gateway
   ├── rekep.get_context ──► rekep_perception
   ├── rekep.plan_task   ──► rekep_planner
   └── rekep.execute_task──► rekep_executor
                                  │ 串行 MovePose/Gripper Action
                                  ▼
                          motion_server + controllers
                                  │ JointCommand
                                  ▼
                              mujoco_sim
                                  │ RGB + depth + JointState
                                  └────────► rekep_perception
```

标准调用顺序：

1. `rekep.get_context` 取得不可变 `PerceptionSnapshot`。
2. `rekep.plan_task` 使用同一 `observation_id` 请求当前 PAOS VLM，产生受限 `ConstraintProgram`。
3. 调用方显式回传完整 ID、digest 和 `allow_motion=true` 给 `rekep.execute_task`。
4. 执行器重新验证绑定关系、求解子目标与路径，然后逐段调用标准 Motion/Gripper Action。
5. 每个 stage 后请求新观测并复验约束；任何过期、篡改、超时、下游失败或约束超差都立即失败关闭。

## 5. 各文件运行逻辑

### Skill 与 profile

| 文件 | 逻辑 |
|---|---|
| `skill-src/rekep/SKILL.md` | 约束 Agent 必须按 context → plan → execute 顺序调用，禁止猜测 ID、自动重试、模板规划和 Python 执行。 |
| `skill-src/rekep/skill.yaml` | 声明 skill v2、仿真 profile、8 个必需二进制和机器人资产；构建时才注入最终 artifact locks。 |
| `profiles/sim-dobot-nova2-robotiq/dataflow.yaml` | 连接 8 个节点；只声明一个 `mujoco_sim`；RGB-D/state 直接进入 perception，执行动作只经过标准 motion/controller 链。 |
| `profiles/sim-dobot-nova2-robotiq/gateway.yaml` | 定义 3 个 Tool 的参数 schema、路由、超时和只监听本机的 Gateway。 |
| `perception.yaml` | 配置相机、深度、标定、分割、DINOv2 模型配置路径及快照目录。 |
| `planner.yaml` | 配置提示模板、约束输出目录和最大 stage/constraint 数。 |
| `executor.yaml` | 配置工作空间、默认末端位姿、路径采样、clearance 和子 Action deadline。 |
| `mujoco-simulator.yaml` | 配置唯一 MuJoCo 场景、关节映射、RGB/深度输出与初始状态。 |
| `motion-server.yaml` | 配置机械臂组、运动学、反馈 freshness 和轨迹 Action。 |
| `joint-trajectory-controller.yaml` | 把标准轨迹 Action 转成位置 `JointCommand`，监测关节反馈与终态误差。 |
| `gripper-action-controller.yaml` | 把 Gripper Action 转成夹爪位置命令并返回标准终态。 |

### 公共契约

| 文件 | 逻辑 |
|---|---|
| `nodes/common/rekep_core/ids.py` | 规范 JSON 序列化、SHA-256 内容 ID 与纳秒时间戳。 |
| `geometry.py` | 有限向量检查、距离和坐标变换。 |
| `contracts.py` | 定义 snapshot/program/result v3 契约，校验受限 AST、引用、grasp/release 配平、内容摘要和跨阶段绑定；提供原子 JSON 持久化。 |
| `provider.py` | 实现 Forge Tool Query/Action provider，维护单并发、progress、status、result、cancel 和 Dora 输出队列。 |

### 感知节点

| 文件 | 逻辑 |
|---|---|
| `nodes/rekep-perception/node_main.py` | 接收 RGB、depth、JointState 和 sim status；处理 `get_context` 与执行器的刷新观测请求；发布 `snapshot_out`。 |
| `perception.py` | 维护 episode/session/revision，做颜色连通域或配置分割、深度反投影、稳定 object track，生成 immutable snapshot 与 overlay。 |
| `dinov2.py` | 从锁定本地源码和权重加载 DINOv2，提取 feature map，用有界且有限值检查的 k-means 生成关键点。 |

### 规划节点

| 文件 | 逻辑 |
|---|---|
| `nodes/rekep-planner/node_main.py` | 缓存 Dora snapshot；只允许按精确 observation ID 读取不可变恢复记录；调用 VLM 后校验、持久化并发布约束程序。 |
| `vlm_client.py` | 读取权限 0600 的当前 PAOS provider 配置，用 OpenAI-compatible multimodal chat-completions 发送 overlay 和精确观测；不支持时返回结构化错误。 |
| `constraint_schema.py` | 将 VLM stages 绑定到 session/revision/observation，计算 `plan_id` 与 `plan_digest`。 |
| `prompt_template.txt` | 只允许白名单 JSON AST 运算符、现有 object/keypoint/region ID，禁止 Python、坐标编造和关节命令。 |

### 执行节点

| 文件 | 逻辑 |
|---|---|
| `nodes/rekep-executor/node_main.py` | 校验 motion 授权、当前观测、完整绑定和 digest；写入 active-execution 标记；求解并执行；最后清除标记。 |
| `constraint_evaluator.py` | 无副作用解释受限 JSON AST，并计算 relation 的违反量。 |
| `subgoal_solver.py` | 在工作空间内做有界候选/坐标搜索；找不到满足约束的点则失败。 |
| `path_solver.py` | 生成带 clearance 的路径并对密集采样点验证 path constraints。 |
| `kinematics.py` | 检查位置、四元数、工作空间和最大分段长度。 |
| `execution.py` | 生成 MovePose/Gripper 段；保证最多一个 active child Action；要求终态成功；stage 后重新观测和验证；写不可变 ExecutionResult。 |

### 锁、构建与运行维护

| 文件 | 逻辑 |
|---|---|
| `locks/nodes.lock.json` | 锁定 5 个复用节点的官方源码、提交、官方构建入口和源版本，不再把高 glibc 预编译包当真值。 |
| `locks/models.lock.json` | 锁定 DINOv2 源码提交、模型名、权重 URL 和 SHA-256。 |
| `locks/upstream.lock.json` | 记录迁移的上游 ReKep 来源、提交、许可证和采用的概念。 |
| `scripts/fetch_nodes.py` | 按锁文件 clone/fetch，核对 origin 和精确 commit；不会下载未知构建环境的二进制。 |
| `rebuild_reused_nodes.py` | 调用 4 个官方仓库自己的 PyInstaller 脚本，检查独立 `--help` 和 GLIBC 上限，生成 5 个单入口归档及本机构建锁。 |
| `build_nodes.py` | 把 3 个 ReKep 节点构建成 PyInstaller one-file，生成确定性归档和 `.build/custom-nodes.lock.json`。 |
| `fetch_models.py` | 准备锁定 DINOv2 源码/权重，校验提交和权重摘要，写权限 0600 的模型配置；可重复运行。 |
| `prepare_runtime.py` | 从当前 PAOS 配置选择 VLM provider/model，只接受已支持的 OpenAI-compatible 协议，写权限 0600 的运行配置。 |
| `package_skill.py` | 临时 staging Skill、profiles/assets/runtime hook，合并 5+3 节点构建锁，写 archive inventory，生成确定性 Skill tar.gz。 |
| `validate_dist.py` | 验证 8 个单入口节点归档、SHA-256、glibc 记录、Skill inventory、无本机绝对路径、8 节点和单模拟器。 |
| `clean_runtime.py` | 仅删除 cache/tmp 和超过保留策略的不可变记录；检测到 active execution 时拒绝清理，不删除模型和安装。 |

### 测试文件

| 文件 | 覆盖内容 |
|---|---|
| `tests/helpers.py` | 动态装载连字符节点目录中的模块。 |
| `test_perception_contract.py` | snapshot 内容 ID、overlay 不参与摘要及篡改检测。 |
| `test_constraint_contract.py` | AST 白名单、引用合法性、事件配平和非法程序拒绝。 |
| `test_plan_binding.py` | session/revision/observation/plan/digest 的精确绑定。 |
| `test_solver.py` | 子目标和路径约束求解、失败关闭。 |
| `test_serial_execution.py` | 下游 Action 串行、终态成功要求和失败传播。 |
| `test_sim_e2e.py` | 8 节点/单模拟器拓扑、直连 RGB-D/state，以及禁止旧模板/fixture/Python 执行路径。 |

## 6. 从零构建

```bash
# Python 3.12；安装运行、构建和测试依赖
uv sync --frozen --extra build --extra dev

# 复用节点：源码锁定 + 低 glibc 重打包
uv run python scripts/fetch_nodes.py
uv run python scripts/rebuild_reused_nodes.py --mode container --max-glibc 2.31

# 自研节点
uv run python scripts/build_nodes.py --clean

# 可选：准备实际运行所需的 DINOv2
uv run python scripts/fetch_models.py --device cuda

# Skill 归档
uv run python scripts/package_skill.py
```

如果机器已有审核过的源码与权重，可用 `fetch_models.py --source-dir ... --weights ...`，仍会校验 commit 和 SHA-256。

## 7. 测试与验收

### 7.1 静态与单元测试

```bash
python -m compileall -q nodes scripts tests
uv run pytest -q
python ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py skill-src/rekep
```

### 7.2 分发物验收

```bash
uv run python scripts/package_skill.py
uv run python scripts/validate_dist.py
(cd dist && sha256sum -c SHA256SUMS)
```

`validate_dist.py` 的成功输出必须是：

```text
distribution validation passed
```

### 7.3 独立二进制验收

节点必须在仓库外目录启动，避免源码目录意外进入 `sys.path`：

```bash
tmp_dir="$(mktemp -d)"
cd "$tmp_dir"
for node in rekep_perception rekep_planner rekep_executor; do
  archive="$(find "$OLDPWD/dist/nodes" -maxdepth 1 -name "${node}-*-linux-x86_64.tar.gz" -print -quit)"
  test -n "$archive"
  tar -xzf "$archive"
  env -u PYTHONPATH "./${node}" --help
  rm "./${node}"
done
cd -
```

复用节点使用 `.build/reused-nodes.lock.json` 中的 artifact ID 做同样检查。若出现 `GLIBC_X.Y not found`，该产物不合格，必须在更低或相同 glibc 基线重新构建，不能用软链接或替换系统 libc 绕过。

### 7.4 结构和安全检查

```bash
# skill source 只能有两个文件
find skill-src/rekep -maxdepth 1 -type f -printf '%f\n' | sort

# 不允许本机绝对路径或旧实现入口泄漏
rg -n --hidden \
  -g '!.git/**' -g '!dist/**' -g '!.build/**' -g '!REFACTOR_AND_TEST_GUIDE.md' \
  '/data[0-9]*/|/home/|reference/'"rekep-real-plugin"'|latest_'"perception"'|latest_'"plan"'|ensure_'"reference_on_path" .

# 每个节点归档只能有一个根入口
for archive in dist/nodes/*.tar.gz; do tar -tzf "$archive"; done
```

## 8. 运行前配置与调用规则

1. PAOS 当前 provider 必须支持 OpenAI-compatible multimodal chat-completions；执行 Skill 的 `start.sh` 会调用 `prepare_runtime.py`。不支持的 provider 会明确失败，不会降级到模板。
2. DINOv2 必须通过 `fetch_models.py` 准备；感知节点不从源码仓库或网络临时 import/download。
3. 调用 `execute_task` 必须提交规划返回的完整 ID/digest，并显式设置 `allow_motion=true`。
4. 仿真时先验证 Gateway readiness、RGB、depth、JointState 和所有 Tool descriptor，再允许动作。
5. 真机 profile 尚未在本仓库伪造：在取得 Dobot 官方 driver/camera node、锁定来源和完成无运动 dry-run 前，不应把仿真 controller 配置冒充真机链路。

## 9. 生成物

```text
dist/nodes/                         8 个节点单入口归档
dist/skills/rekep-0.3.15.tar.gz     最终 Skill 归档
dist/SHA256SUMS                     全部分发物摘要
.build/reused-nodes.lock.json       本机重打包的 5 个复用节点锁
.build/custom-nodes.lock.json       3 个自研节点构建锁
```

`.build/` 与 `dist/` 均被 `.gitignore` 忽略，可以从源码与锁文件完整重建。

## 10. 本次实测结果（2026-09-29）

- 官方 GitHub 四仓库可匿名拉取，锁定提交与远端 HEAD/目标分支一致；GitLab 匿名 Git 访问要求凭据。
- 5 个复用节点已使用各自官方 `scripts/build_pyinstaller.sh` 在本服务器 glibc 2.35 环境重建；5 个入口的独立 `--help` 检查通过。
- 生成的 5 个复用节点锁记录 `build_glibc_baseline=2.35`；外层 PyInstaller bootloader 检测为 `GLIBC_2.14`。Ubuntu 20.04/glibc 2.31 容器路径已实现，但本次没有可用 Docker daemon，因此未把它写成已执行结果。
- 3 个 ReKep 自研节点已重新构建；在 `/tmp` 作为工作目录且清除 `PYTHONPATH` 后，三个入口的 `--help` 均通过。
- `pytest -q`：`11 passed`。
- Python compileall：通过。
- Skill source quick validation：`Skill is valid!`。
- 分发物校验：`distribution validation passed`。
- `dist/nodes/` 精确包含 8 个锁定节点归档，`dist/skills/` 包含 `rekep-0.3.15.tar.gz`。
