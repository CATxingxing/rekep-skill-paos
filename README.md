# ReKep PAOS Skill

This repository builds a relocatable PhyAgentOS Skill with three custom nodes:

- `rekep_perception` consumes one external RGB-D/state stream and emits immutable `PerceptionSnapshot` values.
- `rekep_planner` asks the VLM selected by the current PAOS configuration for a restricted JSON constraint program.
- `rekep_executor` validates and solves the exact program, then serially drives standard motion and gripper actions.

The ReKep nodes never create or mutate a simulator. The simulation profile uses one downloaded `mujoco_sim` node as the only scene owner. Inter-node data travels through Dora; immutable files are restart recovery records, not a `latest`-file transport.

The `sim-dobot-nova2-robotiq` profile models one Dobot Nova 2 arm with a Robotiq 2F-85 gripper. It is not a full Dobot X-Trainer model: the commercial X-Trainer is a dual-arm platform built around two Nova 2 slave arms and uses its own slave grippers and cameras. Results from this profile therefore validate the Nova 2 single-arm simulation chain only.

## Layout

```text
skill-src/rekep/              only the model-facing Skill document and manifest
profiles/sim-dobot-nova2-robotiq/ launch topology and runtime configuration
nodes/                        the three custom nodes and their shared contracts
assets/                       robot assets staged into the built Skill bundle
locks/                        node, model, and migrated-upstream provenance
scripts/                      deterministic fetch/build/package/runtime helpers
tests/                        contract, binding, solver, serialization, and profile tests
```

## Build and validate

```bash
uv sync --extra build --extra dev
uv run python scripts/fetch_nodes.py
uv run python scripts/rebuild_reused_nodes.py --mode container --max-glibc 2.31
uv run python scripts/build_nodes.py
uv run python scripts/package_skill.py
uv run python scripts/validate_dist.py
uv run pytest
```

The five generic Forge nodes are rebuilt from pinned official source instead of trusting downloaded executables. `--mode container` reproduces the upstream Ubuntu 20.04 / glibc 2.31 release baseline; `--mode local --max-glibc 2.35` is the direct build path for an Ubuntu 22.04 host. See `REFACTOR_AND_TEST_GUIDE.md` for the complete file-by-file runtime and test guide.

`prepare_runtime.py` copies only the active PAOS model/provider settings into a mode-0600 runtime file. The planner fails closed if that provider is unavailable or cannot accept OpenAI-compatible multimodal requests. There is no template planner or generated-Python execution path.

The repository does not contain generated `dist/` output, credentials, runtime state, model weights, or package environments.

## 安装与启动使用（5880 服务器，仿真）

> 以下命令都在 5880 上执行（从 2080 用 `ssh wxr5 '<命令>'`）。所有结果只在 MuJoCo **仿真**中得到，不是真机验证。

### 0. 环境约定

```bash
export PATH=/home/wangxinran/.cargo/bin:/data0/wangxinran/miniconda3/envs/paoswx/bin:$PATH   # paos 在 conda 环境 paoswx 中
unset http_proxy https_proxy                                                                   # 本机代理对 5880 内网无效
```

- PAOS 配置：`~/.PhyAgentOS/config.json`，Agent 模型为 `gpt-6-astra-phyagentos`（`openai_responses`）。
- 启动 Skill 时 `prepare_runtime.py` 会把当前 PAOS 的模型设置写成 `~/.paos-rekep/run/rekep/vlm-provider.json`（权限 0600，含 api_key，不要打印或提交），planner 的 VLM 也使用同一模型。
- 建议关闭 `agents.evolution`（`config.json` 里 `"enabled": false`）。已安装的 Forge Skill 被演化写出工作区副本后会遮蔽安装版，导致 `forge_task_create` 报 "primary Skill activation has no Forge binding candidate"；若出现，把 `~/.PhyAgentOS/workspace/skills/rekep/` 移走即可恢复。

### 1. 构建与安装

构建与打包见上面的 "Build and validate"，产物在 `dist/skills/rekep-<版本>.tar.gz` 与 `dist/nodes/`。

自建节点（rekep_*、mujoco_sim、gateway、motion_server 等）不在 Registry 上，直接 `paos skill install --local` 会去 Registry 下载而失败，所以要先装节点、再装 Skill。已封装成脚本：

```bash
# 版本号与 skill.yaml 中的 version 一致，当前为 0.4.8
bash /data0/wangxinran/rekep-evidence/0324-agent/tools/install_skill.sh 0.4.8
```

脚本依次做：校验 `dist/SHA256SUMS` → `paos skill stop rekep` → 用 PAOS NodeInstaller 按包内 lock 安装 `dist/nodes` 中的节点 → `paos skill install --local dist/skills/rekep-0.4.8.tar.gz -y` → 逐个 `paos forge-node verify rekep <节点>` → 输出 `paos skill list` / `paos skill inspect rekep`（记录在 `install-0.4.8/`）。

确认安装：

```bash
paos skill list
paos skill inspect rekep
```

重装规则：节点 artifact 内容变了必须升版本，否则安装器拒绝；Skill 同版本重装会被判 "already ready" 而跳过，所以改了 profile 或 SKILL.md 必须升 Skill 版本。

### 2. 启动、检查、停止

四个 profile：

| profile | 场景 |
|---|---|
| `sim-dobot-nova2-robotiq` | baseline（自由场景，指令自拟） |
| `sim-dobot-nova2-robotiq-general-pickup` | 拾取并抬起 |
| `sim-dobot-nova2-robotiq-stack-blocks` | 堆叠三个方块 |
| `sim-dobot-nova2-robotiq-push-t` | 推 T 形块到轮廓 |

```bash
paos skill start rekep --profile sim-dobot-nova2-robotiq-general-pickup   # 启动（同一时间只能运行一个 profile）
paos skill status rekep      # 期望 State: running，Gateway GET /tools: ready
paos skill logs rekep        # 运行时生命周期日志
paos skill stop rekep        # 停止（不会关闭共享的 Dora 服务）
```

- 换场景：先 `stop` 再用新的 `--profile` `start`。已在运行时再 `start` 会报 "Gateway address http://127.0.0.1:19021 is already in use"，这是预期行为。
- 不要用 `paos skill stop rekep --force`，除非确认没有未收尾的任务绑定（先看 `~/.PhyAgentOS/run/skills/rekep.json` 的 `active_invocations` / `active_sessions` / `active_task_bindings` 是否为空）。
- 启动后约需十几秒到一分钟就绪；`status` 里 Gateway 显示 `ready` 后再下指令。

### 3. 用 `paos agent -m` 执行任务

由 PAOS Agent 自己调用 `rekep.get_context` → `rekep.plan_task` → `rekep.execute_task`。指令末尾必须带授权语句 `已授权在当前仿真 profile 中运动。`，否则 Agent 不会发动作。建议在 `/tmp` 下运行。

```bash
cd /tmp

# pick（先启动 general-pickup profile）
paos agent -m "Pick up the cyan cylinder and lift it off the table。已授权在当前仿真 profile 中运动。"

# stack（先启动 stack-blocks profile）
paos agent -m "Stack the three cubes into one tower on the green square: the yellow cube at the bottom, the red cube in the middle and the blue cube on top。已授权在当前仿真 profile 中运动。"

# push（先启动 push-t profile）
paos agent -m "Push the blue T-shaped block onto the green T-shaped outline so that it matches the outline's position and orientation. Push only; do not lift the block。已授权在当前仿真 profile 中运动。"
```

- 每个任务约 4–5 分钟（push_t 可能更久）；只有一次 `execute_task` 覆盖整个闭环，`deadline_ms` 不确定时用 600000。
- 可加 `--no-markdown`、`--logs`、`-s <会话名>`。
- 首个 `get_context` 偶尔被 PAOS 客户端的 10 秒超时打断（Gateway 偶发丢响应），Agent 没有发运动，属于正确的失败关闭；SKILL.md 已要求只读查询最多重试两次。
- 一个任务结束后若要重跑同一场景，建议 `stop` 再 `start`（`--restart`），保证场景复位。

### 4. Token 用量与剩余额度

`paos agent -m` 结束时（交互模式下每轮回复后）会输出本次任务各模型的调用次数、输入/输出/合计 token、API 剩余额度和网关记账的本次消耗，同时追加到 `~/.PhyAgentOS/logs/token-usage.jsonl`。随时查询剩余额度：

```bash
paos quota
```

统计的是 PAOS 进程内的模型调用；planner 的 VLM 调用运行在节点进程里，不单独列出，但与主 Agent 使用同一个 API key，会计入“网关记账消耗”。

### 任务视频

感知节点（rekep_perception 0.4.3 起）不再保存逐帧图片，而是把相机画面直接编码成视频分段：
`~/.paos-rekep/run/rekep/recordings/<session>/<开始时间 ns>.mp4`，每段最长 10 分钟，画面中断超过 2 秒或分辨率变化时另起一段；旁边的 `.timeline` 是每秒一行的“墙钟时间 仿真时间”，用来算仿真实时率。超过 `retention_s`（默认 3600 秒）的分段会被自动删除。rekep_perception 0.4.4 起只在动作执行期间录制：执行器在 `execute_task` 运行时持有 `~/.paos-rekep/run/rekep/active-execution.json`，感知节点见到它就开始录，它消失后再录 `stop_delay_s`（10 秒）就停止并收尾该分段；规划、等待、核验等阶段不再录制。相关设置在各 profile 的 `perception.yaml`：`recording: {enabled: true, fps: 10, retention_s: 3600, trigger: execution, stop_delay_s: 10}`（`trigger: always` 为一直录制），可另加 `segment_s`。

`paos agent -m` 结束后，以及交互模式（`paos agent`）每轮回复后，都会（等录制收尾）从这些分段里剪出本次任务的片段；没有执行动作的任务没有视频，保存到 `~/.PhyAgentOS/workspace/videos/<会话名>-<时间>.mp4`，并打印路径。用 `tools/agent_run.sh` 跑时，每轮目录里另有 `video.mp4`、`before.jpg`、`after.jpg`（首尾帧）和 `recording.json`。

### 5. 带评测的批量运行（可选）

`/data0/wangxinran/rekep-evidence/0324-agent/tools/` 里的脚本会启动 profile、做两次只读就绪探测、调用 `paos agent -m`，再用 sim_tasks 真值回放生成 `verification.json`、照片和录像：

```bash
cd /data0/wangxinran/rekep-evidence/0324-agent/tools
tmux new-session -d -s wxr-agent  "./agent_run.sh stack_blocks run7 --restart"       # 单轮；run 名不能重复
tmux new-session -d -s wxr-series "./series.sh push_t run8 run9 run10"               # 连续多轮
./baseline_run.sh baseline-run1 "<指令>" --restart                                    # baseline profile
```

长任务请放进 tmux；运行 PAOS 期间不要并行跑离线 harness 或构建。结果文件和 3 个场景的历史轮次见同目录的 `README.md`（阶段 A 结果说明）。
