# rekep-skill-paos

把参考插件 [`PhyAgentOS-rekep-real-plugin`](https://github.com/baiyu858/PhyAgentOS-rekep-real-plugin) 的**仿真 ReKep 能力**，按 PhyAgentOS 最新 Node/Forge 协议重新组织成一个可安装的 Skill（`rekep-sim`），接入 `PhyAgentOS-core`。

**范围**：仅仿真；本机以 **MuJoCo** 实现 `ReKepEnv`（算法照搬参考实现，env 换实现）；不做真机。
（原计划优先 OmniGibson，但 Isaac Sim pip 安装要求 glibc ≥ 2.35，本构建机 2.31 装不了，故改用 MuJoCo —— 见 `docs/P1-报告.md`。）

## 状态

- **P0–P4 完成**；Skill **`rekep-sim` v0.1.7** 可被 `paos skill install/start`。
- **真实 VLM 约束生成闭环**：`deepseek-v4-flash-vision-exp` 生成 ReKep 约束 → AST 沙箱 → 参考 `subgoal/path` 分层求解 → MuJoCo 执行 → `succeeded`（落点误差 ~0.02–0.035 m，产出 MP4）。
- **Agent 端到端**：`paos agent` 读 `SKILL.md`，严格 `get_context → plan_task → execute_task`，AgentTask 判定 `succeeded`。
- 测试：`pytest` **26 passed**（参考对齐 + 沙箱 + VLM 计划路径）。
- **P5 未做**（可选）：当前 node 是**单文件 shell 包装**（本地安装/启动已验证），生产级**可移植单文件**与 **Registry/TOS 发布**留待决定。

## 目录

```text
需求.md                 需求
docs/                   开发手册 + P1/P2/P3/P4 报告
src/rekep_sim/          代码：env(MuJoCo)/runtime/perception/sandbox/provider/refimpl/state
nodes/{perception,plan,execute}/   3 个 Forge Node（单文件 shell 入口 + node_main.py）
skill-src/rekep-sim/    Skill bundle 源码（SKILL.md / skill.yaml / profiles / assets）
tests/                  单元与契约测试
scripts/                env.sh、dev shim、dora shim、构建/安装/验收脚本
reference/              参考实现（git submodule，pin `c532a28`）
evidence/               验收证据（命令、输出、日志、视频）
```

## 快速开始

```bash
source scripts/env.sh                       # 强制所有状态/缓存到 /data7

# 1) 构建 node 归档 + 生成 skill.yaml + 打包 bundle
bash scripts/build_skill.sh

# 2) 安装（本地 resolver；glibc<2.32 用 dev shim）
export PYTHONPATH="$PWD/scripts/shim:$PYTHONPATH"
PAOS_ENV=/data7/home/linjiongxiao/miniconda3/envs/paos
export PATH="$PWD/scripts/shimbin:$PAOS_ENV/bin:$PATH"   # dora shim：私有端口，避免与他人冲突
python scripts/paos_dev_shim.py skill install dist/skills/rekep-sim-0.1.7.tar.gz --local --yes
bash scripts/install_nodes_local.sh

# 3) 启动 / 查看
python scripts/paos_dev_shim.py skill start rekep-sim --profile sim
python scripts/paos_dev_shim.py skill status rekep-sim   # 期望 Gateway GET /tools: ready + 三个 context ready

# 4) Agent 端到端（需 LLM 与 VLM key，禁止打印 key）
export REKEP_VLM_API_KEY=...  REKEP_VLM_BASE_URL=...  REKEP_VLM_MODEL=...
python scripts/paos_dev_shim.py agent -m '使用 rekep-sim：抓取橙色方块并放入绿色目标区；先感知和规划，再执行仿真，最后返回视频路径。'
```

## 关键约束

- **不触碰根盘** `/`、`/tmp`、`/root`；全部状态在 `/data7`（`scripts/env.sh` 统一设置）。
- Node 产物为**单文件 `executable_tar_gz`** + sha256 lock（当前 PAOS 只接受此形态）。
- 构建主机 Ubuntu 20.04 / glibc 2.31：本地用 **dev shim**（`scripts/shim/`）；生产安装/运行在 glibc ≥ 2.32。
- 共享主机上 Dora 默认端口被他人占用 → 用 **dora shim**（`scripts/shimbin/dora`，私有 coordinator 端口）。
- 不改动 `PhyAgentOS-core`、`rekep-paos`。
- VLM 生成的 Python 约束必须经 **AST 白名单沙箱**执行。

## 参考实现

`reference/rekep-real-plugin` 为 git submodule，pin 到 `c532a28`。算法模块（keypoint proposal / constraint generation / subgoal-path solver / utils）**原样引用不改**；`ReKepEnv` 通过接口隔离以便后续换回 OmniGibson。

## 验收

- 离线契约/对齐：`bash scripts/nsrun.sh bash scripts/acceptance_full.sh` → `P4_ACCEPTANCE: PASS`。
- 带 VLM 的 Agent 端到端：`bash scripts/acceptance_agent_shim.sh`（需 VLM/LLM key）。
- 证据：`evidence/P4/`、`evidence/P5v/`；详见 `docs/P4-验收报告.md`。
