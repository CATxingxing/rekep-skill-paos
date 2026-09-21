# rekep-sim-paos

把参考插件 [`PhyAgentOS-rekep-real-plugin`](https://github.com/baiyu858/PhyAgentOS-rekep-real-plugin) 的**仿真 ReKep 能力**，按 PhyAgentOS 最新 Node/Forge 协议重新组织成一个可安装的 Skill（`rekep-sim`），接入 `PhyAgentOS-core`。

**范围**：仅仿真；本机以 **MuJoCo** 实现 `ReKepEnv`（算法照搬参考实现，env 换实现）；不做真机。

## 状态

- **P0 完成**：`需求.md`、`docs/开发手册.md`
- **P1 完成**：基线（PhyAgentOS `v1.0.0` + Dora `0.4.1`）、Skill/Node 契约实测通过、`docs/P1-报告.md`
- **P2+**：进行中（MuJoCo env、3 Tool/2 Node、AST 沙箱、bundle、验收）

## 目录

```text
需求.md                 需求汇总
docs/                   开发手册、P1 报告、验收方案
skill-src/rekep-sim/    Skill bundle 源码（SKILL.md / skill.yaml / profiles / assets）
nodes/                  自定义 Forge Node 源码
sandbox/                VLM 约束的 AST 沙箱执行器
tests/                  单元与契约测试
scripts/                env.sh、dev shim、打包/部署脚本
reference/              参考实现（git submodule，pin 到固定 commit）
evidence/               验收证据（命令、输出、sha256）
```

## 快速开始

```bash
# 强制所有状态/缓存到 /data7（根盘只剩几 GB，禁止使用）
source scripts/env.sh

# 在 glibc<2.32 的构建主机上运行 paos 需要 dev shim（安装/解包使用）
export PYTHONPATH="$PWD/scripts/shim:$PYTHONPATH"
python scripts/paos_dev_shim.py --help
```

## 关键约束

- **不触碰根盘** `/`、`/tmp`、`/root`；全部状态在 `/data7`。
- Node 产物为**单文件 `executable_tar_gz`**（当前 PAOS 只接受此形态）。
- 构建主机 Ubuntu 20.04 / glibc 2.31：本地用 dev shim；**生产安装/运行在 glibc ≥ 2.32**。
- 不改动 `PhyAgentOS-core`、`rekep-paos`。
- VLM 生成的 Python 约束必须经 **AST 白名单沙箱**执行。

## 参考实现

`reference/rekep-real-plugin` 为 git submodule，pin 到 `c532a28`。算法模块（keypoint proposal / constraint generation / subgoal-path solver / utils）**原样引用不改**；`ReKepEnv` 通过接口隔离以便后续换回 OmniGibson。

## 验收

见 `docs/开发手册.md`；分阶段 A 平台/产物契约、B 参考对齐、C Node/Tool 契约、D 沙箱与约束语义、E SKILL/LLM 编排、F 端到端总体。
