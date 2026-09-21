---
name: paos-smoke
description: Minimal Skill used only to validate the PAOS Skill Runtime + single-file node install/verify contract during P1.
metadata: {"PhyAgentOS":{"always":false,"requires":{"runtime":["paos-smoke"]}}}
---

# PAOS Smoke Skill

This Skill exists only to validate packaging, installation, and Node lock verification.
It declares one placeholder Tool (`smoke.ping`) and locks the shared `gateway-1.1.0` Node.
It is not intended to run a real dataflow.
