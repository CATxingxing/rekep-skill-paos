# P1 baseline evidence
2026-09-21T02:16:37Z

## host
OS: Ubuntu 20.04.6 LTS
glibc: ldd (Ubuntu GLIBC 2.31-0ubuntu9.18) 2.31
kernel: 5.4.0-174-generic

## toolchain
PhyAgentOS: 🍞 PhyAgentOS v1.0.0
dora: dora-cli 0.4.1 dora-message: 0.7.0 dora-rs (Python): 0.4.1   

## DINOv2
7764ea0f912e53c92e82eb78a2a1631e92725fc8
b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9  /data7/home/linjiongxiao/.paos-rekep/models/weights/dinov2_vits14.pth

## gateway node
bb42c141b299a854fed2ff31f20be5b83551c7c47a30bce06fccd717da202387  /data7/home/linjiongxiao/.tmp/opencode/gw/gateway.tar.gz
gateway

## smoke skill
sha256: d22023c8533d8a081e3659f426b8b6b230839f7aca6afe4eee9421bc556a3f3e
size_bytes: 1470

## KNOWN BLOCKER: glibc<2.32 chmod follow_symlinks (dev shim used)
## KNOWN BLOCKER: Isaac Sim pip needs glibc>=2.35 -> OmniGibson not installable on build host
