# Environment Snapshot (2026-05-17)

## System
```
Hostname: VM-0-14-ubuntu
User: chenjinfeng
Date: Sun May 17 07:39:10 PM CST 2026
Linux VM-0-14-ubuntu 5.4.119-19-0009.59+ #2 SMP Tue Jun 3 15:40:20 CST 2025 x86_64 x86_64 x86_64 GNU/Linux
NAME="Ubuntu"
VERSION="22.04 (Jammy Jellyfish)"
Uptime:  19:39:10 up 153 days,  8:28, 39 users,  load average: 15.78, 16.76, 15.80
```

## Hardware
```
--- CPU ---
CPU(s):                          124
Model name:                      Intel(R) Xeon(R) Platinum 8374C CPU @ 2.70GHz
Thread(s) per core:              2
Core(s) per socket:              31
Socket(s):                       2
NUMA node0 CPU(s):               0-61
NUMA node1 CPU(s):               62-123

--- Memory ---
               total        used        free      shared  buff/cache   available
Mem:           1.8Ti       331Gi        99Gi       163Gi       1.4Ti       1.3Ti
Swap:             0B          0B          0B

--- Disk (cfsdata) ---
Filesystem      Size  Used Avail Use% Mounted on
/dev/vda2       200G  188G  4.4G  98% /
10.0.0.6:/      1.0P  103T  922T  11% /cfsdata

--- Disk usage of your dirs ---
## Hardware
```
--- CPU ---
CPU(s):                          124
Model name:                      Intel(R) Xeon(R) Platinum 8374C CPU @ 2.70GHz
Thread(s) per core:              2
Core(s) per socket:              31
Socket(s):                       2
NUMA node0 CPU(s):               0-61
NUMA node1 CPU(s):               62-123

--- Memory ---
               total        used        free      shared  buff/cache   available
Mem:           1.8Ti       331Gi        99Gi       163Gi       1.4Ti       1.3Ti
Swap:             0B          0B          0B

--- Disk (cfsdata) ---
Filesystem      Size  Used Avail Use% Mounted on
/dev/vda2       200G  188G  4.4G  98% /
10.0.0.6:/      1.0P  103T  922T  11% /cfsdata

--- Disk usage of your dirs ---
7.2T	/cfsdata/chenjinfeng/datasets
4.2T	/cfsdata/chenjinfeng/projects
75G	/cfsdata/chenjinfeng/models
26G	/cfsdata/chenjinfeng/miniconda3
25G	/cfsdata/chenjinfeng/isaac_sim_4_0_0
20G	/cfsdata/chenjinfeng/openpi_cache
12G	/cfsdata/chenjinfeng/uv
918M	/cfsdata/chenjinfeng/tools
347M	/cfsdata/chenjinfeng/chenjinfeng
65M	/cfsdata/chenjinfeng/docs
53M	/cfsdata/chenjinfeng/bin
360K	/cfsdata/chenjinfeng/tmp
24K	/cfsdata/chenjinfeng/tmpx8xck6e9
24K	/cfsdata/chenjinfeng/tmpu3n_bsm9
24K	/cfsdata/chenjinfeng/tmprytd70hf
12K	/cfsdata/chenjinfeng/LEARNING_LOG.md
4.0K	/cfsdata/chenjinfeng/robotwin_adjust_bottle_autoeval.sh
4.0K	/cfsdata/chenjinfeng/envs
4.0K	/cfsdata/chenjinfeng/downloads
4.0K	/cfsdata/chenjinfeng/custom_icd.json
```

## GPU
```
Sun May 17 19:59:41 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A800-SXM4-80GB          Off |   00000000:23:00.0 Off |                    0 |
| N/A   48C    P0             76W /  400W |   72750MiB /  81920MiB |     38%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A800-SXM4-80GB          Off |   00000000:24:00.0 Off |                    0 |
| N/A   40C    P0             93W /  400W |   68605MiB /  81920MiB |    100%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA A800-SXM4-80GB          Off |   00000000:43:00.0 Off |                    0 |
| N/A   42C    P0             91W /  400W |   68695MiB /  81920MiB |    100%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA A800-SXM4-80GB          Off |   00000000:44:00.0 Off |                    0 |
| N/A   43C    P0             93W /  400W |   68605MiB /  81920MiB |    100%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA A800-SXM4-80GB          Off |   00000000:83:00.0 Off |                    0 |
| N/A   37C    P0            284W /  400W |   13607MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA A800-SXM4-80GB          Off |   00000000:84:00.0 Off |                    0 |
| N/A   30C    P0             70W /  400W |       6MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA A800-SXM4-80GB          Off |   00000000:C3:00.0 Off |                    0 |
| N/A   30C    P0             77W /  400W |     122MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA A800-SXM4-80GB          Off |   00000000:C4:00.0 Off |                    0 |
| N/A   33C    P0             86W /  400W |   15828MiB /  81920MiB |     15%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2615020    C+G   python                                 4126MiB |
|    0   N/A  N/A         4110331      C   .../envs/galaxea_vla/bin/python3      68596MiB |
|    1   N/A  N/A         4110338      C   .../envs/galaxea_vla/bin/python3      68594MiB |
|    2   N/A  N/A         4110339      C   .../envs/galaxea_vla/bin/python3      68684MiB |
|    3   N/A  N/A         4110340      C   .../envs/galaxea_vla/bin/python3      68594MiB |
|    4   N/A  N/A          297866      C   python                                13594MiB |
|    6   N/A  N/A         2615020      G   python                                    6MiB |
|    6   N/A  N/A         3377678      G   python3                                  69MiB |
|    6   N/A  N/A         3795527      G   ...onda3/envs/ga_test/bin/python          6MiB |
|    7   N/A  N/A         3795527    C+G   ...onda3/envs/ga_test/bin/python      15804MiB |
+-----------------------------------------------------------------------------------------+

--- GPU 详细规格 ---
index, name, memory.total [MiB], memory.free [MiB], driver_version, compute_cap
0, NVIDIA A800-SXM4-80GB, 81920 MiB, 8404 MiB, 580.95.05, 8.0
1, NVIDIA A800-SXM4-80GB, 81920 MiB, 12549 MiB, 580.95.05, 8.0
2, NVIDIA A800-SXM4-80GB, 81920 MiB, 12459 MiB, 580.95.05, 8.0
3, NVIDIA A800-SXM4-80GB, 81920 MiB, 12549 MiB, 580.95.05, 8.0
4, NVIDIA A800-SXM4-80GB, 81920 MiB, 67547 MiB, 580.95.05, 8.0
5, NVIDIA A800-SXM4-80GB, 81920 MiB, 81148 MiB, 580.95.05, 8.0
6, NVIDIA A800-SXM4-80GB, 81920 MiB, 81032 MiB, 580.95.05, 8.0
7, NVIDIA A800-SXM4-80GB, 81920 MiB, 65326 MiB, 580.95.05, 8.0

--- CUDA ---
Please ask your administrator.
nvcc not found (probably ok, JAX/PyTorch bring their own)
CUDA_VISIBLE_DEVICES (if set):
```

## Multi-GPU Capability
```
	[4mGPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	NIC0	NIC1	NIC2	NIC3	NIC4	NIC5	NIC6	NIC7	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV8	NV8	NV8	NV8	NV8	NV8	NV8	PIX	PIX	NODE	NODE	SYS	SYS	SYS	SYS	0-61	0		N/A
GPU1	NV8	 X 	NV8	NV8	NV8	NV8	NV8	NV8	PIX	PIX	NODE	NODE	SYS	SYS	SYS	SYS	0-61	0		N/A
GPU2	NV8	NV8	 X 	NV8	NV8	NV8	NV8	NV8	NODE	NODE	PIX	PIX	SYS	SYS	SYS	SYS	0-61	0		N/A
GPU3	NV8	NV8	NV8	 X 	NV8	NV8	NV8	NV8	NODE	NODE	PIX	PIX	SYS	SYS	SYS	SYS	0-61	0		N/A
GPU4	NV8	NV8	NV8	NV8	 X 	NV8	NV8	NV8	SYS	SYS	SYS	SYS	PIX	PIX	NODE	NODE	62-123	1		N/A
GPU5	NV8	NV8	NV8	NV8	NV8	 X 	NV8	NV8	SYS	SYS	SYS	SYS	PIX	PIX	NODE	NODE	62-123	1		N/A
GPU6	NV8	NV8	NV8	NV8	NV8	NV8	 X 	NV8	SYS	SYS	SYS	SYS	NODE	NODE	PIX	PIX	62-123	1		N/A
GPU7	NV8	NV8	NV8	NV8	NV8	NV8	NV8	 X 	SYS	SYS	SYS	SYS	NODE	NODE	PIX	PIX	62-123	1		N/A
NIC0	PIX	PIX	NODE	NODE	SYS	SYS	SYS	SYS	 X 	PIX	NODE	NODE	SYS	SYS	SYS	SYS
NIC1	PIX	PIX	NODE	NODE	SYS	SYS	SYS	SYS	PIX	 X 	NODE	NODE	SYS	SYS	SYS	SYS
NIC2	NODE	NODE	PIX	PIX	SYS	SYS	SYS	SYS	NODE	NODE	 X 	PIX	SYS	SYS	SYS	SYS
NIC3	NODE	NODE	PIX	PIX	SYS	SYS	SYS	SYS	NODE	NODE	PIX	 X 	SYS	SYS	SYS	SYS
NIC4	SYS	SYS	SYS	SYS	PIX	PIX	NODE	NODE	SYS	SYS	SYS	SYS	 X 	PIX	NODE	NODE
NIC5	SYS	SYS	SYS	SYS	PIX	PIX	NODE	NODE	SYS	SYS	SYS	SYS	PIX	 X 	NODE	NODE
NIC6	SYS	SYS	SYS	SYS	NODE	NODE	PIX	PIX	SYS	SYS	SYS	SYS	NODE	NODE	 X 	PIX
NIC7	SYS	SYS	SYS	SYS	NODE	NODE	PIX	PIX	SYS	SYS	SYS	SYS	NODE	NODE	PIX	 X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_bond_0
  NIC1: mlx5_bond_1
  NIC2: mlx5_bond_2
  NIC3: mlx5_bond_3
  NIC4: mlx5_bond_4
  NIC5: mlx5_bond_5
  NIC6: mlx5_bond_6
  NIC7: mlx5_bond_7

```

## Python Environment (openpi)
```
uv 0.10.1
Python 3.11.14
3.11
--- Key packages ---
torch:        2.7.1+cu126  (CUDA: True, devices: 8)
jax:          0.5.3    (devices: [CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3), CudaDevice(id=4), CudaDevice(id=5), CudaDevice(id=6), CudaDevice(id=7)])
transformers: 4.53.2

--- All packages ---
Package                  Version     Editable project location
------------------------ ----------- -----------------------------------------------------------
absl-py                  2.3.0
aiohappyeyeballs         2.6.1
aiohttp                  3.12.4
aiosignal                1.3.2
annotated-types          0.7.0
antlr4-python3-runtime   4.9.3
asttokens                3.0.0
attrs                    25.3.0
augmax                   0.4.1
av                       17.0.0
beartype                 0.19.0
beautifulsoup4           4.13.4
blinker                  1.9.0
cachetools               5.5.2
certifi                  2025.4.26
cffi                     1.17.1
cfgv                     3.4.0
charset-normalizer       3.4.2
chex                     0.1.90
click                    8.2.1
cloudpickle              3.1.1
cmake                    4.0.2
comm                     0.2.2
contourpy                1.3.2
crc32c                   2.7.1
cycler                   0.12.1
datasets                 3.6.0
debugpy                  1.8.14
decorator                5.2.1
deepdiff                 8.5.0
diffusers                0.33.1
dill                     0.3.8
distlib                  0.3.9
dm-control               1.0.14
dm-env                   1.6
dm-tree                  0.1.9
docker-pycreds           0.4.0
docstring-parser         0.16
donfig                   0.8.1.post1
draccus                  0.10.0
einops                   0.8.1
equinox                  0.12.2
etils                    1.12.2
evdev                    1.9.2
executing                2.2.0
farama-notifications     0.0.4
filelock                 3.18.0
flask                    3.1.1
flatbuffers              25.2.10
flax                     0.10.2
fonttools                4.58.1
frozenlist               1.6.0
fsspec                   2025.3.0
gcsfs                    2025.3.0
gdown                    5.2.0
gitdb                    4.0.12
gitpython                3.1.44
glfw                     2.9.0
```

## openpi Repo Status
```
Path: /cfsdata/chenjinfeng/projects/openpi
Git commit: c23745b
Git branch: main
Last commit: 2026-05-06 Remove redundant tree dependency (#937)

--- Submodules ---
-d1dc83afd89ded4379851257fe5d85632d31d5ec third_party/aloha
-f78abd68ee283de9f9be3c8f7e2a9ad60246e95c third_party/libero
```

## Datasets and Weights
```
--- Datasets (sizes) ---
68G	/cfsdata/chenjinfeng/datasets/aloha_pen_uncap_diverse
3.4T	/cfsdata/chenjinfeng/datasets/droid
