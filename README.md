# 教学示范蒸馏实验：Instruct 500条

本目录提供实验操作说明、代码、配置和数据材料。默认教师 **Qwen2.5-7B-Instruct**，学生 **Qwen2.5-0.5B-Instruct**，优先使用AutoDL **PyTorch2.8.0 / Python3.12 / CUDA12.8**镜像base。所需脚本均在本目录中。

环境组合是复现起点，需在当前实例完成依赖解析和GPU检查。环境安装器只操作当前Python环境，不创建新的Python或Conda环境；兼容版本可根据验证结果调整。已配置好的教学镜像应先检查现有环境，通常不需要重新执行安装。

## 目录

| 路径 | 用途 |
|---|---|
| `docs/跟课手册.md` | **跟视频逐步操作的入口**：每步目的+命令+预期结果，含故障处理 |
| [`docs/结构图/`](docs/结构图/README.md) | Qwen2.5 0.5B/7B 模型结构教学示意图（三张，点开即看） |
| `docs/学生实验指南.md` | 实验目标、阶段解释、结果理解与报告要求 |
| `docs/AI辅助实验说明.md` | 学生使用AI协助实验的可选说明 |
| `chat.py` | 双服务对比对话小工具（可选环节用，`python chat.py 端口 服务名`） |
| `configs/course.json` | 唯一默认实验配置，学生为0.5B-Instruct |
| `configs/requirements-*.txt` | 可调整的预编译候选依赖 |
| `scripts/setup_env.py` | 当前镜像环境内解析/安装，默认plan，`--apply`执行 |
| `scripts/course.py` | 下载、数据筛选、生成、训练、导出、评测、冻结、报告 |
| `scripts/run_stage.py` | Linux推理进程监督，限定自身进程组，不使用宽泛pkill |
| `scripts/use_materials.py` | 备用材料显式导入，不伪装成现场生成 |
| `materials/prepared/` | 固定候选1000条、dev100、test300及数据来源/哈希 |
| `materials/teacher500/` | 已验证500条教师示范、审计、哈希；用于可选复现/排障 |
| `tests/`、`bundle-manifest.json` | CPU逻辑测试、文件完整性清单；清单不代表已压缩 |
| [`experiment-jin/`](experiment-jin/README.md) | **同伴实现的另一条路线**：本地 Windows 单机 · 纯 transformers+PEFT 手写 · QLoRA 4bit（1.5B 教师 → 0.5B 学生），不依赖本包脚本与镜像 |

仅包含轻量代码与文本数据；不含完整Numina原始数据、模型基座、adapter、merged、环境或安装缓存。原始数据ID与revision在configs/course.json，模型通过download-models命令从ModelScope下载；随附筛选数据使明确标注的备用路径可离线准备数据。

## 新机起步

将本仓库克隆或上传为 `/root/distill-course`（二选一）：`git clone https://github.com/Beirana/distill-course /root/distill-course`，或下载压缩包解压到该路径。不要与旧目录合并覆盖；若目标已存在，先改用新的目标目录，脚本通过自身位置找配置。

以下在AutoDL Linux终端执行，先核对磁盘预算：教师约15GB、学生/merged各约1GB，另加依赖和下载临时空间。

```bash
cd /root/distill-course
export COURSE_DATA_ROOT=/root/distill-work
python scripts/verify_bundle.py
python scripts/course.py preflight
python scripts/setup_env.py gen
# 看清plan再安装；若解析失败，先调查兼容wheel，不改走源码编译。
python scripts/setup_env.py gen --apply
python scripts/setup_env.py train
python scripts/setup_env.py train --apply
python scripts/hello.py
```

训练依赖若与生成端确有冲突，可建立单独训练环境，在该环境执行train安装器，并设置 `COURSE_TRAIN_ENV` 为实际前缀。默认没有此变量时直接复用当前 `sys.prefix`。train的plan阶段会拉取LLaMA-Factory的Python源码用于依赖解析，但不会安装pip包；这不是vLLM/CUDA源码编译。

## A：默认重新生成500条

新实例首次下载模型前必做：镜像只保存系统盘，数据盘 `/root/autodl-tmp` 每台实例都是空的，`/root/distill-work` 下指向数据盘的 models/cache 符号链接因此悬空，跳过此步 download-models 会报 `FileNotFoundError`：

```bash
mkdir -p /root/autodl-tmp/distill-assets/{models,cache} /root/distill-work
ln -sfn /root/autodl-tmp/distill-assets/models /root/distill-work/models
ln -sfn /root/autodl-tmp/distill-assets/cache  /root/distill-work/cache
```

```bash
python scripts/course.py download-models
python scripts/course.py model-status
python scripts/course.py prepare-data
python scripts/run_stage.py generate --run smoke01 --mode smoke
python scripts/run_stage.py eval --run smoke01 --model before --split dev
python scripts/course.py make-config --run smoke01
python scripts/course.py train --run smoke01
python scripts/course.py export --run smoke01
python scripts/run_stage.py eval --run smoke01 --model student --split dev

python scripts/run_stage.py generate --run instruct500-new-01 --mode pilot500
python scripts/course.py make-config --run instruct500-new-01
python scripts/run_stage.py eval --run instruct500-new-01 --model before --split dev
python scripts/course.py train --run instruct500-new-01
python scripts/course.py export --run instruct500-new-01
python scripts/run_stage.py eval --run instruct500-new-01 --model student --split dev
python scripts/run_stage.py eval --run instruct500-new-01 --model teacher --split dev
python scripts/course.py report --run instruct500-new-01
```

已有完整模型先验证复用；缺模型默认使用 `download-models` 官方 ModelScope 路径，完成后写入课程 manifest。它依次处理两个模型，不代表客户端内部只有一条连接。`model-download-accelerator` 仅作备选：官方路线反复失败、停滞、持续过慢影响时间安排，或用户主动要求时，再评估并行/HF-mirror 及课程登记兼容性。无需先安装或探测并行技能，不为换工具中断健康下载。记录实际耗时与校验，不预设提速。

远程 GPU 推荐本机免密 SSH（复用已有别名），除非使用者另选 Jupyter 等入口。课堂默认使用者自己执行小量验证与 smoke，Agent 逐行解释。下载可在另一个终端持续，等待时使用 [单样本文件讲解与便捷命令](docs/单样本与便捷命令.md)，不必空等。

镜像保持不变；课程包更新放入独立目录，继续使用原生成解释器与训练 CLI。保留旧目录，它可能承载训练环境 editable 安装的 vendor 源码。主机检查按阶段解释：数据盘布局、非参考补丁版本和无关网络故障只提示，不默认重装环境。新包校验失败不等于 CUDA 损坏；模型哈希/必要算子失败仍需修复。更新代码前后对比课程配置，配置和资产契约未变则复用原数据；不要为通过检查改写旧 manifest 或冻结 run。

如果模型由可信镜像、离线包或外部并行下载器取得，不要为了生成课程 manifest 再调用官方客户端下载，也不要手写课程内部清单。先让下载工具产出统一的 provider-neutral source manifest：其中 `source.revision` 必须表示同一个仓库快照，所有文件都要有可信 SHA256。然后只在本地验证并登记：

```bash
python scripts/course.py register-models \
  --teacher-source-manifest /path/teacher-source-manifest.json \
  --student-source-manifest /path/student-source-manifest.json
python scripts/course.py model-status
```

`register-models` 不联网、不下载，也不覆盖已有登记；可以只传已下载完成的一个角色，另一角色稍后单独登记。它要求文件集合、大小和 SHA256 与来源清单一致，并检查配置和分片索引。它检查 commit 字符串格式，但不会联网证明该 commit 是仓库快照，来源工具仍须正确解析/枚举。HF 非 LFS 文件可能没有源端 SHA256，不能把本地计算值假称源端校验和；该导入缺口须另行解决。

这些本地登记命令由此版本新增，旧镜像和历史参考 commit 不含它们。运行前看 `course.py --help`，再选择确实包含接口的固定版本；本轮新增路径已通过离线 CPU 测试，但不冒充在旧镜像上完成了新接口的 GPU 端到端验收。普通官方客户端下载不要求升级课程包。`modelscope download` 不是外部文件的只读验证，可能重下并补缓存索引。显式本地复核使用：

```bash
python scripts/course.py verify-models
```

如果模型目录已存在但课程登记缺失，`generate` 会给出上述恢复命令；它不会静默重下权重。

`before` 表示训练前 Instruct；`student` 表示本 run 训练并导出的 Instruct。每步看实际输出，不把整段当忽略错误的批处理。生成后邀请抽读示范和配置，可以一起看、委托检查或继续；不以人工读满 10 条作为硬门槛。真实文件损坏、路径冲突和配置不兼容仍需修复。

如run_stage退出码为2，表示本次进程组需要强制回收，产物可能已写完，但“正常退出”尚未通过；检查environment内supervisor报告，不再次执行同名run覆盖结果。普通失败保留日志，用新run继续。

## B：显式使用已准备材料

网络数据下载不通时，可在尚未创建工作data目录的前提下使用：

```bash
python scripts/use_materials.py prepared
```

这一步替代prepare-data下载/筛选，可继续A的教师生成。若使用随附的500条教师示范，另建run：

```bash
python scripts/use_materials.py teacher500 --run instruct500-reference-01
python scripts/course.py make-config --run instruct500-reference-01
```

后续基线/训练/导出/评测命令将run改为`instruct500-reference-01`。导入教师示范不代表当前机器已完成数据生成；来源会写进run。模型仍需下载。不得混用新旧run的评测成绩。

## 正式测试

dev闭环稳定、前后三方dev评测完成后冻结。此处test300来自既有GSM8K划分，是教学用评测集，包含自动判分需要的参考答案，不是保密考试题库；该集合曾用于先前实验，报告中应说明。

```bash
python scripts/course.py freeze --run instruct500-new-01
python scripts/run_stage.py eval --run instruct500-new-01 --model before --split test
python scripts/run_stage.py eval --run instruct500-new-01 --model student --split test
python scripts/run_stage.py eval --run instruct500-new-01 --model teacher --split test
python scripts/course.py report --run instruct500-new-01
```

结果分开报告正常结束、格式和协议正确；整卡显存含vLLM缓存，吞吐不是单请求延迟，不预设蒸馏后同规模学生更快或更省显存。本实验规模为500条；更多数据的扩展应另建run并重新验证。

## 课后练习

扩展题源（10000 道经过清洗去重的数学题，含来源记录与哈希清单）在配套仓库：https://github.com/Beirana/distill-exercises 。可用于课后加练：把题源整理为本包 generate 流程可用的候选格式，另建新 run 生成新的教师示范并训练比较。

## 作者与贡献

* **齐睿**（[@Beirana](https://github.com/Beirana)）— 项目主要作者与维护者
* **金正一**（[@JIN-Zhy](https://github.com/JIN-Zhy)）— [`experiment-jin/`](experiment-jin/README.md) 内容作者与贡献者

## 来源与许可

源数据：AI-MO/NuminaMath-CoT 与 openai/gsm8k（两者均允许再分发、要求署名，具体条款以数据集官方页面为准）；`materials/` 中的筛选产物与教师示范为上述数据的衍生物，`teacher500` 含 Qwen2.5-7B-Instruct 的模型输出。模型权重从官方 ModelScope 仓库下载，本仓库不含任何模型权重。
