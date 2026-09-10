#!/usr/bin/env bash
# 一键环境核查脚本（只读，不安装任何东西；依赖解析仅在 --plan 时以 dry-run 运行）
# 用法:
#   bash scripts/check_host.sh            # 快速核查（项目/主机/网络/磁盘预算/目录布局）
#   bash scripts/check_host.sh --plan     # 追加 gen/train 依赖解析 dry-run（较慢，几分钟）
# 退出码: 0=无 FAIL；1=存在 FAIL（WARN 不影响退出码）
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${COURSE_DATA_ROOT:-/root/distill-work}"
DATA_DISK_DIR="${DATA_DISK_DIR:-/root/autodl-tmp}"
PYTHON="${PYTHON:-python}"
PLAN=0
[ "${1:-}" = "--plan" ] && PLAN=1

PASS=0; WARN=0; FAIL=0
ok()   { echo "  [PASS] $*"; PASS=$((PASS+1)); }
warn() { echo "  [WARN] $*"; WARN=$((WARN+1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
line() { echo; echo "======== $* ========"; }

line "1. 项目完整性"
if "$PYTHON" "$PROJ/scripts/verify_bundle.py" >/tmp/check_bundle.log 2>&1; then
  ok "verify_bundle.py 通过（材料与清单一致）"
else
  bad "verify_bundle.py 失败: $(tail -3 /tmp/check_bundle.log)"
fi

line "2. Python / Torch / GPU"
PY_VER="$($PYTHON -c 'import sys; print(".".join(map(str,sys.version_info[:2])))' 2>/dev/null)"
PY_PREFIX="$($PYTHON -c 'import sys; print(sys.prefix)' 2>/dev/null)"
echo "  python: $PY_VER  prefix: $PY_PREFIX"
[ "$PY_VER" = "3.12" ] && ok "Python 3.12 与镜像预期一致" || warn "Python 非 3.12（文档按 3.12 验证，其他版本未验证）"
TORCH_INFO="$($PYTHON - <<'EOF' 2>&1
import torch
info = f"{torch.__version__}|{torch.version.cuda}|{torch.cuda.is_available()}"
if torch.cuda.is_available():
    info += f"|{torch.cuda.get_device_name(0)}|{torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GiB"
    x = torch.randn(64, 64, device="cuda"); (x @ x).sum().item()
    info += "|gpu_tensor_ok"
print(info)
EOF
)"
IFS='|' read -r TORCH_VER TORCH_CUDA CUDA_OK GPU_NAME GPU_MEM GPU_TENSOR <<<"$TORCH_INFO"
echo "  torch: $TORCH_VER  cuda: $TORCH_CUDA"
[ "$CUDA_OK" = "True" ] && ok "CUDA 可用: $GPU_NAME $GPU_MEM" || bad "CUDA 不可用，无法继续实验"
[ "$GPU_TENSOR" = "gpu_tensor_ok" ] && ok "GPU 张量运算通过" || bad "GPU 张量运算失败: $TORCH_INFO"

line "3. GPU 占用 / 内存"
GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
GPU_PROC="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
if [ "${GPU_USED:-9999}" -le 100 ] && [ "${GPU_PROC:-1}" -eq 0 ]; then
  ok "GPU 空闲（已用 ${GPU_USED}MiB，无计算进程）"
else
  warn "GPU 非空闲（已用 ${GPU_USED}MiB，计算进程 ${GPU_PROC} 个）——加载新模型前需确认上一推理进程退出"
fi
MEM_AVAIL_GB="$(free -g | awk '/^Mem:/{print $7}')"
[ "${MEM_AVAIL_GB:-0}" -ge 30 ] && ok "内存可用 ${MEM_AVAIL_GB}GB" || warn "内存可用仅 ${MEM_AVAIL_GB}GB"

line "4. 磁盘与预算"
SYS_FREE_GB="$(df -BG --output=avail / | tail -1 | tr -dc 0-9)"
echo "  系统盘 / 可用: ${SYS_FREE_GB}GB"
DATA_DISK_FREE_GB=""
if [ -d "$DATA_DISK_DIR" ]; then
  DATA_DISK_FREE_GB="$(df -BG --output=avail "$DATA_DISK_DIR" | tail -1 | tr -dc 0-9)"
  echo "  数据盘 $DATA_DISK_DIR 可用: ${DATA_DISK_FREE_GB}GB"
  TMP_PROBE="$DATA_DISK_DIR/.write_probe_$$"
  if touch "$TMP_PROBE" 2>/dev/null; then rm -f "$TMP_PROBE"; ok "数据盘可写"; else bad "数据盘不可写"; fi
else
  warn "未找到数据盘目录 $DATA_DISK_DIR"
fi
# 预算: 系统盘需容纳 依赖~3GB + merged~1GB/run + run日志/数据；数据盘需容纳 模型~17GB + 下载缓存峰值~18GB
[ "$SYS_FREE_GB" -ge 8 ] && ok "系统盘预算足够（依赖+实验产物 约4–5GB，现有 ${SYS_FREE_GB}GB）" || warn "系统盘可用不足 8GB，依赖安装后可能吃紧"
if [ -n "$DATA_DISK_FREE_GB" ]; then
  [ "$DATA_DISK_FREE_GB" -ge 38 ] && ok "数据盘预算足够（模型~17GB+缓存峰值~18GB，现有 ${DATA_DISK_FREE_GB}GB）" \
    || warn "数据盘可用不足 38GB（模型~17GB+缓存峰值~18GB），当前 ${DATA_DISK_FREE_GB}GB，需先核算再下载"
fi

line "5. 工作目录布局（数据盘不进镜像，只允许 模型/缓存；实验产物必须留在系统盘）"
DATA_DISK_TARGET="$(df --output=target "$DATA_DISK_DIR" 2>/dev/null | tail -1)"
on_data_disk() {  # df 会先解析符号链接，直接比较所在文件系统挂载点
  local p="${1:-}"; [ -e "$p" ] || return 2
  [ "$(df --output=target "$p" 2>/dev/null | tail -1)" = "$DATA_DISK_TARGET" ]
}
if [ -d "$WORK" ]; then
  for sub in models cache data runs environment events reference; do
    if [ -e "$WORK/$sub" ]; then
      if on_data_disk "$WORK/$sub"; then loc="数据盘"; else loc="系统盘"; fi
      [ -L "$WORK/$sub" ] && loc="$loc(符号链接→$(readlink "$WORK/$sub"))"
      echo "  $WORK/$sub → $loc"
    fi
  done
  if [ -e "$WORK/runs" ] && on_data_disk "$WORK/runs"; then
    bad "runs/（生成的500条、adapter、评测结果）在数据盘上，不进镜像，必须迁回系统盘"
  elif [ -e "$WORK/runs" ]; then ok "runs/ 实验产物在系统盘，会进镜像"; fi
  if [ -e "$WORK/data" ] && on_data_disk "$WORK/data"; then
    bad "data/（筛选后数据）在数据盘上，必须迁回系统盘"
  elif [ -e "$WORK/data" ]; then ok "data/ 在系统盘，会进镜像"; fi
  if [ -e "$WORK/models" ] && ! on_data_disk "$WORK/models" && [ "$SYS_FREE_GB" -lt 20 ]; then
    warn "models/ 在系统盘且系统盘余量 <20GB：教师15GB+学生1GB放系统盘会很紧；模型可重新下载，建议移到数据盘"
  fi
else
  echo "  $WORK 尚未创建。推荐布局："
  echo "    系统盘(进镜像): $WORK/{data,runs,environment,events,reference}   # 实验产物，体积小、不可再生"
  echo "    数据盘(不进镜像): $WORK/{models,cache}                            # 可再下载/可重建"
  echo "  创建方式: mkdir -p $DATA_DISK_DIR/distill-assets/{models,cache} && mkdir -p $WORK"
  echo "            ln -s $DATA_DISK_DIR/distill-assets/models $WORK/models"
  echo "            ln -s $DATA_DISK_DIR/distill-assets/cache  $WORK/cache"
fi

line "6. 已装包 / 依赖一致性"
if pip check 2>&1 | grep -q "No broken requirements found"; then
  ok "pip check 无损坏依赖"
else
  warn "pip check 报告问题: $(pip check 2>&1 | head -3)"
fi
TRAIN_PY="${COURSE_TRAIN_ENV:-/root/train-env}/bin/python"
[ -x "$TRAIN_PY" ] || TRAIN_PY=""
for pkg in vllm transformers datasets modelscope peft llamafactory; do
  ver="$(pip show "$pkg" 2>/dev/null | awk '/^Version:/{print $2}')"
  [ -z "$ver" ] && ver="$(pip show "${pkg//-/_}" 2>/dev/null | awk '/^Version:/{print $2}')"
  env_tag="base"
  if [ -z "$ver" ] && [ -n "$TRAIN_PY" ]; then
    ver="$("$TRAIN_PY" -m pip show "$pkg" 2>/dev/null | awk '/^Version:/{print $2}')"
    [ -n "$ver" ] && env_tag="train-env"
  fi
  if [ -n "$ver" ]; then
    echo "  [已装] $pkg $ver ($env_tag)"
  else
    echo "  [未装] $pkg   # P1 增量安装解决"
  fi
done
if [ -n "$TRAIN_PY" ] && "$TRAIN_PY" -m pip check 2>/dev/null | grep -q "No broken requirements found"; then
  ok "训练环境($TRAIN_PY) pip check 无损坏依赖"
elif [ -n "$TRAIN_PY" ]; then
  warn "训练环境($TRAIN_PY) pip check 有问题"
fi

line "7. 网络入口"
net() {  # net <名称> <url> <期望code前缀>
  local code; code="$(timeout 15 curl -sIL -o /dev/null -w '%{http_code}' "$2" 2>/dev/null)"
  case "$code" in "$3"*) ok "$1 可达 (HTTP $code)";; 000) bad "$1 不可达（超时/拒连）";; *) warn "$1 返回 HTTP $code";; esac
}
net "ModelScope(模型权重)" "https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/resolve/master/config.json" "2"
net "PyPI镜像(依赖安装)"  "http://mirrors.aliyun.com/pypi/simple/" "2"
code_hf="$(timeout 10 curl -sI -o /dev/null -w '%{http_code}' https://huggingface.co 2>/dev/null)"
if [ -z "$code_hf" ] || [ "$code_hf" = "000" ]; then
  warn "huggingface.co 直连不可达 → prepare-data 需设 HF_ENDPOINT=https://hf-mirror.com，或改用 use_materials.py prepared"
  net "hf-mirror(数据集备援)" "https://hf-mirror.com/datasets/AI-MO/NuminaMath-CoT/resolve/9d8d210c9f6a36c8f3cd84045668c9b7800ef517/README.md" "2"
else
  ok "huggingface.co 可达 (HTTP $code_hf)"
fi

if [ "$PLAN" = "1" ]; then
  line "8. 依赖解析 dry-run（PLAN ONLY，不安装）"
  GEN_LOG=/tmp/check_plan_gen.log; TRAIN_LOG=/tmp/check_plan_train.log
  if "$PYTHON" "$PROJ/scripts/setup_env.py" gen >"$GEN_LOG" 2>&1; then
    if grep -qE "Would install|Planned changes" "$GEN_LOG"; then
      ok "gen 计划可解析（全 wheel，无源码编译）"
      if grep -qE "Would install.*[ (]torch-" "$GEN_LOG"; then bad "gen 计划试图替换 Torch，需先排查"; else ok "Torch 未被 gen 计划替换"; fi
    else ok "gen 依赖已全部满足，无需安装"; fi
  else bad "gen 计划解析失败: $(tail -3 "$GEN_LOG")"; fi
  if "$PYTHON" "$PROJ/scripts/setup_env.py" train >"$TRAIN_LOG" 2>&1; then
    ok "train 计划可解析（LLaMA-Factory 源码仅用于解析，未装 pip 包）"
  else bad "train 计划解析失败: $(tail -3 "$TRAIN_LOG")"; fi
  # 两计划共享包版本是否打架（决定是否需要隔离训练环境）
  CONFLICT_PKGS="$(python3 - "$GEN_LOG" "$TRAIN_LOG" <<'EOF'
import re, sys
def planned(p):
    t = open(p, encoding="utf-8", errors="replace").read()
    m = re.search(r"^Would install (.+)$", t, re.M)
    return dict(x.rsplit("-", 1) for x in m.group(1).split()) if m else {}
g, t = planned(sys.argv[1]), planned(sys.argv[2])
diff = sorted(f"{k}: gen→{g[k]} / train→{t[k]}" for k in g.keys() & t.keys() if g[k] != t[k])
print("\n".join(diff))
EOF
)"
  if [ -n "$CONFLICT_PKGS" ]; then
    warn "gen/train 对以下共享包要求不同版本，同环境先后安装会互相改写（按文档应隔离训练环境，设 COURSE_TRAIN_ENV）:"
    echo "$CONFLICT_PKGS" | sed 's/^/    /'
  else ok "gen/train 无共享包版本冲突（可与 plan 结果为准，再人工确认）"; fi
fi

line "核查汇总"
echo "  PASS=$PASS  WARN=$WARN  FAIL=$FAIL"
[ "$FAIL" -eq 0 ] || { echo "  存在 FAIL，先处理后再进入 P1"; exit 1; }
echo "  核查通过。下一步: P1 增量安装（看清 plan 后 --apply），安装后重跑本脚本确认 vllm/transformers 等已装且 Torch 未被替换。"
exit 0
