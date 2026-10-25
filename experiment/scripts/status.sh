#!/usr/bin/env bash
# 主对比矩阵进度速览。在本地任意 shell 里调用（Windows 上用双引号，勿用单引号）：
#   ssh 智算中心_强化学习 "bash /root/ebgrpo/experiment/scripts/status.sh"
set -uo pipefail
cd /root/ebgrpo || exit 1

echo "服务器时间: $(date '+%F %T')"
# 只有一个 run 能真正在跑：按进程参数里的 experiment_name 精确认定，避免把
# 中途被 kill 掉、没写退出标记的历史 run 也标成"在跑"。
ACTIVE=$(pgrep -fa "verl.trainer.mai[n]_ppo" 2>/dev/null |
  grep -aoE "trainer.experiment_name=[A-Za-z0-9_]+" | head -1 | cut -d= -f2)
echo "当前在跑: ${ACTIVE:-（无）}"
echo

if tmux ls 2>/dev/null | grep -q matrix; then
  echo "矩阵会话: 在跑（tmux attach -t matrix 可看实时输出）"
else
  echo "矩阵会话: 已结束或未启动"
fi
echo "矩阵日志尾（只取进度/起止行，verl 的指标长行已过滤）:"
grep -aE "START GROUP|MATRIX_DONE|FAILED|Training Progress" matrix.log 2>/dev/null | tail -3 |
  sed -E 's/.*(START GROUP=[a-z]+ SEED=[0-9]).*/  \1/; s/.*Training Progress: *([0-9]+%\|[^ ]*\|[^]]*\]).*/  进度 \1/' |
  sed 's/^/  /'
echo

printf '%-26s %6s  %s\n' "RUN" "步数" "最新熵 / 控制量 / 状态"
for d in runs/*/; do
  name=$(basename "$d")
  [ "${name#_}" != "$name" ] && continue   # 跳过 _archive 等归档目录
  log="$d/eb_log.jsonl"
  if [ ! -f "$log" ]; then
    printf '%-26s %6s  %s\n' "$name" "-" "（无日志）"
    continue
  fi
  n=$(wc -l < "$log")
  # 完成判定以 train.sh 写在 console.log 末尾的 RUN_EXIT 为准：verl 收尾时偶尔抛
  # "DataLoader worker is killed by signal"，那是退出噪声（150 步已跑完），不是训练失败。
  state="在跑"
  if grep -aq "RUN_EXIT=0" "$d/console.log" 2>/dev/null; then
    state="完成"
  elif grep -aq "RUN_EXIT=" "$d/console.log" 2>/dev/null; then
    state="失败"
  elif [ "$name" = "$ACTIVE" ]; then
    state="在跑"
  else
    # 没有退出标记、进程也不在跑它：多半是中途被 kill（数据仍可用，按步数自行判断）
    state="已停"
  fi
  /opt/conda/envs/ebgrpo/bin/python - "$log" "$name" "$n" "$state" <<'PY'
import json, sys
log, name, n, state = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rows = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
if not rows:
    print(f"{name:<26} {n:>6}  {state}  （空）")
    raise SystemExit
r = rows[-1]
band = ""
if r.get("band_hi"):
    band = f" 带[{r['band_lo']:.2f},{r['band_hi']:.2f}]"
print(f"{name:<26} {n:>6}  {state:<6} H={r['entropy']:.3f}  u={r.get('u', 0):+.3f}"
      f"  coeff={r.get('entropy_coeff', 0):+.5f}{band}")
PY
done

echo
echo "=== 需要关注的报错（已跑完的 run 只剩收尾噪声，不计入）==="
flagged=""
for f in matrix.log runs/*/console.log; do
  case "$f" in */_*) continue ;; esac
  if grep -aq Traceback "$f" 2>/dev/null && ! grep -aq "RUN_EXIT=0" "$f" 2>/dev/null; then
    flagged="$flagged $f"
  fi
done
echo "疑似异常:${flagged:- 无}"
echo "GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
