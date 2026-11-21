#!/usr/bin/env bash
# Coordinator, user request 2026-11-20: move Edge0's PLE table ple-fp8.gguf (51.2 GB) from the system disk
# (MSI M470 PRO 1TB, /) to /data (Crucial P3 Plus 4TB, its own NVMe) and A/B it. Runs ON the 5090 host.
#
#   bench/plemove.sh copy               direct-I/O copy to /data/edge0/ple-fp8.gguf, sha256 against the HF LFS oid;
#                                       first keeps a system-disk hardlink ~/models/.ple-rootdisk/ple-fp8.gguf
#   bench/plemove.sh switch data|root   point the checkpoint dirs' ple-fp8.gguf at /data (symlink) or back at the
#                                       system disk (hardlink to the backup). Refuses while a server is running.
#   bench/plemove.sh where              where each dir's ple-fp8.gguf resolves
#   bench/plemove.sh ab                 c1, page cache dropped before each server: vLLM PRESET=single root, data, data,
#                                       root (bench/gap-vllm.sh), then Strata root, data (bench/gap-strata.sh ref).
#                                       Ends on data unless vLLM's EOS pass is >2% slower there.
#   bench/plemove.sh all                wait for and take the GPU lock, copy, ab, write the coordinator note, release
#
# serve.sh mounts /data/edge0 at the same path, so the symlink resolves inside the vLLM container; Strata runs on
# the host and follows it directly. Log: logs/plemove.log.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BENCH=${BENCH:-$HOME/work/edge0-bench}
MODELS=$HOME/models
DIRS=(Edge0-145B-NVFP4 Edge0-145B-vllm Edge0-145B-vllm-L4)
BACKUP=$MODELS/.ple-rootdisk/ple-fp8.gguf
DST_DIR=/data/edge0
DST=$DST_DIR/ple-fp8.gguf
OID=c7615c59b060277ddf6f84729fa8f7469cfc82c7fb983d6e7eab5a2f7446f170   # HF LFS sha256 of ple-fp8.gguf
SIZE=51200246112
LOCK=$HOME/work/edge0-vllm-GPU_LOCK
NOTE=$HOME/work/edge0-vllm-COORDINATOR-NOTE.txt
LOG=$REPO/logs/plemove.log
mkdir -p "$REPO/logs"
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
servers_down() {
  ! docker ps --format '{{.Names}}' | grep -q '^edge0-vllm$' && ! tmux has-session -t edge0-serve 2>/dev/null &&
    ! pgrep -f "^$HOME/work/edge0/edge0-strata/build/strata" >/dev/null
}
drop_caches() { sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' && say "page cache dropped"; }

where() {
  for d in "${DIRS[@]}"; do
    local f=$MODELS/$d/ple-fp8.gguf
    [ -e "$f" ] && printf '%s -> %s (inode %s)\n' "$f" "$(readlink -f "$f")" "$(stat -L -c %i "$f")"
  done
}

backup() {
  [ -f "$BACKUP" ] && return 0
  local f=$MODELS/${DIRS[0]}/ple-fp8.gguf
  [ -L "$f" ] && { say "no system-disk file to keep ($f is already a symlink)"; return 1; }
  mkdir -p "$(dirname "$BACKUP")" && ln "$f" "$BACKUP" && say "system-disk hardlink kept: $BACKUP"
}

copy() {
  backup || return 1
  if [ -f "$DST" ] && [ -f "$DST.sha256-ok" ]; then say "verified copy already at $DST"; return 0; fi
  sudo -n mkdir -p "$DST_DIR" && sudo -n chown "$(id -u):$(id -g)" "$DST_DIR" || { say "cannot create $DST_DIR"; return 1; }
  say "copy start; /data free: $(df -h --output=avail /data | tail -1 | tr -d ' ')"
  local t0=$SECONDS
  dd if="$BACKUP" of="$DST.tmp" bs=64M iflag=direct oflag=direct status=none || { say "dd FAILED"; return 1; }
  say "copied in $((SECONDS - t0)) s"
  [ "$(stat -c %s "$DST.tmp")" = "$SIZE" ] || { say "SIZE MISMATCH"; return 1; }
  drop_caches; t0=$SECONDS
  local sum; sum=$(sha256sum "$DST.tmp" | cut -d' ' -f1)
  say "sha256 $sum ($((SECONDS - t0)) s)"
  [ "$sum" = "$OID" ] || { say "SHA256 MISMATCH, want $OID"; return 1; }
  chmod 0444 "$DST.tmp" && mv -f "$DST.tmp" "$DST" && date -Is >"$DST.sha256-ok"
  say "COPY_OK $DST"
}

switch() {   # $1 = data | root
  servers_down || { say "switch refused: a server is running"; return 1; }
  case "${1:-}" in
    data) [ -f "$DST.sha256-ok" ] || { say "switch refused: no verified copy at $DST"; return 1; } ;;
    root) [ -f "$BACKUP" ] || { say "switch refused: no system-disk file $BACKUP"; return 1; } ;;
    *) echo "usage: plemove.sh switch data|root" >&2; return 1 ;;
  esac
  for d in "${DIRS[@]}"; do
    local f=$MODELS/$d/ple-fp8.gguf
    [ -e "$f" ] || [ -L "$f" ] || continue
    # already there: mv refuses to rename one hard link of a file onto another ("are the same file")
    if [ "$1" = data ] && [ "$(readlink "$f")" = "$DST" ]; then continue; fi
    if [ "$1" = root ] && [ ! -L "$f" ] && [ "$f" -ef "$BACKUP" ]; then continue; fi
    if [ "$1" = data ]; then ln -sfn "$DST" "$f.swap"; else ln -f "$BACKUP" "$f.swap"; fi
    mv -Tf "$f.swap" "$f" || { say "switch FAILED at $f"; return 1; }
  done
  say "PLE -> $1"; where >>"$LOG"
}

eos_tok_s() {   # mean output tok/s of the given vllm-bench labels
  python3 - "$BENCH/results" "$@" <<'EOF'
import glob, json, os, sys
vals = []
for label in sys.argv[2:]:
    for f in glob.glob(os.path.join(sys.argv[1], label, "*.json")):
        try:
            vals.append(json.load(open(f))["output_throughput"])
        except (OSError, ValueError, KeyError):
            pass
print("%.1f" % (sum(vals) / len(vals)) if vals else "nan")
EOF
}

FAILED=0
vrun() {   # $1 = label, $2 = data | root
  switch "$2" || { FAILED=1; return 1; }
  drop_caches
  say "vLLM $1 on $2"
  DETAILED=1 bash "$REPO/bench/gap-vllm.sh" "$1" "PRESET=single" >>"$LOG" 2>&1 || say "gap-vllm $1 exit $?"
  [ -d "$BENCH/results/$1-eos" ] || { say "vLLM $1 produced no EOS result"; FAILED=1; }
  say "vLLM $1 done: EOS pass $(eos_tok_s "$1-eos") tok/s, 512 pass $(eos_tok_s "$1") tok/s"
}
srun() {   # $1 = label, $2 = data | root
  switch "$2" || return 1
  drop_caches
  say "Strata $1 on $2"
  DETAILED=1 bash "$REPO/bench/gap-strata.sh" ref "$1" >>"$LOG" 2>&1 || say "gap-strata $1 exit $?"
  say "Strata $1 done: $(eos_tok_s "$1") tok/s"
}

ab() {
  vrun pm-v-root1 root; vrun pm-v-data1 data; vrun pm-v-data2 data; vrun pm-v-root2 root
  srun pm-s-root root; srun pm-s-data data
  local vr vd sr sd final=data
  vr=$(eos_tok_s pm-v-root1-eos pm-v-root2-eos); vd=$(eos_tok_s pm-v-data1-eos pm-v-data2-eos)
  sr=$(eos_tok_s pm-s-root); sd=$(eos_tok_s pm-s-data)
  if [ "$FAILED" = 1 ] || python3 -c "import sys; sys.exit(0 if float('$vd') < 0.98 * float('$vr') else 1)" 2>/dev/null; then
    final=root
  fi
  switch "$final"
  python3 "$REPO/bench/summarize.py" pm-v-root1 pm-v-data1 pm-v-data2 pm-v-root2 >>"$LOG" 2>&1
  python3 "$REPO/bench/summarize.py" pm-v-root1-eos pm-v-data1-eos pm-v-data2-eos pm-v-root2-eos pm-s-root pm-s-data \
    >>"$LOG" 2>&1
  python3 "$REPO/tools/c1_compare.py" pm-s-root pm-s-data pm-v-root1-eos pm-v-data1-eos pm-v-data2-eos pm-v-root2-eos \
    --metrics "$REPO/results/gap" >>"$LOG" 2>&1
  RESULT="c1 EOS-stopping tok/s, PRESET=single: vLLM system disk $vr, /data $vd; Strata system disk $sr, /data $sd. Final: $final."
  say "RESULT $RESULT"
  FINAL=$final
}

all() {
  say "waiting for the GPU lock"
  until mkdir "$LOCK" 2>/dev/null; do sleep 5; done
  echo "coordinator-plemove $(date -Is): user-approved move of ple-fp8.gguf to /data, then an A/B (vLLM PRESET=single root/data/data/root, Strata root/data), about 60 min. NOT stale while tmux edge0-plemove exists: please wait for it. Afterwards read ~/work/edge0-vllm-COORDINATOR-NOTE.txt: the Strata c1 baseline is re-measured." \
    >"$LOCK/owner"
  say "GPU lock taken"
  local t0=$SECONDS
  until servers_down; do
    (( SECONDS - t0 > 600 )) && { say "a server is still running after 10 min; giving up"; rm -rf "$LOCK"; return 1; }
    sleep 10
  done
  if ! copy; then say "copy failed; nothing switched"; rm -rf "$LOCK"; return 1; fi
  ab
  cat >>"$NOTE" <<EOF

$(date '+%Y-%m-%d %H:%M') coordinator, user request: Edge0's PLE table ple-fp8.gguf was copied to /data (Crucial P3 Plus 4TB,
its own NVMe; sha256 = HF LFS oid) and A/B'd with bench/plemove.sh (logs/plemove.log, labels pm-v-*, pm-s-*).
$RESULT
Now on: $FINAL. On "data" the checkpoint dirs hold symlinks to /data/edge0/ple-fp8.gguf and serve.sh mounts /data/edge0
into the container; the system-disk file is kept as ~/models/.ple-rootdisk/ple-fp8.gguf.
"bench/plemove.sh switch root|data" flips all three dirs while no server runs.
For vLLM vs Strata comparisons from now on, use Strata measured on the same disk (pm-s-$FINAL), not cmp-s1 / cmp-s2.
EOF
  rm -rf "$LOCK"
  say "note written, GPU lock released"
}

case "${1:-}" in
  copy) copy ;;
  switch) switch "${2:-}" ;;
  where) where ;;
  ab) ab ;;
  all) all ;;
  *) sed -n '2,17p' "$0"; exit 1 ;;
esac
