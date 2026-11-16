#!/bin/bash
# The robot's side of the hardware-in-the-loop rig, run ON THE J4012.
#
# Everything the dog would carry runs here: go2_board's board_main (motors,
# safety, walking policy; cores 6-7), the Cyberdog brain and VAMOS (cores
# 0-5). The world -- physics, people, the lift, the Mid-360 -- runs on the
# Mac (go2_world), over the USB-C link. scripts/hil/run_hil.sh drives a whole
# run from the Mac; this is what it calls.
#
#   onboard.sh vamos                start VAMOS and wait until its model is loaded
#   onboard.sh board                start board_main (limp until the brain asks)
#   onboard.sh brain DEST [N]       run the brain to DEST, asking VAMOS for N paths (5)
#   onboard.sh stop [all]           stop the brain and the board (all: VAMOS too)
#   onboard.sh log                  board_main's log
#   onboard.sh status               what is running, VAMOS, the clock
#
# Settings (environment, defaults for the rig as set up on the J4012):
#   CYBERDOG GO2_BOARD GO2ENV       this repo, go2_board/ in it, and their venv
#   VAMOS_DIR VAMOS_ENV VAMOS_PORT  the VAMOS server, its venv and port
#   VAMOS_LOC_HEAD                  1 (default) = beam search over <loc> tokens + EOS only
#                                   (vlm_server.py --loc_head; loc_head.py); 0 = generate()
#   VAMOS_MODE                      shadow (default) = VAMOS is asked every frame and its
#                                   paths logged ($CYBERDOG/output/shadow_log.jsonl), the
#                                   map route drives, no VLM speed cap (up to MAX_V);
#                                   steer = VAMOS steers (--vamos), pace capped by its
#                                   latency (run_building.VLM_SEE_D, ~0.32 m/s on the J4012)
#   IFACE                           DDS interface to the world (USB-C: l4tbr0)
#   POLICY                          walking policy, relative to GO2_BOARD
#   STALE_MS                        go limp after this long without a state. 200 for
#                                   the Mac's world, which now and then pauses for
#                                   100+ ms; empty for board_main's own 50 ms, the
#                                   limit for anything with real motors
#   LIFT_HOST                       where the world takes lift calls (the Mac)
#   BOARD_CORES BRAIN_CORES         taskset lists
#   TRACE=1                         brain prints its commands, ~5 lines/s
#   BRAIN_ARGS                      more run_building arguments, e.g.
#                                   "--puddles 2 --floor-guard sim"
#   TRACK_LOG=1                     brain writes every track, every tick, to
#                                   $CYBERDOG/output/tracks.jsonl (--track-log)
#   CLOUD_LOG=1                     brain saves its scans to
#                                   $CYBERDOG/output/clouds.npz (--cloud-log)
#   POSE=lio                        board: report the Mid-360's odometry as the pose,
#                                   not the world's (board_main --pose lio)
#   LIO_RECORD=DIR                  board: save the raw Mid-360 frames, its IMU and
#                                   the bus pose to DIR (board_main --record)
set -u

CYBERDOG=${CYBERDOG:-$HOME/cyberdog_jetson}
GO2_BOARD=${GO2_BOARD:-$CYBERDOG/go2_board}
GO2ENV=${GO2ENV:-$HOME/go2env}
VAMOS_DIR=${VAMOS_DIR:-$CYBERDOG/vendor/VAMOS/server}   # synced with the repo (run_hil.sh sync)
VAMOS_ENV=${VAMOS_ENV:-$HOME/vamosenv}
VAMOS_PORT=${VAMOS_PORT:-8009}
VAMOS_LOC_HEAD=${VAMOS_LOC_HEAD:-1}
VAMOS_MODE=${VAMOS_MODE:-shadow}
IFACE=${IFACE:-l4tbr0}
POLICY=${POLICY:-policies/go2_flat_v5}
STALE_MS=${STALE_MS-200}
LIFT_HOST=${LIFT_HOST:-192.168.55.100}
BOARD_CORES=${BOARD_CORES:-6,7}
BRAIN_CORES=${BRAIN_CORES:-0-5}
TRACE=${TRACE:-0}
BOARD_LOG=$HOME/onboard_board.log
VAMOS_LOG=$HOME/vamos_server.log
HERE=$(cd "$(dirname "$0")" && pwd)

vamos_up() {
    curl -s --max-time 2 "http://127.0.0.1:$VAMOS_PORT/health" | grep -q '"model_loaded":true'
}

vamos_to_brain_cores() {
    for p in $(pgrep -f "[v]lm_server.py"); do taskset -a -p -c "$BRAIN_CORES" "$p" > /dev/null; done
}

case "${1:-}" in
  vamos)
    if vamos_up; then echo "VAMOS: already up"; exit 0; fi
    # Only the server goes in the background, with every fd off ssh's: a
    # backgrounded `cd && . && nohup ...` keeps a shell on ssh's stdout, and
    # the ssh that started it never returns.
    (cd "$VAMOS_DIR" || exit 1; . "$VAMOS_ENV/bin/activate"
     HF_HUB_OFFLINE=1 nohup taskset -c "$BRAIN_CORES" python vlm_server.py --host 127.0.0.1 \
         --port "$VAMOS_PORT" --model_path mateoguaman/vamos \
         $([ "$VAMOS_LOC_HEAD" = 1 ] && echo --loc_head) > "$VAMOS_LOG" 2>&1 < /dev/null &)
    for _ in $(seq 1 120); do
        vamos_up && { echo "VAMOS: up"; exit 0; }
        sleep 2
    done
    echo "VAMOS: not up after 240 s, see $VAMOS_LOG"; exit 1 ;;

  board)
    if pgrep -f "scripts/[b]oard_main.py" > /dev/null; then
        echo "board: already running (onboard.sh stop first)"; exit 1
    fi
    stale=()
    [ -n "$STALE_MS" ] && stale=(--stale-ms "$STALE_MS")
    [ -n "${LIO_RECORD:-}" ] && stale+=(--record "$LIO_RECORD")   # :- for set -u: unset is no recording
    [ -n "${POSE:-}" ] && stale+=(--pose "$POSE")
    rm -f "$BOARD_LOG"
    (cd "$GO2_BOARD" || exit 1; . "$GO2ENV/bin/activate"
     nohup taskset -c "$BOARD_CORES" python -u scripts/board_main.py --iface "$IFACE" \
         --policy "$POLICY" "${stale[@]}" > "$BOARD_LOG" 2>&1 < /dev/null &)
    for _ in $(seq 1 30); do
        grep -q "\[board\] up" "$BOARD_LOG" 2>/dev/null && break
        sleep 0.5
    done
    grep -v multicast "$BOARD_LOG"
    grep -q "\[board\] up" "$BOARD_LOG" || { echo "board: did not come up"; exit 1; }
    vamos_to_brain_cores ;;          # a VAMOS started before the board, off its cores

  brain)
    dest=${2:?"brain DEST [N]"}
    samples=${3:-5}
    cd "$CYBERDOG" && . "$GO2ENV/bin/activate" || exit 1
    if [ "$TRACE" = 1 ]; then run=("$HERE/trace_run_building.py"); else run=(-m cyberdog.sim.run_building); fi
    case "$VAMOS_MODE" in
        shadow) mode=(--shadow --shadow-log "$CYBERDOG/output/shadow_log.jsonl")
                mkdir -p "$CYBERDOG/output" ;;
        steer)  mode=(--vamos) ;;
        *)      echo "VAMOS_MODE: shadow or steer, not '$VAMOS_MODE'"; exit 1 ;;
    esac
    echo "VAMOS_MODE=$VAMOS_MODE"
    tracklog=()
    if [ "${TRACK_LOG:-0}" = 1 ]; then
        mkdir -p "$CYBERDOG/output" && rm -f "$CYBERDOG/output/tracks.jsonl"
        tracklog=(--track-log "$CYBERDOG/output/tracks.jsonl")
    fi
    if [ "${CLOUD_LOG:-0}" = 1 ]; then
        mkdir -p "$CYBERDOG/output" && rm -f "$CYBERDOG/output/clouds.npz"
        tracklog+=(--cloud-log "$CYBERDOG/output/clouds.npz")
    fi
    # MUJOCO_GL=egl: the camera view VAMOS is shown is rendered with no display
    # (23 ms a frame); on the dog it comes from a USB camera.
    MUJOCO_GL=egl exec taskset -c "$BRAIN_CORES" python -u "${run[@]}" "$dest" --robot go2_board \
        --lift-host "$LIFT_HOST" "${mode[@]}" --vamos-samples "$samples" --auto-confirm --no-video "${tracklog[@]}" \
        ${BRAIN_ARGS:-} ;;

  stop)
    # [x]yz: matches the process, not a shell whose command line holds the pattern
    pkill -f "scripts/[b]oard_main.py"
    pkill -f "cyberdog.sim.[r]un_building"
    pkill -f "[t]race_run_building.py"
    [ "${2:-}" = all ] && pkill -f "[v]lm_server.py"
    sleep 1
    pgrep -af "[b]oard_main|[r]un_building|[v]lm_server" || echo "stopped" ;;

  log)
    grep -v multicast "$BOARD_LOG" ;;

  status)
    pgrep -af "[b]oard_main|[r]un_building|[v]lm_server" || echo "nothing running"
    if vamos_up; then echo "VAMOS: up"; else echo "VAMOS: down"; fi
    echo "clock: $(date -u '+%Y-%m-%d %H:%M:%S UTC') (no RTC: wrong after a boot until set)"
    nvpmodel -q 2>/dev/null | head -1 ;;

  *)
    sed -n '2,30p' "$0"; exit 2 ;;
esac
