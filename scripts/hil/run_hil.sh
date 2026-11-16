#!/bin/bash
# The hardware-in-the-loop rig, driven from the Mac: the world here (go2_world:
# physics, people, the lift, the Mid-360), everything the dog carries on the
# J4012 over USB-C (scripts/hil/onboard.sh there). See scripts/hil/README.md.
#
#   bash scripts/hil/run_hil.sh sync              copy this repo (brain and go2_board/), as checked out, to the board
#   bash scripts/hil/run_hil.sh clock             set the board's clock from this Mac (sudo there)
#   bash scripts/hil/run_hil.sh walk              stand, walk 4 s, stand: is the board's loop sound? (~1 min)
#   bash scripts/hil/run_hil.sh route DEST [N]    the whole route to DEST, VAMOS asked for N paths (5)
#   bash scripts/hil/run_hil.sh stop              stop the world here and the board and brain there
#
# Each run writes its logs to $OUT (output/hil/<time>-<what>) and ends with a
# summary: the brain's outcome, how regularly states and commands moved, and
# a route's truth from the world ("world truth:": collisions and people, on the
# physics itself; go2_world/score.py).
# The world, the LiDAR and the board are stopped at the end, or on Ctrl-C;
# VAMOS is left up (it takes ~25 s to load): `onboard.sh stop all` there stops it.
#
# Settings (environment):
#   BOARD        ssh target of the J4012            (jittarin@192.168.55.1)
#   MAC_IF       this Mac's interface on the link   (en9)
#   MAC_IP       this Mac's address on the link     (192.168.55.100)
#   BOARD_IF     the board's interface on the link  (l4tbr0)
#   GO2_WORLD    the go2_world checkout             (~/Work/unitree/go2_world)
#   REMOTE       this repo on the board, under ~    (cyberdog_jetson)
#   PY           Python with go2_world's packages   (python)
#   PEOPLE       people walking in the world        (4)
#   WORLD_ARGS   more run_world arguments, e.g. --held-pd
#   LIDAR_ARGS   more run_lidar arguments, e.g. --imu-noise (a real unit's IMU)
#   VIEWER=1     the world with MuJoCo's viewer (mjpython next to PY), to watch or
#                screen-record a run; it can hold the physics long enough to trip
#                the board's stale-state guard, so headless is the default
#   RECORD=1     go2_world records the run to $OUT/world.mp4 (offscreen, chase camera;
#                needs go2_world's [record] extra in PY's environment)
#   TRACE=1      the brain prints its commands
#   VAMOS_MODE   route: shadow (default: VAMOS logged, the map drives, no VLM speed
#                cap; its log copied to $OUT/shadow_log.jsonl) or steer (VAMOS steers,
#                pace capped by its latency). See onboard.sh.
#   BRAIN_ARGS   route: more run_building arguments, e.g. "--puddles 2 --floor-guard sim"
#   TRACK_LOG=1  route: the brain logs every track every tick; copied to $OUT/tracks.jsonl
#   CLOUD_LOG=1  route: the brain saves its scans (5 a second); copied to $OUT/clouds.npz
#   POSE         route: where the board's pose comes from              (lio)
#                lio: the Mid-360's odometry (small_gicp + gyro, needs small_gicp in
#                ~/go2env), as the real dog has it; its error vs the world is in the
#                board log. sim: the world's true pose. walk always uses sim
#   DENSE        route: the Mid-360 at a real unit's point rate (run_lidar --dense,
#                ~20k points a sweep, not ~2k)   (1 with POSE=lio, else 0)
#                lio on the sparse pattern drifted 4x more (11-12: 1.02 m vs 0.26 m max)
#   LIO_RECORD=1 route: the board saves the raw Mid-360 frames, its IMU and the world's
#                true pose (go2_board.record); copied to $OUT/lio/. With
#                LIDAR_ARGS=--dense for the real unit's point rate
#   ROUTE_S      a route is stopped after this long, in seconds  (560)
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
CYBERDOG=$(cd "$HERE/../.." && pwd)
BOARD=${BOARD:-jittarin@192.168.55.1}
MAC_IF=${MAC_IF:-en9}
MAC_IP=${MAC_IP:-192.168.55.100}
BOARD_IF=${BOARD_IF:-l4tbr0}
GO2_WORLD=${GO2_WORLD:-$HOME/Work/unitree/go2_world}
REMOTE=${REMOTE:-cyberdog_jetson}
PY=${PY:-python}
PEOPLE=${PEOPLE:-4}
WORLD_ARGS=${WORLD_ARGS:-}
LIDAR_ARGS=${LIDAR_ARGS:-}
DENSE=${DENSE:-}
VIEWER=${VIEWER:-0}
RECORD=${RECORD:-0}
TRACE=${TRACE:-0}
TRACK_LOG=${TRACK_LOG:-0}
VAMOS_MODE=${VAMOS_MODE:-shadow}
CLOUD_LOG=${CLOUD_LOG:-0}
LIO_RECORD=${LIO_RECORD:-0}
POSE=${POSE:-lio}
LIO_DIR=lio_record                 # on the board, under ~
ONBOARD="~/$REMOTE/scripts/hil/onboard.sh"
BOARD_PY='. ~/go2env/bin/activate; python'

on_board() { ssh -o BatchMode=yes -o ConnectTimeout=8 "$BOARD" "$@"; }

new_out() {
    OUT=${OUT:-$CYBERDOG/output/hil/$(date +%Y%m%d-%H%M%S)-$1}
    mkdir -p "$OUT"
    echo "logs: $OUT"
}

clock_offset() {    # board minus Mac, whole seconds
    local b; b=$(on_board 'date -u +%s') || return 1
    echo $(( b - $(date -u +%s) ))
}

check_clock() {
    local off; off=$(clock_offset) || { echo "board: unreachable at $BOARD"; exit 1; }
    if [ "${off#-}" -gt 2 ]; then
        echo "WARNING: the board's clock is ${off} s off this Mac's (it has no RTC). The rig runs"
        echo "         on monotonic clocks, so this only spoils log times: run_hil.sh clock fixes it."
    fi
}

# Background jobs: only the program goes in the background, with every fd off
# ours -- a backgrounded `cd && prog` keeps a shell holding our stdout, and
# whatever reads it (a pipe, ssh) waits for it to end. And [x]yz patterns, so
# pkill -f matches the process and not a shell whose command line holds it.
stop_world() {
    pkill -f "scripts/[r]un_world.py"; pkill -f "scripts/[r]un_lidar.py"
    # A recording world finishes its MP4 on SIGTERM: give it a moment.
    for _ in 1 2 3 4 5 6; do pgrep -f "scripts/[r]un_world.py" > /dev/null || break; sleep 0.5; done
}

start_world() {
    stop_world
    on_board "$ONBOARD stop" > /dev/null 2>&1
    sleep 1
    # shellcheck disable=SC2086
    local world=("$PY" -u scripts/run_world.py --headless)
    if [ "$VIEWER" = 1 ]; then
        # The viewer needs mjpython on macOS: the one in PY's environment.
        local mjpy; mjpy="$(dirname "$(command -v "$PY")")/mjpython"
        [ -x "$mjpy" ] || { echo "VIEWER=1: no mjpython next to $PY"; exit 1; }
        world=("$mjpy" -u scripts/run_world.py)
        echo "world: with the viewer (double-click the dog, then Ctrl+right double-click to follow it)"
    fi
    [ "$RECORD" = 1 ] && world+=(--record="$OUT/world.mp4")
    (cd "$GO2_WORLD" || exit 1; nohup "${world[@]}" --people="$PEOPLE" \
        --lift-listen="$MAC_IP" $WORLD_ARGS "$MAC_IF" > "$OUT/world.log" 2>&1 < /dev/null &)
    for _ in $(seq 1 30); do grep -q "people walking" "$OUT/world.log" 2>/dev/null && break; sleep 0.5; done
    grep -q "people walking" "$OUT/world.log" || { echo "world: did not start, see $OUT/world.log"; exit 1; }
    grep -E "^world:|^lift:" "$OUT/world.log"
    (cd "$GO2_WORLD" || exit 1; nohup "$PY" -u scripts/run_lidar.py --interface "$MAC_IF" \
        --host "${BOARD#*@}" $LIDAR_ARGS > "$OUT/lidar.log" 2>&1 < /dev/null &)
    sleep 13                                   # the emulator loads the scene, then streams
    grep "^mid360:" "$OUT/lidar.log" | head -2 | grep . || { echo "lidar: not streaming, see $OUT/lidar.log"; exit 1; }
}

start_gaps() {
    on_board "($BOARD_PY ~/$REMOTE/scripts/hil/dds_gaps.py 3600 $BOARD_IF > ~/hil_gaps.log 2>&1 < /dev/null &)"
}

show_gaps() {
    on_board 'pkill -TERM -f "[d]ds_gaps.py"; for i in $(seq 1 20); do grep -q lowcmd ~/hil_gaps.log && break; sleep 0.5; done; cat ~/hil_gaps.log' \
        | grep -v multicast | tee "$OUT/gaps.log"
}

stop_all() {
    stop_world
    on_board "pkill -f '[d]ds_gaps.py'; $ONBOARD stop" > /dev/null 2>&1
}

case "${1:-}" in
  sync)
    # What is checked out is what goes: say which branch, and ask. A sync
    # from the wrong checkout put an old go2_board on the board once, and
    # nothing said so until its walk fell; the brain and the board are one
    # repo now, so they can no longer be on different commits.
    dirty=$(git -C "$CYBERDOG" status --porcelain --untracked-files=no | wc -l | tr -d ' ')
    printf '  %s %s @ %s%s\n' "$(basename "$CYBERDOG")" "$(git -C "$CYBERDOG" branch --show-current)" \
        "$(git -C "$CYBERDOG" log -1 --format='%h %s' | cut -c1-60)" \
        "$([ "$dirty" != 0 ] && echo "  (+$dirty uncommitted)")"
    if [ -t 0 ]; then
        read -r -p "copy these to $BOARD? [y/N] " yes
        [ "$yes" = y ] || { echo "not synced"; exit 1; }
    elif [ "${SYNC_YES:-0}" != 1 ]; then
        echo "not synced: no terminal to ask; SYNC_YES=1 to sync anyway"; exit 1
    fi
    # Of vendor/VAMOS only the server goes: onboard.sh runs VAMOS from it, so the
    # board's server is the one in this repo (it was a hand-kept copy in ~/vamos).
    rsync -a --exclude __pycache__ --exclude .pytest_cache --exclude '*.egg-info' --exclude .git \
        --exclude /models --exclude /output --exclude /datasets \
        --include '/vendor/VAMOS/server/***' --exclude '/vendor/VAMOS/*' \
        --exclude '*.docx' --exclude '*.pptx' --exclude /vamos_train.tgz \
        "$CYBERDOG/" "$BOARD:~/$REMOTE/" && echo "$(basename "$CYBERDOG") -> $BOARD:~/$REMOTE" ;;

  clock)
    # A sudo password prompt needs a terminal: run this one yourself, not from a script.
    ssh -t "$BOARD" "sudo date -u -s @$(date -u +%s) > /dev/null" && echo "board clock off by $(clock_offset) s" ;;

  walk)
    new_out walk; check_clock
    trap stop_all EXIT
    start_world
    on_board "$ONBOARD board" | tail -1
    start_gaps
    on_board "cd ~/$REMOTE/go2_board && $BOARD_PY -u scripts/brain_cli.py stand:4 walk:4:0.54,0,0.47 stand:2" \
        > "$OUT/brain_cli.log" 2>&1
    grep "t=" "$OUT/brain_cli.log" | sed 's/  */ /g' | tail -n +8
    show_gaps
    if grep -q "tripped=tilted" "$OUT/brain_cli.log"; then echo "RESULT: FELL"; exit 1; fi
    echo "RESULT: walked" ;;

  route)
    dest=${2:?"route DEST [N]"}
    new_out route; check_clock
    case "$POSE" in
      lio) on_board "$BOARD_PY -c 'import small_gicp'" > /dev/null 2>&1 || {
               echo "POSE=lio (the default): small_gicp is not in ~/go2env on the board"
               echo "  install its wheel there (HANDOFF / scripts/hil/README.md), or run with POSE=sim"
               exit 1; } ;;
      sim) ;;
      *) echo "POSE=$POSE: lio or sim"; exit 2 ;;
    esac
    # The odometry is only as good as the scan it matches: dense, as a real
    # Mid-360 is, unless asked otherwise.
    [ -z "$DENSE" ] && { [ "$POSE" = lio ] && DENSE=1 || DENSE=0; }
    if [ "$DENSE" = 1 ]; then
        case " $LIDAR_ARGS " in *" --dense "*) ;; *) LIDAR_ARGS="$LIDAR_ARGS --dense" ;; esac
    elif [ "$POSE" = lio ]; then
        echo "WARNING: POSE=lio on the sparse LiDAR pattern (DENSE=0): expect ~1 m of drift, not ~0.3"
    fi
    echo "pose: $POSE, LiDAR $([ "$DENSE" = 1 ] && echo dense || echo sparse)"
    trap stop_all EXIT
    start_world
    on_board "$ONBOARD vamos"
    if [ "$LIO_RECORD" = 1 ]; then
        on_board "rm -rf ~/$LIO_DIR; POSE=$POSE LIO_RECORD=~/$LIO_DIR $ONBOARD board" | tail -2
    else
        on_board "POSE=$POSE $ONBOARD board" | tail -1
    fi
    start_gaps
    on_board "BRAIN_ARGS='${BRAIN_ARGS:-}' VAMOS_MODE=$VAMOS_MODE TRACE=$TRACE TRACK_LOG=$TRACK_LOG CLOUD_LOG=$CLOUD_LOG timeout ${ROUTE_S:-560} $ONBOARD brain '$dest' ${3:-5}" > "$OUT/brain.log" 2>&1
    grep -E "VAMOS_MODE|ARRIVED|FAILED|HALTED|leg |obstacles|collisions|VAMOS floor|VLM ASYNC|VLM SPEED CAP|SHADOW|FLOOR GUARD|floor guard|puddles|help|planner|\[udp\]" \
        "$OUT/brain.log" | awk '!seen[$0]++' | cut -c1-230
    show_gaps
    # The board first (it prints the odometry's error and how it stopped on its
    # way out), then the world, which writes the run's truth as it goes
    # (go2_world/score.py): what the dog touched and how near it came to people,
    # on the real physics -- the brain scores only on the board's pose, and has
    # no people of its own here. World first left the board without states.
    on_board "$ONBOARD stop" > /dev/null 2>&1
    on_board "$ONBOARD log" | tail -n +3
    stop_world
    grep -E "^world: t=|^world truth:" "$OUT/world.log" | cut -c1-230
    if [ "$TRACK_LOG" = 1 ]; then
        scp -q -o BatchMode=yes "$BOARD:$REMOTE/output/tracks.jsonl" "$OUT/tracks.jsonl" \
            && echo "tracks: $OUT/tracks.jsonl" || echo "tracks: not copied"
    fi
    if [ "$CLOUD_LOG" = 1 ]; then
        scp -q -o BatchMode=yes "$BOARD:$REMOTE/output/clouds.npz" "$OUT/clouds.npz" \
            && echo "clouds: $OUT/clouds.npz" || echo "clouds: not copied"
    fi
    if [ "$VAMOS_MODE" = shadow ]; then
        scp -q -o BatchMode=yes "$BOARD:$REMOTE/output/shadow_log.jsonl" "$OUT/shadow_log.jsonl" \
            && echo "shadow: $OUT/shadow_log.jsonl" || echo "shadow: not copied"
    fi
    if [ "$LIO_RECORD" = 1 ]; then
        on_board "$ONBOARD stop" > /dev/null 2>&1      # board_main writes its last chunk on exit
        rsync -a -e "ssh -o BatchMode=yes" "$BOARD:$LIO_DIR/" "$OUT/lio/" \
            && echo "lio: $OUT/lio ($(ls "$OUT/lio" | grep -c chunk_) chunks, $(du -sh "$OUT/lio" | cut -f1))" \
            || echo "lio: not copied"
    fi
    grep -q "^ARRIVED" "$OUT/brain.log" ;;

  stop)
    stop_all; echo "stopped (VAMOS left up)" ;;

  *)
    sed -n '2,28p' "$0"; exit 2 ;;
esac
