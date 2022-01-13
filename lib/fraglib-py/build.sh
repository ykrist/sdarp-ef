#!/bin/bash
set -e
source common.sh

function usage_and_exit {
    echo "usage: build.sh (release|debug)" >&2;
    exit $1
}

LIB_FILENAME=libfraglib_py.so

PROFILE=${1:-release}
case $PROFILE in
    release)
    TARGET_LIB=target/release/$LIB_FILENAME
    CARGO_BUILD_ARGS=( --release )
    ;;
    debug)
    TARGET_LIB=target/debug/$LIB_FILENAME
    CARGO_BUILD_ARGS=( --features logging )
    ;;
    *)
    usage_and_exit 1
    ;;
esac

start_msg
echo "building in $PROFILE mode..."
echo_and_run cargo build "${CARGO_BUILD_ARGS[@]}"
echo_and_run cp $TARGET_LIB $PYTHON_SHARED_LIB
success_msg
