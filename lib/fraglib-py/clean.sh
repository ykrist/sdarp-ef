#!/bin/bash
source common.sh

start_msg
echo_and_run cargo clean
echo_and_run rm -f $PYTHON_SHARED_LIB
success_msg
