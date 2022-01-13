PYTHON_SHARED_LIB=python/fraglibpy.so

# -------------------------------
function echo_and_run {
    echo "[cmd] ${@}"
    eval ${@}
}

function strpad100 {
    S="${*}"
    N=${#S}
    if (( $N % 2 )) ; then
        EXTRA="-"
    else
        EXTRA=""
    fi
    N=$((100 - N - 2))
    N=$((N/2))
    DASHES=`printf %${N}s | tr " " "-"`

    echo "${DASHES} ${S} ${EXTRA}${DASHES}"

}

function start_msg {
    strpad100 "start: " `date '+%T'`
}

function success_msg {
    strpad100 "completed:" `date '+%T'`
}
