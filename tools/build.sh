#!/bin/sh

set -e

POS="${0%/*}"; test "$POS" = "$0" && POS=.
OS="$(uname)"
if [ "$OS" = "Darwin" ]; then
    # https://stackoverflow.com/questions/3572030/bash-script-absolute-path-with-osx/30795461#30795461
    POS="$(perl -e 'use Cwd "abs_path"; print abs_path(@ARGV[0])' -- "$POS")"
else
    POS="$(readlink -f -- "$POS")"
fi

if [ "$1" = "--meson" ]; then
    shift
fi

ASSIGN2="-Ddebug=false -Doptimization=3"
if [ "$1" = "--debug" ]; then
    ASSIGN2="-Ddebug=true -Doptimization=0"
    shift
    : "${VOXIE_BUILD_DIR:=$POS/../build-debug}"
else
    : "${VOXIE_BUILD_DIR:=$POS/../build}"
fi

if [ -f "$VOXIE_BUILD_DIR/Makefile" ]; then
    # Remove qmake build directory
    rm -rf "$VOXIE_BUILD_DIR"
fi
mkdir -p "$VOXIE_BUILD_DIR"
cd "$VOXIE_BUILD_DIR"

MESON=meson
if [ "$1" != "${1#--meson-binary=}" ]; then
    MESON="${1#--meson-binary=}"
    shift
elif [ "$1" = "--unpack-meson" ]; then
    MESONVERSION=0.56.2
    #MESONVERSION=0.57.0
    "../tools/download-dep.sh" --unpack "meson-$MESONVERSION.tar.gz"
    MESON="../tools/unpack-build-dep/meson-$MESONVERSION.tar.gz/meson-$MESONVERSION/meson.py"
    shift
fi

MESON_OPT=

MESON_OPT="$MESON_OPT -Dboost_include_path="
MESON_OPT="$MESON_OPT -Dlapacke_path="

FLAGS=
if [ "$1" = "--verbose" ]; then
    shift
    FLAGS=-v
fi

if [ "$1" != "${1#--hdf5-path=}" ]; then
    ASSIGN1="-Dhdf5_path=${1#--hdf5-path=}"
    shift
else
    ASSIGN1="-Dhdf5_path="
fi

if [ "$1" = "--disable-help-browser" ]; then
    MESON_OPT="$MESON_OPT -Dhelp_browser=disabled"
    shift
else
    MESON_OPT="$MESON_OPT -Dhelp_browser=enabled"
fi

if [ "$1" = "--disable-hdf5" ]; then
    MESON_OPT="$MESON_OPT -Dhdf5=disabled"
    shift
else
    MESON_OPT="$MESON_OPT -Dhdf5=enabled"
fi

if [ "$1" = "--disable-lapack" ]; then
    MESON_OPT="$MESON_OPT -Dlapack=disabled"
    shift
else
    MESON_OPT="$MESON_OPT -Dlapack=enabled"
fi

if [ "$1" = "--only-lib" ]; then
    MESON_OPT="$MESON_OPT -Dlibvoxie=disabled -Dmain=disabled -Dplugins=disabled -Dext=disabled -Dextra=disabled -Dtest=disabled"
    shift
else
    MESON_OPT="$MESON_OPT -Dlibvoxie=enabled -Dmain=enabled -Dplugins=enabled -Dext=enabled -Dextra=enabled -Dtest=enabled"
fi

MESON_RECONF=--reconfigure
if [ ! -f "build.ninja" ]; then
    MESON_RECONF=
fi
"$MESON" .. $MESON_RECONF $MESON_OPT "$ASSIGN1" $ASSIGN2
ninja $FLAGS "$@"
