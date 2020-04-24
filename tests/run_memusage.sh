#!/bin/sh
# Test script for memusage program (requires grep to look at generated file)

set -e
echo
echo "Testing memusage stats..."
# GPTL writes memusage stats to stderr
./memusage 2> out.memusage
grep -q grew out.memusage
if test "$?" = 0; then
  echo "SUCCESS!"
  exit 0
else
  echo "FAILURE!"
  exit 1
fi
