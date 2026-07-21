#!/usr/bin/env bash
# CI-style gate: every shipped .sh parses under Linux bash (catches CRLF and syntax slips).
fail=0
for f in /kit/scripts/*.sh /kit/demo/*.sh /kit/selftest/*.sh; do
  if bash -n "$f" 2>/tmp/err; then
    echo "OK   $f"
  else
    echo "FAIL $f"
    cat /tmp/err
    fail=1
  fi
done
if [ "$fail" -eq 0 ]; then echo "GATE PASSED: all bash scripts parse"; fi
exit "$fail"
