#!/bin/bash
# Quick check on the 1-question test

echo "=========================================="
echo "Test Job Status"
echo "=========================================="

# Get latest test job
JOB_ID=$(squeue -u ytang454 -n beat300_test_1q -h -o "%i" | head -1)

if [ -z "$JOB_ID" ]; then
    echo "[INFO] No running test job found"
    echo ""
    echo "Checking recent logs..."
    LATEST_LOG=$(ls -t logs/test_one_question_*.out 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo ""
        echo "=========================================="
        echo "Latest test output:"
        echo "=========================================="
        cat "$LATEST_LOG"
    else
        echo "[INFO] No test logs found yet"
    fi
else
    echo "[RUNNING] Job ID: $JOB_ID"
    squeue -j $JOB_ID
    echo ""
    echo "Monitor with:"
    echo "  watch -n 5 squeue -u ytang454"
    echo ""
    echo "View output when complete:"
    echo "  cat logs/test_one_question_${JOB_ID}.out"
fi

echo ""
