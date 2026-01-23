#!/bin/bash
# Monitor model download progress

JOBID=$1

if [ -z "$JOBID" ]; then
    echo "Usage: bash monitor_download.sh <job_id>"
    echo ""
    echo "Current jobs:"
    squeue -u $USER
    echo ""
    echo "Latest download job:"
    LATEST=$(ls -t logs/download_models_*.out 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        JOBID=$(basename "$LATEST" | sed 's/download_models_//; s/.out//')
        echo "  Job ID: $JOBID"
        echo "  Log: $LATEST"
    fi
    exit 0
fi

echo "=========================================="
echo "Monitoring Job $JOBID"
echo "=========================================="
echo ""

# Check job status
echo "Job status:"
squeue -j $JOBID 2>/dev/null || echo "  Job not in queue (completed or cancelled)"
echo ""

# Show log if exists
LOGFILE="logs/download_models_${JOBID}.out"
if [ -f "$LOGFILE" ]; then
    echo "Latest output from $LOGFILE:"
    echo "----------------------------------------"
    tail -50 "$LOGFILE"
    echo "----------------------------------------"
    echo ""
    echo "To follow live output:"
    echo "  tail -f $LOGFILE"
else
    echo "Log file not yet created (job still pending in queue)"
    echo ""
    echo "Check queue position:"
    echo "  squeue -j $JOBID"
fi

echo ""
echo "Cache status:"
du -sh /storage/project/r-pkastner3-0/ytang454/hf_cache/ 2>/dev/null || echo "  Cache directory empty"
