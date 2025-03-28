#!/bin/bash

# Maintenance script for VishwamAI CI/CD environment
# Recommended to run weekly via cron:
# 0 0 * * 0 /path/to/maintain_ci_cd.sh --project-id PROJECT_ID --repo OWNER/REPO

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
PROJECT_ID=""
GITHUB_REPO=""
LOG_DIR="/var/log/vishwamai"
MAX_LOG_AGE=30  # days
NOTIFY=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --repo)
            GITHUB_REPO="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --no-notify)
            NOTIFY=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$PROJECT_ID" ] || [ -z "$GITHUB_REPO" ]; then
    echo "Usage: $0 --project-id <project-id> --repo <owner/repo> [--log-dir <dir>] [--no-notify]"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/maintenance_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log() {
    local message="$(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo -e "$message"
    echo "$message" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOG_FILE"
}

# Function to send notifications
notify() {
    if [ "$NOTIFY" = true ]; then
        local message="$1"
        local level="$2"
        
        # Slack notification
        if [ ! -z "$SLACK_WEBHOOK" ]; then
            curl -s -X POST -H 'Content-type: application/json' \
                --data "{\"text\":\"$message\"}" \
                "$SLACK_WEBHOOK"
        fi
        
        # Email notification for errors
        if [ "$level" = "error" ] && [ ! -z "$EMAIL_USERNAME" ]; then
            echo "$message" | mail -s "VishwamAI CI/CD Maintenance Alert" \
                -r "$EMAIL_USERNAME" "$NOTIFICATION_EMAIL"
        fi
    fi
}

# Function to check resource quotas
check_quotas() {
    log "${YELLOW}Checking resource quotas...${NC}"
    
    # Check TPU quota
    tpu_quota=$(gcloud compute tpus list --limit=1 2>&1 || true)
    if [[ $tpu_quota == *"Quota exceeded"* ]]; then
        notify "⚠️ TPU quota exceeded in project $PROJECT_ID" "error"
        log "${RED}TPU quota exceeded${NC}"
    fi
    
    # Check API quotas
    api_quotas=$(gcloud services quota list --service=compute.googleapis.com \
        --project="$PROJECT_ID" \
        --format="table(metric.limit,usage)" 2>/dev/null)
    log "API Quotas:\n$api_quotas"
}

# Function to clean old resources
cleanup_old_resources() {
    log "${YELLOW}Cleaning up old resources...${NC}"
    
    # Clean old TPU instances
    old_tpus=$(gcloud compute tpus list \
        --filter="createTime<-P1D" \
        --format="get(name)" 2>/dev/null)
    if [ ! -z "$old_tpus" ]; then
        log "${RED}Found old TPU instances:${NC}"
        log "$old_tpus"
        notify "🗑️ Cleaning up old TPU instances in $PROJECT_ID" "info"
        gcloud compute tpus delete $old_tpus --quiet
    fi
    
    # Clean old logs
    find "$LOG_DIR" -name "*.log" -mtime +$MAX_LOG_AGE -delete
}

# Function to check GitHub Actions usage
check_actions_usage() {
    log "${YELLOW}Checking GitHub Actions usage...${NC}"
    
    # Get workflow usage
    workflow_usage=$(gh api \
        "/repos/$GITHUB_REPO/actions/workflows" \
        --jq '.workflows[] | {name, state, runs: .runs_count}' 2>/dev/null)
    log "Workflow Usage:\n$workflow_usage"
    
    # Check for failed runs
    failed_runs=$(gh run list -R "$GITHUB_REPO" \
        --json conclusion,databaseId,displayTitle \
        --jq '.[] | select(.conclusion=="failure")' 2>/dev/null)
    if [ ! -z "$failed_runs" ]; then
        log "${RED}Found failed workflow runs:${NC}"
        log "$failed_runs"
        notify "❌ Found failed workflow runs in $GITHUB_REPO" "error"
    fi
}

# Function to rotate old secrets
check_secret_rotation() {
    log "${YELLOW}Checking secret rotation...${NC}"
    
    # Check service account key age
    old_keys=$(gcloud iam service-accounts keys list \
        --iam-account="github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
        --format="table(name,validAfterTime)" \
        --filter="validAfterTime<-P90D" 2>/dev/null)
    
    if [ ! -z "$old_keys" ]; then
        log "${YELLOW}Found old service account keys:${NC}"
        log "$old_keys"
        notify "🔄 Rotating old service account keys in $PROJECT_ID" "info"
        
        # Rotate secrets
        "./cleanup_ci_cd.sh" \
            --project-id "$PROJECT_ID" \
            --repo "$GITHUB_REPO" \
            --rotate \
            --force
    fi
}

# Function to verify configuration
verify_configuration() {
    log "${YELLOW}Verifying CI/CD configuration...${NC}"
    
    # Check required secrets
    missing_secrets=""
    secrets=("GOOGLE_CREDENTIALS" "GCP_PROJECT_ID")
    for secret in "${secrets[@]}"; do
        if ! gh secret list -R "$GITHUB_REPO" | grep -q "$secret"; then
            missing_secrets="$missing_secrets\n- $secret"
        fi
    done
    
    if [ ! -z "$missing_secrets" ]; then
        log "${RED}Missing required secrets:${NC}$missing_secrets"
        notify "⚠️ Missing required GitHub secrets in $GITHUB_REPO" "error"
    fi
    
    # Check workflow files
    if [ ! -f ".github/workflows/test.yml" ]; then
        log "${RED}Missing workflow file: .github/workflows/test.yml${NC}"
        notify "⚠️ Missing workflow configuration in $GITHUB_REPO" "error"
    fi
}

# Main maintenance process
log "${GREEN}Starting CI/CD maintenance tasks${NC}"
log "Project ID: $PROJECT_ID"
log "GitHub Repository: $GITHUB_REPO"

# Run maintenance tasks
verify_configuration
check_quotas
cleanup_old_resources
check_actions_usage
check_secret_rotation

# Final status report
log "${GREEN}Maintenance tasks completed${NC}"
if [ -s "$LOG_FILE" ]; then
    notify "✅ CI/CD maintenance completed for $GITHUB_REPO\nSee logs: $LOG_FILE" "info"
fi

# Cleanup old logs
find "$LOG_DIR" -name "*.log" -mtime +$MAX_LOG_AGE -delete
