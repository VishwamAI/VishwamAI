#!/bin/bash

# Cleanup script for VishwamAI CI/CD environment

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
PROJECT_ID=""
GITHUB_REPO=""
SERVICE_ACCOUNT_NAME="github-actions"
ROTATE_SECRETS=false
DELETE_RESOURCES=false
FORCE=false

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
        --rotate)
            ROTATE_SECRETS=true
            shift
            ;;
        --delete)
            DELETE_RESOURCES=true
            shift
            ;;
        --force)
            FORCE=true
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
    echo "Usage: $0 --project-id <project-id> --repo <owner/repo> [--rotate] [--delete] [--force]"
    echo
    echo "Options:"
    echo "  --rotate    Rotate secrets (create new and delete old)"
    echo "  --delete    Delete all CI/CD resources"
    echo "  --force     Skip confirmation prompts"
    exit 1
fi

# Check gcloud installation
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud is not installed${NC}"
    exit 1
fi

# Check GitHub CLI installation
if ! command -v gh &> /dev/null; then
    echo -e "${RED}Error: GitHub CLI is not installed${NC}"
    exit 1
fi

# Function to confirm action
confirm() {
    if [ "$FORCE" = true ]; then
        return 0
    fi
    
    read -p "$1 (y/N) " response
    [[ $response =~ ^[Yy]$ ]]
}

# Function to rotate secrets
rotate_secrets() {
    echo -e "\n${YELLOW}Rotating CI/CD secrets...${NC}"
    
    # Create new service account key
    echo "Creating new service account key..."
    new_key_file="new_key.json"
    gcloud iam service-accounts keys create "$new_key_file" \
        --iam-account="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"
    
    # Update GitHub secret
    echo "Updating GitHub secrets..."
    gh secret set GOOGLE_CREDENTIALS < "$new_key_file" -R "$GITHUB_REPO"
    
    # List and delete old keys
    echo "Cleaning up old service account keys..."
    old_keys=$(gcloud iam service-accounts keys list \
        --iam-account="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
        --format="get(name)" | grep -v "$(cat $new_key_file | jq -r '.private_key_id')")
    
    for key in $old_keys; do
        gcloud iam service-accounts keys delete "$key" \
            --iam-account="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
            --quiet
    done
    
    # Cleanup
    rm -f "$new_key_file"
    
    echo -e "${GREEN}Secrets rotated successfully${NC}"
}

# Function to delete resources
delete_resources() {
    echo -e "\n${YELLOW}Deleting CI/CD resources...${NC}"
    
    # Delete service account
    echo "Deleting service account..."
    gcloud iam service-accounts delete \
        "$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
        --quiet
    
    # Delete GitHub secrets
    echo "Deleting GitHub secrets..."
    secrets=(
        "GOOGLE_CREDENTIALS"
        "GCP_PROJECT_ID"
        "ACTIONS_RUNNER_DEBUG"
        "ACTIONS_STEP_DEBUG"
    )
    
    for secret in "${secrets[@]}"; do
        gh secret remove "$secret" -R "$GITHUB_REPO" 2>/dev/null || true
    done
    
    # Optional: Delete notification secrets
    if confirm "Delete notification secrets?"; then
        notification_secrets=(
            "SLACK_WEBHOOK"
            "EMAIL_USERNAME"
            "EMAIL_PASSWORD"
            "NOTIFICATION_EMAIL"
        )
        
        for secret in "${notification_secrets[@]}"; do
            gh secret remove "$secret" -R "$GITHUB_REPO" 2>/dev/null || true
        done
    fi
    
    echo -e "${GREEN}Resources deleted successfully${NC}"
}

# Main cleanup process
echo -e "${GREEN}Starting CI/CD environment cleanup${NC}"
echo "Project ID: $PROJECT_ID"
echo "GitHub Repository: $GITHUB_REPO"

# Configure project
gcloud config set project "$PROJECT_ID"

# Rotate secrets if requested
if [ "$ROTATE_SECRETS" = true ]; then
    if confirm "Rotate CI/CD secrets?"; then
        rotate_secrets
    fi
fi

# Delete resources if requested
if [ "$DELETE_RESOURCES" = true ]; then
    if confirm "Delete all CI/CD resources? This action cannot be undone!"; then
        delete_resources
    fi
fi

# Check for orphaned resources
echo -e "\n${YELLOW}Checking for orphaned resources...${NC}"

# Check TPU resources
orphaned_tpus=$(gcloud compute tpus list --format="table(name,zone)" 2>/dev/null)
if [ ! -z "$orphaned_tpus" ]; then
    echo -e "${RED}Found orphaned TPU resources:${NC}"
    echo "$orphaned_tpus"
    if confirm "Delete orphaned TPU resources?"; then
        gcloud compute tpus delete $(gcloud compute tpus list --format="get(name)") --quiet
    fi
fi

# Check service account keys
old_keys=$(gcloud iam service-accounts keys list \
    --iam-account="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
    --format="table(name,validAfterTime)")
if [ ! -z "$old_keys" ]; then
    echo -e "${YELLOW}Service account keys:${NC}"
    echo "$old_keys"
    if confirm "Rotate service account keys?"; then
        rotate_secrets
    fi
fi

echo -e "\n${GREEN}Cleanup complete!${NC}"
if [ "$ROTATE_SECRETS" = false ] && [ "$DELETE_RESOURCES" = false ]; then
    echo -e "\nNo changes were made. Use --rotate to rotate secrets or --delete to remove resources."
fi
