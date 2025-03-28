#!/bin/bash

# Setup script for VishwamAI CI/CD environment

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
PROJECT_ID=""
REGION="us-central1"
ZONE="us-central1-a"
SERVICE_ACCOUNT_NAME="github-actions"
GITHUB_REPO=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --zone)
            ZONE="$2"
            shift 2
            ;;
        --repo)
            GITHUB_REPO="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$PROJECT_ID" ] || [ -z "$GITHUB_REPO" ]; then
    echo "Usage: $0 --project-id <project-id> --repo <owner/repo> [--region <region>] [--zone <zone>]"
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

echo -e "${GREEN}Setting up CI/CD environment for VishwamAI${NC}"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Zone: $ZONE"
echo "GitHub Repository: $GITHUB_REPO"

# Setup Google Cloud
echo -e "\n${YELLOW}Setting up Google Cloud...${NC}"

# Configure project
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "Enabling required APIs..."
apis=(
    "compute.googleapis.com"
    "tpu.googleapis.com"
    "cloudresourcemanager.googleapis.com"
    "iam.googleapis.com"
)

for api in "${apis[@]}"; do
    gcloud services enable "$api"
done

# Create service account
echo "Creating service account..."
gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
    --description="Service account for GitHub Actions" \
    --display-name="GitHub Actions"

# Grant permissions
echo "Granting permissions..."
roles=(
    "roles/compute.admin"
    "roles/tpu.admin"
    "roles/iam.serviceAccountUser"
    "roles/storage.objectViewer"
)

for role in "${roles[@]}"; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
        --role="$role"
done

# Create and download service account key
echo "Creating service account key..."
key_file="key.json"
gcloud iam service-accounts keys create "$key_file" \
    --iam-account="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"

# Setup GitHub Secrets
echo -e "\n${YELLOW}Setting up GitHub Secrets...${NC}"

# Check if logged in to GitHub
if ! gh auth status &> /dev/null; then
    echo "Please login to GitHub:"
    gh auth login
fi

# Add secrets to GitHub repository
echo "Adding secrets to GitHub repository..."
gh secret set GOOGLE_CREDENTIALS < "$key_file" -R "$GITHUB_REPO"
gh secret set GCP_PROJECT_ID -b "$PROJECT_ID" -R "$GITHUB_REPO"

# Optional: Setup notification secrets
read -p "Do you want to setup Slack notifications? (y/N) " setup_slack
if [[ $setup_slack =~ ^[Yy]$ ]]; then
    read -p "Enter Slack webhook URL: " slack_webhook
    gh secret set SLACK_WEBHOOK -b "$slack_webhook" -R "$GITHUB_REPO"
fi

read -p "Do you want to setup email notifications? (y/N) " setup_email
if [[ $setup_email =~ ^[Yy]$ ]]; then
    read -p "Enter email username: " email_username
    read -s -p "Enter email app password: " email_password
    echo
    read -p "Enter notification email address: " notification_email
    
    gh secret set EMAIL_USERNAME -b "$email_username" -R "$GITHUB_REPO"
    gh secret set EMAIL_PASSWORD -b "$email_password" -R "$GITHUB_REPO"
    gh secret set NOTIFICATION_EMAIL -b "$notification_email" -R "$GITHUB_REPO"
fi

# Cleanup
echo "Cleaning up sensitive files..."
rm -f "$key_file"

# Create debug logging secrets
echo "Setting up debug logging..."
gh secret set ACTIONS_RUNNER_DEBUG -b "true" -R "$GITHUB_REPO"
gh secret set ACTIONS_STEP_DEBUG -b "true" -R "$GITHUB_REPO"

echo -e "\n${GREEN}CI/CD environment setup complete!${NC}"
echo -e "\nNext steps:"
echo "1. Update .github/workflows/test.yml if needed"
echo "2. Push changes to trigger the workflow"
echo "3. Monitor the Actions tab for test results"
echo -e "\nFor more information, see docs/ci_cd_setup.md"
