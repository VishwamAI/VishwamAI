# CI/CD Setup Guide for VishwamAI

This guide explains how to set up Continuous Integration and Continuous Deployment (CI/CD) for VishwamAI testing infrastructure.

## GitHub Actions Setup

### Required Secrets

Set up the following secrets in your GitHub repository (Settings > Secrets and variables > Actions):

#### Essential Secrets

1. Google Cloud Platform (for TPU tests):
```yaml
GOOGLE_CREDENTIALS: |
  {
    # Your Google Cloud service account JSON key
  }
GCP_PROJECT_ID: "your-project-id"
```

2. Notifications (optional):
```yaml
# Slack notifications
SLACK_WEBHOOK: "https://hooks.slack.com/services/..."

# Email notifications
EMAIL_USERNAME: "your-email@gmail.com"
EMAIL_PASSWORD: "your-app-specific-password"
NOTIFICATION_EMAIL: "notifications@your-domain.com"
```

### Setting Up Secrets

1. Google Cloud Setup:
```bash
# Create service account
gcloud iam service-accounts create github-actions \
    --description="Service account for GitHub Actions" \
    --display-name="GitHub Actions"

# Grant required permissions
gcloud projects add-iam-policy-binding your-project-id \
    --member="serviceAccount:github-actions@your-project-id.iam.gserviceaccount.com" \
    --role="roles/compute.admin"

gcloud projects add-iam-policy-binding your-project-id \
    --member="serviceAccount:github-actions@your-project-id.iam.gserviceaccount.com" \
    --role="roles/tpu.admin"

# Create and download key
gcloud iam service-accounts keys create key.json \
    --iam-account=github-actions@your-project-id.iam.gserviceaccount.com
```

2. Gmail Setup (for notifications):
```
1. Go to Google Account settings
2. Security > App passwords
3. Generate app password for "GitHub Actions"
4. Use this password in EMAIL_PASSWORD secret
```

3. Slack Setup:
```
1. Go to Slack Apps
2. Create new app or use existing
3. Add Incoming Webhooks
4. Create webhook for your channel
5. Copy webhook URL to SLACK_WEBHOOK secret
```

## Workflow Configuration

### Test Matrix

The workflow runs tests across multiple configurations:

1. CPU Tests:
- Python versions: 3.8, 3.9, 3.10, 3.11
- All test types
- Runs on every push/PR

2. GPU Tests:
- Python 3.10
- CUDA 11.8
- All test types
- Runs after CPU tests pass

3. TPU Tests:
- Python 3.10
- TPU v4-8
- All test types
- Runs on schedule/manual trigger only

### Triggers

The workflow runs on:
```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight UTC
  workflow_dispatch:      # Manual trigger
```

### Resource Management

1. TPU Resources:
```yaml
# Created during test run
- v4-8 TPU VM
- Zone: us-central1-a
- Auto-cleanup after tests

# Estimated costs
- ~$4.50/hour
```

2. GPU Resources:
```yaml
# GitHub-hosted runner
- NVIDIA Tesla K80
- CUDA 11.8
- No additional cost
```

3. CPU Resources:
```yaml
# GitHub-hosted runner
- 2-core CPU
- No additional cost
```

## Monitoring and Notifications

### Test Reports

1. Artifacts:
```yaml
# Generated per run
- Test results JSON
- Coverage reports
- Performance benchmarks
- Visualizations
```

2. Locations:
```yaml
# GitHub Actions
- Artifacts tab in workflow run
- Download available for 90 days
```

### Notifications

1. Success:
```yaml
# Slack
- Test completion status
- Performance metrics
- Links to reports
```

2. Failure:
```yaml
# Email
- Failure details
- Error logs
- Links to workflow run

# Slack
- Immediate failure alert
- Error summary
```

## Customization

### Adding New Tests

1. Update test matrix:
```yaml
strategy:
  matrix:
    python-version: [...]    # Add versions
    platform: [...]         # Add platforms
```

2. Add test job:
```yaml
test-new-platform:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    # Add setup steps
    # Add test steps
```

### Modifying Notifications

1. Add notification channel:
```yaml
notify:
  steps:
    - name: New notification
      uses: your-action
      with:
        # Configure notification
```

2. Update notification content:
```yaml
# Edit notification templates in
- .github/workflows/templates/
```

## Troubleshooting

### Common Issues

1. TPU Access:
```
Error: Could not create TPU VM
Solution: Check GCP credentials and permissions
```

2. GPU Tests:
```
Error: CUDA not available
Solution: Verify CUDA installation in workflow
```

3. Notifications:
```
Error: Could not send notification
Solution: Verify secret availability and values
```

### Debug Tips

1. Enable debug logging:
```bash
# Add secret
ACTIONS_RUNNER_DEBUG: true
ACTIONS_STEP_DEBUG: true
```

2. Local workflow testing:
```bash
# Install act
brew install act

# Run workflow locally
act -j test-cpu
```

## Maintenance

### Regular Tasks

1. Secret rotation:
```bash
# Every 90 days
- Update GCP service account key
- Update notification tokens
```

2. Dependency updates:
```bash
# Monthly
- Update action versions
- Update test dependencies
```

3. Cost monitoring:
```bash
# Weekly
- Review TPU usage
- Optimize test duration
- Adjust scheduled runs
```

## Support

For issues or questions:
1. Open GitHub issue
2. Tag with 'ci-cd' label
3. Include workflow run URL
4. Attach relevant logs
