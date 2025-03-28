#!/bin/bash

# Script to manage cloud resources for VishwamAI testing

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
ZONE="us-central1-a"
TPU_TYPE="v4-8"
GPU_TYPE="nvidia-tesla-v100"
CPU_TYPE="n2-standard-16"

# Function to check gcloud installation
check_gcloud() {
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}gcloud not found. Please install Google Cloud SDK${NC}"
        exit 1
    fi
}

# Function to create resources
create_resources() {
    local platform=$1
    case $platform in
        tpu)
            echo -e "${YELLOW}Creating TPU VM...${NC}"
            gcloud compute tpus tpu-vm create vishwamai-tpu \
                --zone=$ZONE \
                --accelerator-type=$TPU_TYPE \
                --version=tpu-vm-base
            ;;
        gpu)
            echo -e "${YELLOW}Creating GPU instance...${NC}"
            gcloud compute instances create vishwamai-gpu \
                --zone=$ZONE \
                --machine-type=n1-standard-8 \
                --accelerator="type=$GPU_TYPE,count=1" \
                --maintenance-policy=TERMINATE \
                --image-family=debian-11-gpu \
                --image-project=debian-cloud
            ;;
        cpu)
            echo -e "${YELLOW}Creating CPU instance...${NC}"
            gcloud compute instances create vishwamai-cpu \
                --zone=$ZONE \
                --machine-type=$CPU_TYPE \
                --image-family=debian-11 \
                --image-project=debian-cloud
            ;;
        *)
            echo -e "${RED}Invalid platform: $platform${NC}"
            exit 1
            ;;
    esac
}

# Function to delete resources
delete_resources() {
    local platform=$1
    case $platform in
        tpu)
            echo -e "${YELLOW}Deleting TPU VM...${NC}"
            gcloud compute tpus tpu-vm delete vishwamai-tpu --zone=$ZONE --quiet
            ;;
        gpu)
            echo -e "${YELLOW}Deleting GPU instance...${NC}"
            gcloud compute instances delete vishwamai-gpu --zone=$ZONE --quiet
            ;;
        cpu)
            echo -e "${YELLOW}Deleting CPU instance...${NC}"
            gcloud compute instances delete vishwamai-cpu --zone=$ZONE --quiet
            ;;
        all)
            echo -e "${YELLOW}Deleting all resources...${NC}"
            gcloud compute tpus tpu-vm delete vishwamai-tpu --zone=$ZONE --quiet 2>/dev/null || true
            gcloud compute instances delete vishwamai-gpu --zone=$ZONE --quiet 2>/dev/null || true
            gcloud compute instances delete vishwamai-cpu --zone=$ZONE --quiet 2>/dev/null || true
            ;;
        *)
            echo -e "${RED}Invalid platform: $platform${NC}"
            exit 1
            ;;
    esac
}

# Function to check resource status
check_status() {
    local platform=$1
    case $platform in
        tpu)
            echo -e "${YELLOW}TPU VM Status:${NC}"
            gcloud compute tpus tpu-vm describe vishwamai-tpu --zone=$ZONE
            ;;
        gpu)
            echo -e "${YELLOW}GPU Instance Status:${NC}"
            gcloud compute instances describe vishwamai-gpu --zone=$ZONE
            ;;
        cpu)
            echo -e "${YELLOW}CPU Instance Status:${NC}"
            gcloud compute instances describe vishwamai-cpu --zone=$ZONE
            ;;
        all)
            echo -e "${YELLOW}All Resource Status:${NC}"
            echo -e "\nTPU VM Status:"
            gcloud compute tpus tpu-vm describe vishwamai-tpu --zone=$ZONE 2>/dev/null || echo "No TPU VM found"
            echo -e "\nGPU Instance Status:"
            gcloud compute instances describe vishwamai-gpu --zone=$ZONE 2>/dev/null || echo "No GPU instance found"
            echo -e "\nCPU Instance Status:"
            gcloud compute instances describe vishwamai-cpu --zone=$ZONE 2>/dev/null || echo "No CPU instance found"
            ;;
        *)
            echo -e "${RED}Invalid platform: $platform${NC}"
            exit 1
            ;;
    esac
}

# Function to monitor resource usage
monitor_resources() {
    local platform=$1
    case $platform in
        tpu)
            echo -e "${YELLOW}Monitoring TPU VM...${NC}"
            watch -n 5 "gcloud compute tpus tpu-vm describe vishwamai-tpu --zone=$ZONE | grep 'state\|utilization'"
            ;;
        gpu)
            echo -e "${YELLOW}Monitoring GPU Instance...${NC}"
            gcloud compute ssh vishwamai-gpu --zone=$ZONE --command="watch nvidia-smi"
            ;;
        cpu)
            echo -e "${YELLOW}Monitoring CPU Instance...${NC}"
            gcloud compute ssh vishwamai-cpu --zone=$ZONE --command="top"
            ;;
        *)
            echo -e "${RED}Invalid platform: $platform${NC}"
            exit 1
            ;;
    esac
}

# Function to estimate costs
estimate_costs() {
    local platform=$1
    local hours=${2:-1}
    
    case $platform in
        tpu)
            local tpu_cost=$(echo "scale=2; 4.50 * $hours" | bc)
            echo -e "${YELLOW}Estimated TPU cost for $hours hours: \$$tpu_cost${NC}"
            ;;
        gpu)
            local gpu_cost=$(echo "scale=2; 2.48 * $hours" | bc)
            echo -e "${YELLOW}Estimated GPU cost for $hours hours: \$$gpu_cost${NC}"
            ;;
        cpu)
            local cpu_cost=$(echo "scale=2; 0.76 * $hours" | bc)
            echo -e "${YELLOW}Estimated CPU cost for $hours hours: \$$cpu_cost${NC}"
            ;;
        all)
            local total_cost=$(echo "scale=2; (4.50 + 2.48 + 0.76) * $hours" | bc)
            echo -e "${YELLOW}Estimated total cost for $hours hours: \$$total_cost${NC}"
            ;;
        *)
            echo -e "${RED}Invalid platform: $platform${NC}"
            exit 1
            ;;
    esac
}

# Main function
main() {
    check_gcloud

    case $1 in
        create)
            create_resources $2
            ;;
        delete)
            delete_resources $2
            ;;
        status)
            check_status $2
            ;;
        monitor)
            monitor_resources $2
            ;;
        estimate)
            estimate_costs $2 $3
            ;;
        *)
            echo "Usage: $0 {create|delete|status|monitor|estimate} {tpu|gpu|cpu|all} [hours]"
            echo
            echo "Commands:"
            echo "  create   - Create cloud resources"
            echo "  delete   - Delete cloud resources"
            echo "  status   - Check resource status"
            echo "  monitor  - Monitor resource usage"
            echo "  estimate - Estimate costs (optional: specify hours)"
            echo
            echo "Platforms:"
            echo "  tpu  - TPU VM (v4-8)"
            echo "  gpu  - GPU Instance (V100)"
            echo "  cpu  - CPU Instance (n2-standard-16)"
            echo "  all  - All platforms"
            echo
            echo "Examples:"
            echo "  $0 create tpu      # Create TPU VM"
            echo "  $0 monitor gpu     # Monitor GPU usage"
            echo "  $0 estimate all 24 # Estimate 24-hour cost for all resources"
            exit 1
            ;;
    esac
}

main "$@"
