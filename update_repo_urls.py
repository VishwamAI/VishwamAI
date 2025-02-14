#!/usr/bin/env python3
"""
Script to update repository URLs across documentation files.
"""

import os
import re

def update_repo_urls(old_url: str, new_url: str):
    """Update repository URLs in all documentation files."""
    
    # Files to update
    doc_files = [
        'README.md',
        'SETUP.md',
        'TRAINING.md',
        'MODEL_CARD.md'
    ]
    
    # Pattern to match URLs (handles both markdown and raw links)
    url_pattern = re.compile(f'({re.escape(old_url)})')
    
    for filename in doc_files:
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found")
            continue
            
        # Read file
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace URLs
        if old_url in content:
            new_content = url_pattern.sub(new_url, content)
            
            # Write updated content
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            print(f"Updated {filename}")
        else:
            print(f"No changes needed in {filename}")

def main():
    """Main function to update repository URLs."""
    
    # Current repository URL
    current_url = "https://github.com/VishwamAI/VishwamAI"
    
    print("Update Repository URLs")
    print("=====================")
    print(f"Current URL: {current_url}")
    
    # Get new URL
    new_url = input("Enter new repository URL (or press Enter to cancel): ").strip()
    
    if not new_url:
        print("Operation cancelled")
        return
        
    # Confirm update
    confirm = input(f"Update repository URL from {current_url} to {new_url}? [y/N]: ").lower()
    
    if confirm != 'y':
        print("Operation cancelled")
        return
        
    # Update URLs
    print("\nUpdating repository URLs...")
    update_repo_urls(current_url, new_url)
    print("Done!")

if __name__ == "__main__":
    main()
