#!/usr/bin/env python3
"""
Interactive script to set up Kaggle API credentials.
This will help you configure kaggle.json properly.
"""

import os
import json
from pathlib import Path

print("="*80)
print("🔑 KAGGLE API CREDENTIALS SETUP")
print("="*80)
print()

# Check if kaggle.json already exists
kaggle_dir = Path.home() / ".kaggle"
kaggle_json_path = kaggle_dir / "kaggle.json"

if kaggle_json_path.exists():
    print(f"✅ Found existing kaggle.json at: {kaggle_json_path}")
    print()
    
    try:
        with open(kaggle_json_path, 'r') as f:
            creds = json.load(f)
        
        print("Current credentials:")
        print(f"  Username: {creds.get('username', 'NOT SET')}")
        print(f"  Key: {creds.get('key', 'NOT SET')[:10]}...{creds.get('key', '')[-4:]}")
        print()
        
        response = input("Do you want to update these credentials? (y/n): ").strip().lower()
        if response != 'y':
            print("\nKeeping existing credentials. You're all set!")
            exit(0)
    except Exception as e:
        print(f"⚠️  Error reading existing file: {e}")
        print("Will create a new one...")

print()
print("To get your Kaggle credentials:")
print("1. Go to: https://www.kaggle.com/settings")
print("2. Scroll to 'API' section")
print("3. Click 'Create New Token'")
print("4. Open the downloaded kaggle.json file")
print()
print("="*80)
print()

# Option 1: Load from downloaded kaggle.json
print("Option 1: Load from downloaded kaggle.json")
print("-" * 80)
downloads_path = Path.home() / "Downloads" / "kaggle.json"

if downloads_path.exists():
    print(f"✅ Found kaggle.json in Downloads!")
    response = input("Load credentials from Downloads/kaggle.json? (y/n): ").strip().lower()
    
    if response == 'y':
        try:
            with open(downloads_path, 'r') as f:
                creds = json.load(f)
            
            username = creds.get('username')
            key = creds.get('key')
            
            if username and key:
                print(f"\n✅ Loaded credentials:")
                print(f"   Username: {username}")
                print(f"   Key: {key[:10]}...{key[-4:]}")
                
                # Save to ~/.kaggle/
                kaggle_dir.mkdir(exist_ok=True)
                with open(kaggle_json_path, 'w') as f:
                    json.dump(creds, f, indent=2)
                
                # Set permissions
                try:
                    os.chmod(kaggle_json_path, 0o600)
                except:
                    pass
                
                print(f"\n✅ Saved to: {kaggle_json_path}")
                print("\n🎉 Setup complete! You can now run the training notebook.")
                exit(0)
        except Exception as e:
            print(f"❌ Error loading file: {e}")
else:
    print("❌ kaggle.json not found in Downloads")

print()

# Option 2: Manual entry
print("Option 2: Enter credentials manually")
print("-" * 80)
print()

username = input("Enter your Kaggle username: ").strip()
key = input("Enter your Kaggle API key: ").strip()

if not username or not key:
    print("\n❌ Username and key cannot be empty!")
    exit(1)

# Validate format
if len(key) != 32:
    print(f"\n⚠️  Warning: API key should be 32 characters, you entered {len(key)}")
    response = input("Continue anyway? (y/n): ").strip().lower()
    if response != 'y':
        exit(1)

# Save credentials
creds = {
    "username": username,
    "key": key
}

kaggle_dir.mkdir(exist_ok=True)
with open(kaggle_json_path, 'w') as f:
    json.dump(creds, f, indent=2)

# Set permissions
try:
    os.chmod(kaggle_json_path, 0o600)
    print(f"\n✅ Set file permissions to 600")
except:
    print(f"\n⚠️  Could not set file permissions (Windows?)")

print(f"\n✅ Credentials saved to: {kaggle_json_path}")
print(f"   Username: {username}")
print(f"   Key: {key[:10]}...{key[-4:]}")

# Test the credentials
print("\n" + "="*80)
print("🧪 TESTING CREDENTIALS")
print("="*80)

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("✅ Authentication successful!")
    print("\n🎉 Setup complete! You can now run the training notebook.")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
    print("\nPlease check:")
    print("1. Your username is correct")
    print("2. Your API key is correct")
    print("3. You have an active Kaggle account")

