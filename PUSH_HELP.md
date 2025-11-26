# GitHub Push Instructions

## Repository Created: ✓
- Name: STU031
- Owner: bkp12345
- URL: https://github.com/bkp12345/STU031

## Files Ready to Push:
- flags.txt (all 3 flags)
- README.md (approach)
- reflection.md (methodology)
- solver.py (solution code)

## If Push Fails with "Repository not found":

This usually means GitHub authentication is needed. Try one of these:

### Option 1: Use GitHub CLI (Recommended)
```powershell
# Install GitHub CLI if not already installed
# Then authenticate:
gh auth login

# Push using GitHub CLI
cd c:\Users\krish\Downloads\STU031
gh repo create bkp12345/STU031 --public --source=. --remote=origin --push
```

### Option 2: Use Personal Access Token (HTTPS)
1. Go to: https://github.com/settings/tokens
2. Create new token with 'repo' scope
3. Copy the token
4. Run:
```powershell
cd c:\Users\krish\Downloads\STU031
git remote remove origin
git remote add origin https://YOUR_TOKEN@github.com/bkp12345/STU031.git
git push -u origin main
```

### Option 3: Use SSH (if configured)
```powershell
cd c:\Users\krish\Downloads\STU031
git remote remove origin
git remote add origin git@github.com:bkp12345/STU031.git
git push -u origin main
```

### Option 4: Use Git Credential Manager
```powershell
# On Windows, Git Credential Manager should prompt for credentials
# Just run:
cd c:\Users\krish\Downloads\STU031
git push -u origin main

# When prompted, enter your GitHub credentials
```

## Verify After Push:
Visit: https://github.com/bkp12345/STU031
You should see all 4 files with commit message containing flags.

## Submission Ready:
Once pushed, submit:
- Repo URL: https://github.com/bkp12345/STU031
- FLAG1: 70755B97
- FLAG2: FLAG2{979DA9FA}
- FLAG3: FLAG3{67111029}
