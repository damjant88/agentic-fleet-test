# 📦 DevContainer Installation Report

**Generated on:** Fri Sep 18 12:45:31 UTC 2026

## 📊 Installation Summary

### 🖥️ Tmux Installation
### 🐙 GitHub CLI Installation
### 🤖 Claude Code Installation
### 🐍 UV Installation
### 📊 Claude Monitor Installation
### 🌊 Claude Flow Installation
### 🐝 RUV Swarm Installation
### 📈 CCUsage Installation
| Tool | Status | Notes |
|------|--------|-------|
| tmux | ❌ Failed | Installation failed - see manual instructions below |
| GitHub CLI | ✅ Success | Installed via apt-get with sudo |
| Claude Code | ✅ Success | Installed via npm |
| UV | ✅ Success | Installed via official installer |
| Claude Monitor | ✅ Success | Installed via UV tool |
| Claude Flow | ✅ Success | Installed via npm (alpha), ruflo@3.42.4. Requires Node 22 LTS — Node 24.19+ crashes on `init` (better-sqlite3/RemoveEnvironmentCleanupHook assertion, see PR #1); pinned in devcontainer.json's node feature. |
| RUV Swarm | ✅ Success | Installed via npm |
| CCUsage | ✅ Success | Installed via npm |

## ⚠️ Manual Installation Instructions

Some tools failed to install automatically. Please follow these instructions to install them manually:

### 🖥️ Installing tmux manually

**For Debian/Ubuntu:**
```bash
sudo apt update
sudo apt install -y tmux
```

**For Red Hat/CentOS/Fedora:**
```bash
sudo yum install -y tmux
```

**For macOS:**
```bash
brew install tmux
```


---

*Report generated at: Fri Sep 18 12:53:38 UTC 2026*
*Claude Flow note updated Sep 18 2026 after diagnosing a Node 24 / better-sqlite3 crash on `claude-flow init` — see [PR #1](https://github.com/damjant88/agentic-fleet-test/pull/1).*
