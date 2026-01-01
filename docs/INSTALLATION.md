# Installation Guide

This document describes the steps required to install and configure ST-BESA on a local machine.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Tested with 3.11 |
| Google Cloud Project | — | With Earth Engine API enabled |
| Google Account | — | Registered with Earth Engine |

## Step 1: Clone the Repository

```bash
git clone https://github.com/cssturkiye/st-besa.git
cd st-besa
```

## Step 2: Create a Virtual Environment

Using Conda (recommended):
```bash
conda create -n stbesa python=3.11
conda activate stbesa
```

Using venv:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Configure Google Earth Engine

### 4.1 Create a Google Cloud Project
1. Navigate to [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing one.
3. Note the **Project ID** (e.g., `my-stbesa-project`).

### 4.2 Enable the Earth Engine API
1. In the Cloud Console, go to **APIs & Services > Library**.
2. Search for "Earth Engine API".
3. Click **Enable**.

### 4.3 Authenticate

The application uses a robust 3-step authentication flow:

| Step | Method | Trigger |
|------|--------|---------|
| 1 | Existing credentials | Automatic if previously authenticated |
| 2 | Interactive link flow | Opens browser for OAuth consent |
| 3 | Notebook popup flow | For hosted environments (Colab) |

For local development, authenticate once via the command line:
```bash
earthengine authenticate
```

This stores credentials locally. Subsequent runs of `python app.py` will use these credentials automatically.

## Step 5: Run the Application

```bash
python app.py
```

The application will launch and provide a local URL (default: `http://127.0.0.1:7860`).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure the virtual environment is activated and dependencies are installed. |
| `Earth Engine authentication failed` | Run `earthengine authenticate` and ensure the API is enabled for your project. |
| `WinError 5: Access is denied` (export cleanup) | This is typically caused by cloud sync services (e.g., Google Drive) locking files. The exported ZIP file is still valid. |

## Next Steps

- [User Guide](USER_GUIDE.md): Learn how to use the application.
- [Architecture](ARCHITECTURE.md): Understand the technical design.
