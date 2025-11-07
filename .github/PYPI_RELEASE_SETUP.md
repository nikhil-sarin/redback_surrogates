# PyPI Automatic Release Setup

This repository now includes an automated workflow to publish releases to PyPI.

## How It Works

The workflow (`.github/workflows/pypi-publish.yml`) automatically publishes the package to PyPI when a new GitHub release is created.

## Setup Instructions

### 1. Create a PyPI API Token

1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Scroll down to "API tokens" section
3. Click "Add API token"
4. Set the token name (e.g., "redback_surrogates GitHub Actions")
5. Set the scope to "Project: redback_surrogates" (or "Entire account" if preferred)
6. Copy the generated token (starts with `pypi-`)

### 2. Add the Token to GitHub Secrets

1. Go to your GitHub repository settings
2. Navigate to "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Name: `PYPI_API_TOKEN`
5. Value: Paste the PyPI token you copied
6. Click "Add secret"

### 3. Creating a Release

To trigger an automatic PyPI upload:

1. Update the version in `setup.py`
2. Commit and push your changes
3. Create a new release on GitHub:
   - Go to "Releases" → "Create a new release"
   - Create a new tag (e.g., `v0.2.6`)
   - Write release notes
   - Click "Publish release"

The workflow will automatically:
- Build the package
- Check the package for errors
- Upload it to PyPI

## Alternative: Trusted Publisher (Recommended)

Instead of using API tokens, PyPI now supports "Trusted Publishers" which is more secure:

1. Go to your PyPI project page
2. Navigate to "Publishing"
3. Add GitHub as a trusted publisher:
   - Owner: `nikhil-sarin`
   - Repository: `redback_surrogates`
   - Workflow: `pypi-publish.yml`
   - Environment: (leave blank)

Then update the workflow to remove the `password` parameter (it will authenticate automatically).

## Troubleshooting

- Ensure the version in `setup.py` is incremented before each release
- Check the Actions tab on GitHub to see workflow execution logs
- Verify the `PYPI_API_TOKEN` secret is correctly set
