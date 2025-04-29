# Release Process

This document outlines the process for releasing new versions of the langchain-aerospike package.

## Prerequisites

- You must have push access to the repository

## Release Steps

1. **Update Version**: Update the version in `pyproject.toml`.
   
   ```toml
   [tool.poetry]
   name = "langchain-aerospike"
   version = "X.Y.Z"  # Update this version
   ```

2. **Commit Changes**: Commit the version bump.
   
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to X.Y.Z"
   ```

3. **Create Release Tag**: Create a new Git tag with the version number.
   
   ```bash
   git tag vX.Y.Z
   ```

4. **Push Changes and Tag**: Push both the commit and the tag.
   
   ```bash
   git push origin main
   git push origin vX.Y.Z
   ```

5. **Monitor CI/CD Workflow**: Go to the GitHub Actions tab and monitor the `Build and Release` workflow. This workflow will:
   - Build the wheel and source distribution
   - Create a GitHub Release with the built packages attached

6. **Verify Release on GitHub**: Confirm that the new release is available on the GitHub Releases page with the wheel and source distribution attached.

## Version Numbering

We follow semantic versioning (SemVer):

- **MAJOR version**: Incompatible API changes
- **MINOR version**: Added functionality in a backward-compatible manner
- **PATCH version**: Backward-compatible bug fixes

## Handling Hotfixes

For critical issues requiring a quick fix:

1. Create a fix branch from the tagged release:
   ```bash
   git checkout vX.Y.Z
   git checkout -b hotfix/X.Y.Z+1
   ```

2. Make the necessary changes and update the version to X.Y.Z+1.

3. Commit, tag, and push as described above.

4. Cherry-pick or merge the changes back to the main branch if applicable. 