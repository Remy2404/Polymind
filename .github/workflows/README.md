# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the Polymind project.

## Workflows

### 🔄 CI (`ci.yml`)
**Triggers**: Push/PR to `main` or `develop` branches

Runs comprehensive CI checks including:
- **Testing**: Runs pytest on multiple Python versions (3.11, 3.12)
- **Linting**: Code quality checks with Ruff
- **Security**: Bandit security scanning
- **Docker**: Builds and tests Docker image
- **Coverage**: Code coverage reporting with Codecov

### ✨ Code Quality (`quality.yml`)
**Triggers**: Pull requests to `main` or `develop`

Additional quality checks:
- Security vulnerability scanning
- Type checking with mypy
- Large file detection
- Secrets detection
- Dependency updates check
- Coverage badge generation

### 📦 Release (`release.yml`)
**Triggers**: Git tags matching `v*.*.*`

Creates GitHub releases:
- Runs full test suite
- Builds Docker image
- Generates changelog
- Creates GitHub release

### 🌙 Nightly (`nightly.yml`)
**Triggers**: Scheduled nightly at 2 AM UTC, or manual dispatch

Maintenance tasks:
- Full test suite run
- Documentation link checking
- Branch cleanup suggestions
- Disk usage monitoring
- Nightly report generation

## Configuration

### Dependabot
Automated dependency updates are configured in `dependabot.yml`:
- Weekly Python package updates
- Weekly GitHub Actions updates
- Automatic PR creation with proper labels

## Usage

### Manual Triggers
Some workflows can be triggered manually:
- Nightly: Go to Actions → Nightly → "Run workflow"

### Branch Protection
Consider setting up branch protection rules for `main`:
- Require CI checks to pass
- Require up-to-date branches
- Require code reviews

## Troubleshooting

### Common Issues

1. **Tests failing**: Check test logs and ensure all dependencies are properly installed
2. **Linting errors**: Run `uv run ruff check` locally to fix issues
3. **Docker build failing**: Check Dockerfile and ensure all required files are included

### Local Testing

Before pushing, you can run similar checks locally:

```bash
# Install dependencies
uv sync --dev

# Run tests
uv run pytest tests/

# Run linting
uv run ruff check src/ tests/

# Build Docker image
docker build -t polymind:test .
```

## Contributing

When adding new workflows:
1. Follow naming conventions (lowercase, descriptive)
2. Add proper documentation
3. Test workflows on a feature branch first
4. Update this README with new workflow details