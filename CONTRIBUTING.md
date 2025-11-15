# Contributing to ReleAF AI

Thank you for your interest in contributing to ReleAF AI! This document provides guidelines for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/your-repo/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, GPU, etc.)
   - Relevant logs or screenshots

### Suggesting Features

1. Check existing feature requests
2. Create a new issue with:
   - Clear use case
   - Proposed solution
   - Potential impact
   - Implementation considerations

### Contributing Code

#### 1. Fork and Clone

```bash
git clone https://github.com/your-username/Sustainability-AI-Model.git
cd Sustainability-AI-Model
```

#### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/updates

#### 3. Set Up Development Environment

```bash
bash scripts/setup.sh
source venv/bin/activate
pip install -e ".[dev]"
```

#### 4. Make Changes

Follow our coding standards:

**Python Style**:
- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Add type hints where appropriate
- Write docstrings for functions and classes

**Example**:
```python
def classify_waste(
    image: Image.Image,
    model: torch.nn.Module,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Classify waste item from image.
    
    Args:
        image: PIL Image object
        model: Classification model
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with predictions and confidence scores
    """
    # Implementation
    pass
```

#### 5. Write Tests

Add tests for new functionality:

```bash
# Create test file
touch tests/unit/test_your_feature.py
```

Example test:
```python
import pytest
from services.vision_service import VisionService

def test_classify_waste():
    service = VisionService()
    # Test implementation
    assert result is not None
```

Run tests:
```bash
pytest tests/
```

#### 6. Format Code

```bash
# Format with Black
black .

# Sort imports
isort .

# Check with flake8
flake8 .

# Type checking
mypy services/ training/
```

#### 7. Commit Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add waste classification endpoint

- Implement POST /api/v1/vision/classify
- Add support for base64 and URL images
- Include confidence scores in response
- Add unit tests for classifier"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Formatting
- `refactor:` - Code restructuring
- `test:` - Tests
- `chore:` - Maintenance

#### 8. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Create a pull request with:
- Clear title and description
- Link to related issues
- Screenshots/examples if applicable
- Checklist of changes

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

## Development Guidelines

### Project Structure

Respect the existing structure:
- `services/` - Microservices code
- `training/` - Training scripts
- `configs/` - Configuration files
- `docs/` - Documentation
- `tests/` - Test files

### Adding New Services

1. Create service directory in `services/`
2. Implement FastAPI server
3. Add health check endpoint
4. Update `docker-compose.yml`
5. Add configuration in `configs/`
6. Document in `docs/`

### Adding New Models

1. Add configuration in `configs/`
2. Create training script in `training/`
3. Update service to load model
4. Add evaluation metrics
5. Document model architecture

### Documentation

Update documentation for:
- New features
- API changes
- Configuration options
- Architecture changes

Documentation locations:
- `README.md` - Overview
- `docs/` - Detailed guides
- Code docstrings - Implementation details
- `CHANGELOG.md` - Version history

## Testing

### Test Types

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test service interactions
3. **End-to-End Tests**: Test complete workflows

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_vision.py

# With coverage
pytest --cov=services --cov=training

# Verbose
pytest -v
```

### Writing Good Tests

```python
def test_feature():
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result.status == "success"
    assert len(result.items) > 0
```

## Code Review Process

1. Automated checks run on PR
2. Maintainers review code
3. Address feedback
4. Approval and merge

Review criteria:
- Code quality and style
- Test coverage
- Documentation
- Performance impact
- Security considerations

## Community

- **Discussions**: GitHub Discussions
- **Chat**: Discord server
- **Email**: releaf-ai@example.com

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

Feel free to ask questions in:
- GitHub Discussions
- Discord
- Issue comments

Thank you for contributing to ReleAF AI! 🌱

