# Shadow CI

A Python package for scientific code development.

## Features

- Modern Python package structure with `src` layout
- Scientific computing dependencies (NumPy, SciPy, Pandas, Matplotlib)
- Testing infrastructure with pytest
- Code quality tools (black, ruff, mypy)
- Jupyter notebook support

## Installation

### Basic Installation

```bash
pip install -e .
```

### Development Installation

Install with all development dependencies:

```bash
pip install -e ".[dev]"
```

### Optional Dependencies

Install with machine learning tools:

```bash
pip install -e ".[ml]"
```

Install with advanced visualization tools:

```bash
pip install -e ".[viz]"
```

Install everything:

```bash
pip install -e ".[all]"
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

Format code with black:

```bash
black src/ tests/
```

Lint code with ruff:

```bash
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Project Structure

```
shadow-ci/
├── src/
│   └── shadow_ci/          # Main package code
│       └── __init__.py
├── tests/                   # Test files
│   └── __init__.py
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # This file
└── .gitignore             # Git ignore patterns
```

## Usage

```python
import shadow_ci

# Your scientific code here
```

## Contributing

1. Install development dependencies: `pip install -e ".[dev]"`
2. Create a new branch for your feature
3. Make your changes
4. Run tests: `pytest`
5. Format code: `black src/ tests/`
6. Check linting: `ruff check src/ tests/`
7. Submit a pull request

## License

MIT License
