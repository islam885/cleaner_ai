# Contributing to Cleaner AI

## Development Setup

1. Fork the repository
2. Clone your fork
3. Install dependencies: `pip install -r requirements.txt`
4. Run quick start: `python quick_start.py`

## Code Style

- Follow PEP 8
- Use type hints where possible
- Keep functions focused and small
- No comments in production code (self-documenting code)

## Project Structure

```
cleaner_ai/
├── Core Modules
│   ├── model.py              # Neural network architecture
│   ├── train.py              # Training pipeline
│   ├── infer.py              # Inference engine
│   └── utils.py              # Utility functions
├── Data Pipeline
│   ├── captcha_generator.py  # Dataset generation
│   ├── augmentation.py       # Data augmentation
│   └── dataset_cache.py      # Caching system
├── GUI & CLI
│   ├── cleaner_ai.py         # Main GUI application
│   ├── train_cli.py          # CLI training
│   ├── infer_cli.py          # CLI inference
│   └── generate_cli.py       # CLI generation
├── Tools
│   ├── validate.py           # Model validation
│   ├── benchmark.py          # Performance testing
│   ├── export_model.py       # Model export
│   └── config.py             # Configuration management
└── Setup
    ├── quick_start.py        # Quick setup script
    ├── setup.bat             # Windows installer
    └── run.bat               # Windows launcher
```

## Adding Features

### New Model Architecture

1. Create new class in `model.py`
2. Inherit from `nn.Module`
3. Implement `forward()` method
4. Update `Trainer` to use new model

### New Noise Type

1. Add method to `CaptchaGenerator`
2. Update `add_noise()` method
3. Add to noise type list
4. Update metadata encoding

### New Augmentation

1. Add to `AugmentationPipeline` in `augmentation.py`
2. Configure probability
3. Test with sample images

### New Loss Function

1. Create loss class in `train.py`
2. Update `Trainer.__init__()`
3. Update loss calculation in training loop

## Testing

### Unit Tests

```python
python -m pytest tests/
```

### Integration Tests

```bash
python quick_start.py --test
```

### Benchmark

```bash
python benchmark.py
```

## Performance Guidelines

- Use vectorized operations (NumPy/PyTorch)
- Avoid Python loops for large data
- Use GPU when available
- Profile before optimizing

## Memory Management

- Clear cache when not needed
- Use generators for large datasets
- Monitor GPU memory usage
- Implement batch processing

## Documentation

- Update README.md for user-facing changes
- Update ARCHITECTURE.md for technical changes
- Update USAGE.md for new features
- Keep docstrings current

## Pull Request Process

1. Create feature branch
2. Make changes
3. Test thoroughly
4. Update documentation
5. Submit PR with description

## Code Review Checklist

- [ ] Code follows style guide
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No performance regression
- [ ] Memory usage acceptable
- [ ] GPU/CPU compatible

## Release Process

1. Update version number
2. Update CHANGELOG
3. Test on clean environment
4. Create release tag
5. Build distribution

## Bug Reports

Include:
- OS and Python version
- GPU/CPU info
- Error message
- Steps to reproduce
- Expected vs actual behavior

## Feature Requests

Include:
- Use case description
- Expected behavior
- Alternative solutions considered
- Willingness to implement

## Community

- Be respectful
- Help others
- Share knowledge
- Contribute positively

## License

By contributing, you agree to license your contributions under the MIT License.
