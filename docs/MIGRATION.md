# Migration Guide

This guide explains how to migrate from the current file structure to the new organized structure.

## Current vs New Structure

### Current Structure
```
coin_game/
├── analysis.ipynb
├── visualization.ipynb
├── example_visualization.py
├── visualize_rllib_models.py
├── training_pruebas.ipynb
├── training.py
├── training_RLLIB.py
├── launch_training.py
├── inputs/
└── logs/
```

### New Structure
```
src/
├── environments/
│   ├── coin_game.py
│   └── coin_game_rllib.py
├── training/
│   ├── trainer.py
│   └── config.py
└── analysis/
    ├── metrics.py
    └── visualization.py

examples/
├── basic_training.py
├── attitude_experiments.py
├── training_analysis.ipynb
└── visualization_demo.ipynb

configs/
├── attitudes/
│   ├── selfish.txt
│   ├── cooperative.txt
│   └── altruistic.txt
└── training_configs/
    ├── quick.yaml
    └── standard.yaml

scripts/
├── generate_attitudes.py
├── batch_training.py
└── analyze_results.py
```

## Migration Steps

### Step 1: Move Environment Files

```bash
# Move environment implementations
mv coin_game/coin_game.py src/environments/
mv coin_game/coin_game_rllib_env.py src/environments/coin_game_rllib.py
```

### Step 2: Move Training Files

```bash
# Move training implementations
mv coin_game/make_train.py src/training/trainer.py

# Move training scripts (keep for backward compatibility)
# These can stay in coin_game/ for now
```

### Step 3: Move Analysis Files

```bash
# Move analysis notebooks
mv coin_game/analysis.ipynb examples/training_analysis.ipynb
mv coin_game/visualization.ipynb examples/visualization_demo.ipynb

# Move analysis scripts
mv coin_game/visualize_rllib_models.py src/analysis/visualization.py
mv coin_game/example_visualization.py examples/example_visualization.py
```

### Step 4: Move Configuration Files

```bash
# Move input files
mv coin_game/inputs/* configs/attitudes/

# Create training configs (already done)
# configs/training_configs/quick.yaml
# configs/training_configs/standard.yaml
```

### Step 5: Update Import Paths

Update import statements in all files to reflect the new structure:

```python
# Old imports
from jaxmarl.environments.coin_game.make_train import make_train

# New imports
from src.training.trainer import make_train
from src.environments.coin_game import CoinGame
```

### Step 6: Update Scripts

Update the training scripts to use the new structure:

```python
# In training.py, update import
from src.training.trainer import make_train
```

## Backward Compatibility

To maintain backward compatibility during migration:

### 1. Keep Original Scripts

Keep the original training scripts in `coin_game/` with updated imports:

```python
# coin_game/training.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.trainer import make_train
# ... rest of the script
```

### 2. Create Symlinks (Optional)

Create symlinks for commonly used files:

```bash
# Create symlinks for backward compatibility
ln -s src/environments/coin_game.py coin_game/coin_game.py
ln -s src/training/trainer.py coin_game/make_train.py
```

### 3. Update Documentation

Update all documentation to reference the new structure while maintaining references to the old structure for backward compatibility.

## Testing the Migration

### 1. Test Basic Functionality

```bash
# Test basic training
python examples/basic_training.py

# Test attitude experiments
python examples/attitude_experiments.py

# Test script generation
python scripts/generate_attitudes.py --predefined
```

### 2. Test Backward Compatibility

```bash
# Test original scripts still work
python coin_game/training.py configs/attitudes/selfish.txt 0 0.001 3
```

### 3. Test Analysis

```bash
# Test analysis notebooks
jupyter notebook examples/training_analysis.ipynb
```

## File Mapping

| Old Location | New Location | Purpose |
|--------------|--------------|---------|
| `coin_game/coin_game.py` | `src/environments/coin_game.py` | Main environment |
| `coin_game/coin_game_rllib_env.py` | `src/environments/coin_game_rllib.py` | RLlib wrapper |
| `coin_game/make_train.py` | `src/training/trainer.py` | Training function |
| `coin_game/analysis.ipynb` | `examples/training_analysis.ipynb` | Analysis notebook |
| `coin_game/visualization.ipynb` | `examples/visualization_demo.ipynb` | Visualization demo |
| `coin_game/visualize_rllib_models.py` | `src/analysis/visualization.py` | Visualization tools |
| `coin_game/inputs/` | `configs/attitudes/` | Attitude configs |
| `coin_game/training.py` | `coin_game/training.py` | Keep for compatibility |
| `coin_game/launch_training.py` | `coin_game/launch_training.py` | Keep for compatibility |

## Benefits of New Structure

### 1. Better Organization
- Clear separation of concerns
- Logical grouping of related files
- Easier to find and maintain code

### 2. Improved Usability
- Clear examples for new users
- Standardized configuration format
- Automated scripts for common tasks

### 3. Enhanced Maintainability
- Modular design
- Clear dependencies
- Easier testing and debugging

### 4. Better Documentation
- Comprehensive guides
- API documentation
- Usage examples

## Troubleshooting

### Import Errors

If you encounter import errors:

1. Check that the `src/` directory is in the Python path
2. Verify that all files have been moved correctly
3. Update import statements to use the new paths

### Missing Files

If files are missing after migration:

1. Check the file mapping table above
2. Ensure all files were moved correctly
3. Restore from version control if needed

### Configuration Issues

If configurations don't work:

1. Verify that configuration files are in the correct format
2. Check that paths in configurations are updated
3. Test with the example configurations first

## Next Steps

After migration:

1. **Test Everything**: Run all examples and scripts
2. **Update Documentation**: Ensure all docs reflect the new structure
3. **Clean Up**: Remove old files once everything works
4. **Version Control**: Commit the new structure
5. **Share**: Update any shared documentation or instructions 