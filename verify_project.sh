#!/bin/bash

echo "=========================================="
echo "PROJECT VERIFICATION SCRIPT"
echo "Adaptive Spectral Rewiring for Heterogeneous Molecular Graphs"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

pass=0
fail=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((pass++))
    else
        echo -e "${RED}✗${NC} $1"
        ((fail++))
    fi
}

# 1. Core files exist
echo "1. Checking core project files..."
[ -f "requirements.txt" ]; check "requirements.txt exists"
[ -f "pyproject.toml" ]; check "pyproject.toml exists"
[ -f "LICENSE" ]; check "LICENSE exists"
[ -f "README.md" ]; check "README.md exists"
[ -f ".gitignore" ]; check ".gitignore exists"
echo ""

# 2. Configuration files
echo "2. Checking configuration files..."
[ -f "configs/default.yaml" ]; check "configs/default.yaml exists"
[ -f "configs/ablation.yaml" ]; check "configs/ablation.yaml exists"
echo ""

# 3. Scripts
echo "3. Checking executable scripts..."
[ -f "scripts/train.py" ]; check "scripts/train.py exists"
[ -f "scripts/evaluate.py" ]; check "scripts/evaluate.py exists"
[ -f "scripts/predict.py" ]; check "scripts/predict.py exists"
echo ""

# 4. Source modules
echo "4. Checking source modules..."
[ -f "src/adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs/__init__.py" ]; check "Package __init__.py"
[ -f "src/adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs/data/loader.py" ]; check "data/loader.py"
[ -f "src/adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs/data/preprocessing.py" ]; check "data/preprocessing.py"
[ -f "src/adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs/models/model.py" ]; check "models/model.py"
[ -f "src/adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs/models/components.py" ]; check "models/components.py"
[ -f "src/adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs/training/trainer.py" ]; check "training/trainer.py"
[ -f "src/adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs/evaluation/metrics.py" ]; check "evaluation/metrics.py"
[ -f "src/adaptive_spectral_rewiring_for_heterogeneous_molecular_graphs/evaluation/analysis.py" ]; check "evaluation/analysis.py"
echo ""

# 5. Tests
echo "5. Checking test files..."
[ -f "tests/conftest.py" ]; check "tests/conftest.py"
[ -f "tests/test_data.py" ]; check "tests/test_data.py"
[ -f "tests/test_model.py" ]; check "tests/test_model.py"
[ -f "tests/test_training.py" ]; check "tests/test_training.py"
echo ""

# 6. Python syntax
echo "6. Validating Python syntax..."
python3 -m py_compile scripts/train.py 2>/dev/null; check "train.py compiles"
python3 -m py_compile scripts/evaluate.py 2>/dev/null; check "evaluate.py compiles"
python3 -m py_compile scripts/predict.py 2>/dev/null; check "predict.py compiles"
echo ""

# 7. YAML syntax
echo "7. Validating YAML syntax..."
python3 -c "import yaml; yaml.safe_load(open('configs/default.yaml'))" 2>/dev/null; check "default.yaml is valid"
python3 -c "import yaml; yaml.safe_load(open('configs/ablation.yaml'))" 2>/dev/null; check "ablation.yaml is valid"
echo ""

# 8. No scientific notation in YAML
echo "8. Checking YAML format..."
! grep -q "e-[0-9]" configs/*.yaml; check "No scientific notation in YAML files"
echo ""

# 9. Directories
echo "9. Checking directory structure..."
[ -d "models" ]; check "models/ directory exists"
[ -d "results" ]; check "results/ directory exists"
[ -d "data" ] || mkdir -p data; check "data/ directory exists"
echo ""

echo "=========================================="
echo "VERIFICATION COMPLETE"
echo "=========================================="
echo -e "Passed: ${GREEN}$pass${NC}"
echo -e "Failed: ${RED}$fail${NC}"
echo ""

if [ $fail -eq 0 ]; then
    echo -e "${GREEN}All checks passed! Project is ready.${NC}"
    exit 0
else
    echo -e "${RED}Some checks failed. Please review.${NC}"
    exit 1
fi
