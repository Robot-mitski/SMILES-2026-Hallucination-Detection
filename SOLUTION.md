# Hallucination Detection Solution

## Reproducibility

```bash
git clone https://github.com/ahdr3w/SMILES-HALLUCINATION-DETECTION.git
cd SMILES-HALLUCINATION-DETECTION
pip install -r requirements.txt
python solution.py
```

## Final Solution

### Architecture
- Feature extraction: Layers 12-23 with weighted mean pooling + text features
- Preprocessing: RobustScaler + PCA (95% variance)
- Model: Voting ensemble (3 LogisticRegression + SVC + MLP)
- Threshold: Optimized to maintain class distribution

### Key Findings
- Hallucinated responses are 1.9x longer (790 vs 418 characters)
- Response length is the strongest single predictor
- Simple linear models outperform complex ones on 689 samples

### Results
- Validation Accuracy: ~74%
- PCA components: 345
- Explained variance: 95.0%

### Failed Approaches
- XGBoost/RandomForest: Severe overfitting (99% train / 74% val)
- Stacking with meta-classifier: No improvement
- Perplexity features: Unstable between runs