
"""
Hallucination Detection — Final Integrated Solution
"""

import os
import time
import gc
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from aggregation import aggregation_and_feature_extraction
from evaluate import print_summary, run_evaluation, save_predictions, save_results
from model import MAX_LENGTH, get_model_and_tokenizer
from probe import HallucinationProbe
from splitting import split_data

DATA_FILE     = "./data/dataset.csv"
OUTPUT_FILE   = "results.json"
BATCH_SIZE    = 1
USE_GEOMETRIC = True
TEST_FILE        = "./data/test.csv"
PREDICTIONS_FILE = "predictions.csv"
CACHE_FILE       = "cached_advanced_features.npz"

if __name__=='__main__':
    torch.cuda.empty_cache()
    gc.collect()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    print(f"📊 Batch size: {BATCH_SIZE}")
    print(f"🔧 Geometric features: {USE_GEOMETRIC}")

    df = pd.read_csv(DATA_FILE)
    all_texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    all_labels = np.array([int(float(h)) for h in df["label"]])

    print(f"✅ Loaded {len(all_labels)} samples")

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    if os.path.exists(CACHE_FILE):
        print("📦 Loading cached features...")
        cached = np.load(CACHE_FILE, allow_pickle=True)
        X = cached['X']
        y = cached['y']
    else:
        print("🔄 Extracting advanced features...")
        all_features = []
        t0 = time.time()

        for idx in tqdm(range(len(all_texts)), desc="Processing"):
            text = all_texts[idx]
            
            encoding = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

            hidden = torch.stack(outputs.hidden_states, dim=1).float()
            mask = attention_mask.cpu()

            feat = aggregation_and_feature_extraction(
                hidden[0], mask[0], use_geometric=USE_GEOMETRIC,
                input_ids=input_ids[0], model=model
            )
            all_features.append(feat.cpu())
            
            del outputs, hidden, input_ids, attention_mask
            torch.cuda.empty_cache()

        X = np.vstack([f.numpy() for f in all_features])
        y = all_labels
        
        np.savez_compressed(CACHE_FILE, X=X, y=y)
        print(f"💾 Features cached: {X.shape}")

    print(f"📐 Feature dimensions: {X.shape[1]}")
    
    splits = split_data(y, df)
    fold_results = run_evaluation(splits, X, y, HallucinationProbe)
    print_summary(fold_results, X.shape[1], len(X), 0)
    save_results(fold_results, X.shape[1], len(X), 0, OUTPUT_FILE)

    # Test predictions
    df_test = pd.read_csv(TEST_FILE)
    test_texts = [f"{row['prompt']}{row['response']}" for _, row in df_test.iterrows()]
    
    test_features = []
    for idx in tqdm(range(len(test_texts)), desc="Test"):
        text = test_texts[idx]
        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        feat = aggregation_and_feature_extraction(hidden[0], attention_mask.cpu()[0], use_geometric=USE_GEOMETRIC)
        test_features.append(feat.cpu())
        
        del outputs, hidden, input_ids, attention_mask
        torch.cuda.empty_cache()

    X_test = np.vstack([f.numpy() for f in test_features])
    
    final_probe = HallucinationProbe()
    final_probe.fit(X, y)
    
    predictions = final_probe.predict(X_test)
    
    pd.DataFrame({'id': df_test.index, 'label': predictions}).to_csv(PREDICTIONS_FILE, index=False)
    print(f"\n✅ Saved {PREDICTIONS_FILE}")
    print(f"Hallucination rate: {predictions.mean():.1%}")
