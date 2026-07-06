# pyrefly: ignore [missing-import]
import numpy as np, torch, json
from phase2.adapter_ridge import RidgeAdapter
from phase2.loader import load_split

def extract():
    for subj in ["CSI1", "CSI2", "CSI3", "CSI4"]:
        try:
            split = load_split(subject=subj, mode="bold5000", loader_kwargs={})
            adapter = RidgeAdapter.load(f"resultados/phase2_outputs/adapter/{subj}/ridge_adapter.joblib")     
            m = adapter.evaluate(split.betas_test, split.clip_test)
            print(f"[{subj}] R2_macro: {m.r2_macro:+.4f} | Cosine: {m.cosine_mean:+.4f} | MSE: {m.mse:.4f}")
        except Exception as e:
            print(f"[{subj}] Error: {e}")

if __name__ == "__main__":
    extract()
