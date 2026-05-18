# ============================================================
# STRICT 2-STAGE VERIFICATION SCRIPT (BGE Radar -> SLM Judge)
# ============================================================

import pickle
import faiss
import numpy as np
import os
import httpx
from sentence_transformers import SentenceTransformer
import re

from services.slm import query_slm

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = os.path.abspath("minja-bge-base-finetuned")

# The singular line in the sand. 
# Anything below this is Green. Anything above is Red (Send to SLM).
BGE_RED_FLAG_THRESHOLD = 0.60

print("[+] Loading fine-tuned model...")


# SentenceTransformer(MODEL_PATH)

# ============================================================
# LOAD BGE ML ASSETS
# ============================================================
print("[+] Loading BGE fine-tuned model...")
model = SentenceTransformer("deba2012ddx/minja_bge_ft")

print("[+] Loading classifier...")
with open("classifier.pkl", "rb") as f:
    clf = pickle.load(f)

print("[+] Loading FAISS index...")
index = faiss.read_index("faiss.index")

# ============================================================
# SLM JUDGE FUNCTION (The Supreme Court)
# ============================================================


def slm_evaluate(text: str) -> bool:
    """
    Called ONLY when BGE throws a Red Flag. 
    Uses a hybrid approach: Python rules for structural encoding, 
    and an anchored Few-Shot SLM for semantic text.
    """
    print("\n[SLM JUDGE] Evaluating with SLM...")
    slm_response = query_slm(text)
    # print(type(slm_response))
    return slm_response

# ============================================================
# DETECTION CASCADE
# ============================================================
def detect(text: str, top_k=5) -> bool:
    """
    Returns True if Malicious, False if Safe.
    """
    print("\n================================================")
    print("INPUT:")
    print(text)

    text_lower = text.strip().lower()

    # ------------------------------------------------
    # LAYER 1 & 2: HEURISTICS (Speed & Obvious Threats)
    # ------------------------------------------------
    safe_starts = ["hello", "hi", "hey", "can you write", "what is", "calculate"]
    if any(text_lower.startswith(start) for start in safe_starts) and len(text_lower) < 150:
        print("\nACTION: ALLOW (Bypassed via Layer 1 Heuristics)")
        return False

    blatant_attacks = ["system override", "ignore previous", "bypass safety"]
    if any(attack in text_lower for attack in blatant_attacks):
        print("\nACTION: REJECT (Blocked via Layer 2 Heuristics)")
        return True

    # ------------------------------------------------
    # LAYER 3: BGE SEMANTIC RADAR
    # ------------------------------------------------
    embedding = model.encode([text], normalize_embeddings=True).astype("float32")
    
    probs = clf.predict_proba(embedding)[0]
    D, I = index.search(embedding, top_k)
    
    similarity = float(D[0][0])
    malicious_prob = probs[2] if len(probs) > 2 else probs[-1]
    
    # Unified score based on maximum threat detected
    threat_score = min(max(similarity, malicious_prob), 1.0)
    
    print(f"\n[BGE RADAR] Threat Score: {threat_score:.4f}")

    # ------------------------------------------------
    # LAYER 4: STRICT ROUTING LOGIC
    # ------------------------------------------------
    if threat_score >= BGE_RED_FLAG_THRESHOLD:
        # BGE threw a Red Flag. It does NOT have permission to block. 
        # It MUST ask the SLM.
        is_malicious = slm_evaluate(text)

        if is_malicious:
            print("\nFINAL ACTION: REJECT (SLM Confirmed Threat)")
            return True
        else:
            print("\nFINAL ACTION: ALLOW (SLM Overrode BGE Grudge)")
            return False

    else:
        # BGE threw a Green Flag. Let it pass.
        print("\nFINAL ACTION: ALLOW (Cleared by BGE)")
        return False

__all__ = ["detect"]