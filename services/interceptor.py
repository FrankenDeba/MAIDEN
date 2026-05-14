from minja_infer import detect

def intercept(prompt):
    predicted_label, probs, similarity, action = detect(prompt)
    return {
        "predicted_label": predicted_label,
        "probs": probs,
        "similarity": similarity,
        "action": action
    }

__all__ = ["intercept"]