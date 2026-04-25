def exact_match(pred, truth):
    try:
        return truth.lower().strip() in pred.lower().strip()
    except:
        return False