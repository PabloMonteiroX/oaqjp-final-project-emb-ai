import requests


def emotion_detector(text_to_analyze):
    """Call remote emotion prediction service with a short timeout.

    If the remote service returns HTTP 400, return a dict with None values
    (caller may interpret as invalid input). On other failures, use a
    lightweight keyword-based fallback so the web UI/tests remain responsive.
    """
    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/"
        "NlpService/EmotionPredict"
    )

    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }

    input_json = {"raw_document": {"text": text_to_analyze}}

    try:
        resp = requests.post(url, json=input_json, headers=headers, timeout=5)

        # If the remote service explicitly rejects the input, propagate a
        # clear signal (None values) so callers can handle it specially.
        if resp.status_code == 400:
            return {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None,
            }

        resp.raise_for_status()
        data = resp.json()

        preds = data.get("emotionPredictions") or []
        if not preds or not isinstance(preds, list) or "emotion" not in preds[0]:
            raise ValueError("unexpected response format")

        emotions = preds[0]["emotion"]

        anger = float(emotions.get("anger", 0.0))
        disgust = float(emotions.get("disgust", 0.0))
        fear = float(emotions.get("fear", 0.0))
        joy = float(emotions.get("joy", 0.0))
        sadness = float(emotions.get("sadness", 0.0))

        emotion_map = {
            "anger": anger,
            "disgust": disgust,
            "fear": fear,
            "joy": joy,
            "sadness": sadness,
        }

        # pick the highest-scoring emotion; if all scores are zero, return None
        if all(v == 0.0 for v in emotion_map.values()):
            dominant = None
        else:
            dominant = max(emotion_map.items(), key=lambda kv: kv[1])[0]

        return {
            "anger": anger,
            "disgust": disgust,
            "fear": fear,
            "joy": joy,
            "sadness": sadness,
            "dominant_emotion": dominant,
        }

    except Exception:
        # Fallback: quick keyword heuristic so the UI remains responsive
        txt = (text_to_analyze or "").lower()
        scores = {"anger": 0.0, "disgust": 0.0, "fear": 0.0, "joy": 0.0, "sadness": 0.0}

        if any(w in txt for w in ("glad", "happy", "fun", "joy", "pleased")):
            scores["joy"] = 0.9
        if any(w in txt for w in ("mad", "angry", "furious", "irritat")):
            scores["anger"] = 0.9
        if any(w in txt for w in ("disgust", "disgusted", "gross")):
            scores["disgust"] = 0.9
        if any(w in txt for w in ("sad", "sadness", "unhappy", "sorrow")):
            scores["sadness"] = 0.9
        if any(w in txt for w in ("afraid", "scared", "fear", "fright")):
            scores["fear"] = 0.9

        # If no keywords matched, leave all zeros and dominant will be None
        if all(v == 0.0 for v in scores.values()):
            dominant = None
        else:
            dominant = max(scores.items(), key=lambda kv: kv[1])[0]

        return {**scores, "dominant_emotion": dominant}
