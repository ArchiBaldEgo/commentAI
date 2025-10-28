import os
from fastapi.testclient import TestClient
from src.sentiment.server import create_app


def test_predict_and_feedback(tmp_path):
    model_dir = tmp_path / "online"
    os.makedirs(model_dir, exist_ok=True)
    app = create_app(str(model_dir))
    client = TestClient(app)

    # Health
    r = client.get("/health")
    assert r.status_code == 200

    # Predict
    r = client.post("/predict", json={"texts": ["Отличный товар", "Очень плохо"]})
    assert r.status_code == 200
    data = r.json()
    assert "items" in data and len(data["items"]) == 2

    # Feedback single
    r = client.post("/feedback", json={"text": "Отличный товар", "rating": 5})
    assert r.status_code == 200

    # Feedback bulk
    r = client.post(
        "/feedback_bulk",
        json={
            "items": [
                {"text": "Хорошее качество", "label": "pos"},
                {"text": "Сломался быстро", "rating": 1},
            ]
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["received"] == 2
    assert out["stored"] == 2
    assert out["trained"] >= 0

    # Train batch from stored
    r = client.post("/train_batch", json={"limit": 10, "clear_after": True})
    assert r.status_code == 200
    tb = r.json()
    assert "seen" in tb and "trained" in tb and "remaining" in tb
