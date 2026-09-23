from fastapi.testclient import TestClient
from api import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Netflix Prediction API is running"


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "release_year": 2019,
            "runtimeMinutes": 98,
            "averageRating": 6.4,
            "numVotes": 10000
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert data["prediction"] in ["Movie", "TV Show"]