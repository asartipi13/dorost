import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_get_resume():
    mock_file = "test_df.csv"
    test_file = open(mock_file, "rb")

    response = client.post("/resume/", files={"file": test_file})
    assert response.status_code == 200

    data = response.json()
    assert "pred_id" in data
    assert "pred_name" in data

    test_file.close()