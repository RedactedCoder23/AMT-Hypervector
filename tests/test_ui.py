from ui.app import app


def test_encode_endpoint():
    client = app.test_client()
    resp = client.get("/encode", query_string={"text": "hello"})
    assert resp.status_code == 200

    # update memory and query similarity
    resp = client.post("/update", json={"text": "hello", "positive": True})
    assert resp.status_code == 200
    resp = client.get("/similarity", query_string={"text": "hello"})
    assert resp.status_code == 200
