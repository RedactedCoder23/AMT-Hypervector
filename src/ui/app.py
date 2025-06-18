"""Flask web UI exposing BHRE operations."""

from flask import Flask, jsonify, request

from amt.encoder import HypervectorEncoder
from amt.adf_update import ADFMemory

app = Flask(__name__)
encoder = HypervectorEncoder(dim=6, alpha=[1.0] * 6)
memory = ADFMemory(dim=6)


@app.route("/")
def index() -> str:
    return "OK"


@app.get("/encode")
def encode() -> any:
    text = request.args.get("text", "")
    hv = encoder.encode(text)
    return jsonify(hv.tolist())


@app.post("/update")
def update() -> any:
    data = request.get_json(force=True)
    hv = encoder.encode(data.get("text", ""))
    memory.update(hv, positive=bool(data.get("positive", True)))
    return jsonify({"status": "updated"})


@app.get("/similarity")
def similarity() -> any:
    text = request.args.get("text", "")
    hv = encoder.encode(text)
    scores = memory.similarity_table([hv])
    return jsonify(scores)


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
