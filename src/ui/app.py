from flask import Flask, request, jsonify
from amt.encoder import HypervectorEncoder
from amt.adf_update import ADFMemory

app = Flask(__name__)
mem = ADFMemory(dim=6)
enc = HypervectorEncoder(dim=6)


@app.route("/encode")
def encode():
    txt = request.args.get("text", "")
    return jsonify(hv=enc.encode(txt).tolist())


@app.route("/update", methods=["POST"])
def update():
    js = request.json
    hv = enc.encode(js["text"])
    mem.update(hv, js.get("positive", True))
    return jsonify(status="ok")


@app.route("/similarity")
def similarity():
    hv = enc.encode(request.args.get("text", ""))
    pos, neg = mem.similarity_table([hv])[0]
    return jsonify(positive=pos, negative=neg)


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
