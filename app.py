from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)


MODELS = {
    "udi": joblib.load("models/best_decision_tree_model.pkl"),
    "more": joblib.load("models/best_decision_tree_model_UDI_more.pkl"),
    "less": joblib.load("models/best_decision_tree_model_UDI_less.pkl")
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    target_model = None
    
    if request.method == "POST":
        try:
            
            features = [
                float(request.form.get("wwr")),
                float(request.form.get("tvis")),
                float(request.form.get("n_aletas")),
                float(request.form.get("angulo")),    
                float(request.form.get("largura")),
                float(request.form.get("direcao"))
            ]
            
            tipo_modelo = request.form.get("tipo_modelo")
            model = MODELS.get(tipo_modelo)
            
            if model:
                res = model.predict([features])
                prediction = round(res[0], 2)
                target_model = tipo_modelo.upper()
        except Exception as e:
            prediction = f"Erro: {e}"

    return render_template("index.html", prediction=prediction, target=target_model)

if __name__ == "__main__":
    app.run(debug=True)