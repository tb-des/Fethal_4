from flask import Flask, render_template, request

app = Flask(__name__)

# Une fonction Python que vous voulez utiliser
def ma_fonction_python(valeur):
    valeur = str(int(valeur) + 250)
    return f"Vous avez entré : {valeur.upper()}"

# Route pour afficher la page HTML
@app.route('/')
def index():
    return render_template('index.html')

# Route pour gérer le formulaire et utiliser la fonction Python
@app.route('/resultat', methods=['POST'])
def resultat():
    valeur = request.form['input_valeur']  # Récupère les données envoyées par l'utilisateur
    resultat = ma_fonction_python(valeur)  # Appelle la fonction Python
    return render_template('index.html', resultat=resultat)

if __name__ == '__main__':
    app.run(debug=True)
