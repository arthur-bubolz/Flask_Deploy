import predictor
from flask import Flask, render_template, request


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Pegar os dados do formulário
        field1 = request.form.get('field1')
        field2 = request.form.get('field2')
        field3 = request.form.get('field3')
        field4 = request.form.get('field4')
        field5 = request.form.get('field5')

        # Colocar os dados em uma lista
        data_list = [field1, field2, field3, field4, field5]
        prediction = predictor.get_the_prediction(data_list)

        # Retornar a lista para o navegador
        return render_template('index.html', prediction=prediction)

    return render_template('index.html', prediction='Insira as informações!')

if __name__ == '__main__':
    app.run(debug=True)
