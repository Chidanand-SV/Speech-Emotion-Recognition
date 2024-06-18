from warnings import filterwarnings
filterwarnings('ignore')

from emotions import app

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
