# Installation directement sur la machine 

- Creer un environnement python et activez le avec les commandes : \
_python -m venv venv_

- Installer les dépendances de l'application : \
_pip install -r requirements.txt_

- Démarrez l'API avec la commande : \
_uvicorn main:app --reload_

# Installation via Docker

Se placer dans le dossier de l'application et executer les commandes suivantes :
- Construire l'image Docker\
_docker build -t fastapi-keras-app ._

- Exécuter le conteneur Docker\
_docker run -d --name fastapi-keras-container -p 8000:80 fastapi-keras-app_

# Utilisation 
Ouvrez votre navigateur et entrez l'adresse **http://127.0.0.1:8000/docs** \
pour accéder à la documentation de l'API. Vous y trouverez les détails sur \
les appels de notre API de reconnaissance des chiffres à partir de leur audios.\
**NB :** Si le port 8000 est occupé, vous pouvez le changer avec un autre port libre de votre système.