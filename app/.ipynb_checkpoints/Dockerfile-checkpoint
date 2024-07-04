# Utiliser une image de base Python
FROM python:3.12.3

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de l'application
COPY . /app

# Installer les dépendances
RUN pip install -r requirements.txt

# Exposer le port de l'application
EXPOSE 80

# Commande pour démarrer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
