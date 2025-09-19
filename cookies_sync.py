import paramiko
import os

# === CONFIG ===
VPS_HOST = "51.178.80.188"
VPS_USER = "ubuntu"
VPS_PASS = "Kevjes@20@25"
REMOTE_ENV_PATH = "/var/www/html/gemini/.env"
SERVICE_NAME = "gemini"
NGINX_SERVICE = "nginx"

# === Fichier .env local (dans le même dossier que ce script) ===
LOCAL_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")

if not os.path.exists(LOCAL_ENV_PATH):
    raise FileNotFoundError("⚠️ Fichier .env introuvable dans ce dossier.")

print("[OK] Fichier .env trouvé, préparation transfert...")

# === Connexion SSH ===
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS)

# === Transfert fichier via SFTP ===
sftp = ssh.open_sftp()
sftp.put(LOCAL_ENV_PATH, REMOTE_ENV_PATH)
sftp.close()
print(f"[OK] .env copié sur le serveur ({REMOTE_ENV_PATH})")

# === Redémarrage des services ===
commands = [
    f"sudo systemctl restart {SERVICE_NAME}",
    f"sudo systemctl restart {NGINX_SERVICE}"
]

for cmd in commands:
    stdin, stdout, stderr = ssh.exec_command(cmd)
    stdout_text = stdout.read().decode().strip()
    stderr_text = stderr.read().decode().strip()
    if stdout_text:
        print(stdout_text)
    if stderr_text:
        print("⚠️", stderr_text)

print("[OK] Gemini et Nginx redémarrés avec succès !")

ssh.close()
