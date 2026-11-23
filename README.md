# Transcribe

App Mac de transcription **100 % locale** : tu déposes une vidéo ou un audio, ou tu colles un lien (YouTube, TikTok, Instagram…), et tu récupères le texte, un `.srt` et un `.json`. Whisper tourne sur la puce Apple (MLX). Rien ne quitte ton Mac.

*Local, private speech-to-text app for Apple Silicon Macs (MLX Whisper). French UI.*

## Télécharger

👉 **[Transcribe-mac-arm64.zip](https://github.com/YesStonksMan/transcribe/releases/latest/download/Transcribe-mac-arm64.zip)** (page des versions : [Releases](https://github.com/YesStonksMan/transcribe/releases))

Il te faut un **Mac avec puce Apple (M1 ou plus récent)** et macOS 12 ou plus récent.

## Installer

1. Décompresse le zip et glisse **Transcribe.app** dans **Applications**.
2. Au premier lancement, macOS bloque l'app : elle n'est pas signée par un compte développeur Apple.
   Ouvre **Réglages Système › Confidentialité et sécurité**, descends en bas et clique **Ouvrir quand même**.
   Sinon, dans le Terminal :
   ```bash
   xattr -dr com.apple.quarantine /Applications/Transcribe.app
   ```
3. Installe les deux outils dont l'app a besoin. Avec [Homebrew](https://brew.sh) :
   ```bash
   brew install ffmpeg yt-dlp
   ```
   (Si Homebrew est déjà installé, l'app propose aussi de les installer d'un clic dans ses Réglages.)

La première transcription télécharge le modèle Whisper large-v3 (environ 3 Go). Les suivantes démarrent tout de suite.

## Utiliser

- **Fichier** : glisse une vidéo ou un audio dans la fenêtre.
- **Lien** : colle une URL. Pour Instagram ou une vidéo privée, sois connecté au site dans ton navigateur (Brave, Chrome, Safari, Firefox…) : l'app emprunte ta session. Pour Safari, il faut donner à Transcribe l'**Accès complet au disque** (Réglages Système › Confidentialité et sécurité).
- **Vocabulaire personnalisé** (Réglages) : noms propres et jargon, pour qu'ils soient bien orthographiés.
- Résultat : copier le texte, ou exporter en `.md`, `.srt`, `.json`.

## Brancher à une IA (MCP)

L'app embarque un serveur [MCP](https://modelcontextprotocol.io), le standard que suivent Claude, Cursor, VS Code, LM Studio et d'autres. Une fois branchée, ton IA peut transcrire un fichier ou un lien elle-même (« transcris cette vidéo YouTube »). Elle ouvre l'app toute seule si besoin.

Le plus simple : dans l'app, **Réglages › Brancher à une IA**, puis copie la commande ou la config.

**Claude Code** (Terminal) :
```bash
claude mcp add --scope user transcribe -- /Applications/Transcribe.app/Contents/MacOS/python /Applications/Transcribe.app/Contents/Resources/mcp_transcribe.py
```

**Claude Desktop, Cursor et la plupart des autres** : ajoute ceci à leur fichier de config MCP (Claude Desktop : *Settings › Developer › Edit Config* ; Cursor : `~/.cursor/mcp.json`), puis redémarre l'app :
```json
{
  "mcpServers": {
    "transcribe": {
      "command": "/Applications/Transcribe.app/Contents/MacOS/python",
      "args": ["/Applications/Transcribe.app/Contents/Resources/mcp_transcribe.py"]
    }
  }
}
```

Outils fournis : `transcrire_fichier`, `transcrire_url`, `suivre_transcription`, `lister_modeles`, `choisir_modele_defaut`.

**IA sans MCP ou script** : l'app répond aussi en HTTP sur `http://127.0.0.1:7878` (en local seulement). `POST /api/jobs/local` avec `{"path": "/chemin/video.mp4", "language": "fr"}` ou `POST /api/jobs/url` avec `{"url": "…"}` renvoie un `job_id`. `GET /api/jobs/<job_id>` donne l'avancement, puis les fichiers produits.

## Mises à jour

Quand une nouvelle version sort, un bouton **« Version x.y dispo »** apparaît en haut de l'app : il ouvre la page de téléchargement. Remplace l'ancienne app dans Applications par la nouvelle (tes réglages sont gardés).
C'est la seule connexion que fait l'app d'elle-même : elle demande à GitHub le numéro de la dernière version. Tes fichiers et transcriptions ne quittent jamais ton Mac.

## Lancer depuis le code

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install mlx-whisper fastapi uvicorn python-multipart pywebview pytest httpx
TRANSCRIBE_VENV="$PWD/.venv" ./run.sh      # ouvre http://localhost:7878
```

Options avancées (facultatives, l'app marche sans) :
- `TRANSCRIBE_VAD_PYTHON` : un Python avec `torch` + `silero-vad`, pour découper l'audio sur les silences (meilleur sur les longs fichiers).
- `TRANSCRIBE_MLX_AUDIO_PYTHON` : un Python avec `mlx-audio`, pour les modèles non-Whisper (Qwen3-ASR…).

Construire l'app : `TRANSCRIBE_VENV=/chemin/du/venv ./build.sh --no-install` (avec `py2app` installé), résultat dans `dist/`.

Tests : `python -m pytest`.

## Crédits

- [MLX Whisper](https://github.com/ml-explore/mlx-examples) (Apple), [yt-dlp](https://github.com/yt-dlp/yt-dlp), [FFmpeg](https://ffmpeg.org).
- Orbes animés : [thinking-orbs](static/vendor/thinking-orbs/LICENSE), Jakub Antalik, licence MIT.

## Licence

MIT, voir [LICENSE](LICENSE).
