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
