# DESIGN.md — Transcribe (Clay)

## Direction
Claymorphism froid (25-09-2026, remplace « Sky Aero »). Clair lavande par défaut, sombre ardoise-violet en option.

## Color (tokens `:root` / `[data-theme="dark"]` dans static/index.html)
- Fond `--bg` #e8ebf8 / #1d1f30 ; surfaces `--surface`, `--surface-2`.
- Accent violet `--accent` #6c5ce7 (dark #8b7df6), dégradé depuis `--accent-hi`.
- Succès menthe `--mint`, erreur `--bad`. Encre `--ink` / `--ink-soft` / `--ink-faint`.

## Typography
- `ui-rounded` (SF Pro Rounded, local, zéro réseau) ; `ui-monospace` pour timecodes/chiffres.
- Titres 800, labels 700.

## Surfaces
- `--clay` (cartes), `--clay-sm` (boutons/puces), `--clay-in` (champs, pistes, zones creusées), `--clay-accent` (CTA violet).
- Radius : 36 cartes, 26 zones, 18 champs, pilules pour boutons.

## Motion
- thinking-orbs vendorisé (`static/vendor/thinking-orbs/`, MIT, v0.3.2, moteur 2D canvas) : une boucle rAF partagée, pause si onglet caché, frame statique en reduced-motion. Orbe teinté `--accent`.
- États : idle=breathing, drag=searching, source choisie=shaping, download=connecting, audio=working, modèle=solving, transcription=listening, écriture=composing.
- Blobs de fond en CSS (transform only, 22-32 s). Ressort `--spring` sur press/hover.
