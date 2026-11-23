"""py2app build configuration for Transcribe.app.

Usage:
    python setup.py py2app          # production bundle in dist/Transcribe.app
    python setup.py py2app -A       # alias bundle (fast dev, symlinks source)
"""
import sys

# py2app's modulegraph blows the stack on big packages (numba, llvmlite). Bump
# the recursion limit before the build kicks off.
sys.setrecursionlimit(5000)

from setuptools import setup

APP = ['app.py']

DATA_FILES = [
    ('static', ['static/index.html']),
    ('static/vendor/thinking-orbs', [
        'static/vendor/thinking-orbs/engine.js',
        'static/vendor/thinking-orbs/engine-core.js',
        'static/vendor/thinking-orbs/LICENSE',
    ]),
]

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'icon.icns',
    'plist': {
        'CFBundleName': 'Transcribe',
        'CFBundleDisplayName': 'Transcribe',
        'CFBundleIdentifier': 'fr.adham.transcribe',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1',
        'NSHighResolutionCapable': True,
        'LSUIElement': False,
        'LSMinimumSystemVersion': '12.0',
        # Defense-in-depth against the ASCII-locale decode bug: GUI launches
        # (Dock/Finder/Spotlight) inherit a minimal env with no LANG/LC_*, so
        # the interpreter's preferred encoding falls back to ASCII and any
        # text-mode subprocess read of accented yt-dlp/ffprobe output crashes
        # (UnicodeDecodeError 0xc3). LSEnvironment is applied by LaunchServices
        # BEFORE the interpreter starts, so PYTHONUTF8=1 forces UTF-8 mode and
        # LANG/LC_ALL give children (yt-dlp) a sane UTF-8 locale. The explicit
        # encoding="utf-8" on each subprocess call is the primary fix; this is
        # the belt-and-suspenders layer that also covers any path we missed.
        'LSEnvironment': {
            'PYTHONUTF8': '1',
            'LANG': 'en_US.UTF-8',
            'LC_ALL': 'en_US.UTF-8',
        },
    },
    # `packages` = copied wholesale into the bundle (preserves non-Python files
    # like dylibs). `mlx` is a PEP 420 namespace package (no __init__.py) so
    # modulegraph chokes on it — copy it via post-build script (see build.sh).
    # `mlx_whisper` IS copied here because it has __init__.py (regular package)
    # and pulls in the modelling code.
    'packages': [
        'fastapi',
        'starlette',
        'pydantic',
        'pydantic_core',
        'uvicorn',
        'webview',
        'mlx_whisper',
        'anyio',          # dynamic _backends._asyncio loaded at runtime
        'sniffio',
        'h11',
        'httptools',
        'click',
    ],
    'includes': [
        'server',
        'transcribe',
        'tiktoken',
    ],
    'excludes': [
        'tkinter',
        'pytest',
        'IPython',
    ],
}

setup(
    name='Transcribe',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
