"""
Управление сессиями разметки: структура папок, имена, список недавних.

Модуль не зависит от Qt-виджетов, чтобы его можно было тестировать.
Qt (QStandardPaths) используется только для вычисления стандартного
пути конфига по умолчанию.
"""

import json
import re
from pathlib import Path
from typing import Optional

DEFAULT_SESSION_ROOT = Path.home() / "BacteriaLabeling"
SESSION_SUBDIRS = ("source", "masks", "cropped", "cropped_masks")

MAX_RECENT = 10
MAX_NAME_LEN = 60
INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*]')

CONFIG_FILENAME = "bacteria_analyzer.json"


def sanitize_name(name: str) -> Optional[str]:
    """Очистка имени сессии от символов, недопустимых в именах папок Windows.

    Возвращает None, если после очистки имя пусто.
    """
    if not name:
        return None
    cleaned = INVALID_FS_CHARS.sub("", name).strip()
    cleaned = cleaned[:MAX_NAME_LEN]
    return cleaned or None


def ensure_session_structure(session_dir: Path) -> Path:
    """Создаёт корень сессии и все внутренние поддиректории."""
    session_dir = Path(session_dir).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    for sub in SESSION_SUBDIRS:
        (session_dir / sub).mkdir(parents=True, exist_ok=True)
    return session_dir


def default_config_path() -> Path:
    """Стандартный путь к конфигурационному файлу приложения."""
    try:
        from PyQt6.QtCore import QStandardPaths

        base = Path(
            QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppConfigLocation
            )
        )
    except Exception:
        base = Path.home() / ".config" / "BacteriaAnalyzer"
    return base / CONFIG_FILENAME


class SessionManager:
    """Настройки сессий (расположение, список недавних) в JSON-файле."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = Path(config_path) if config_path else default_config_path()
        self._recent: list[dict] = []
        self._root: Path = DEFAULT_SESSION_ROOT
        self._load()

    @property
    def current_root(self) -> Path:
        """Текущий корень хранения сессий."""
        return self._root

    @property
    def recent_sessions(self) -> list[dict]:
        """Список недавних сессий: [{name, path}] от свежих к старым."""
        return list(self._recent)

    def set_root(self, path: Path) -> None:
        """Переопределяет корень хранения сессий и сохраняет настройку."""
        self._root = Path(path).resolve()
        self._save()

    def session_path(self, name: str) -> Path:
        """Путь к сессии по имени внутри текущего корня."""
        cleaned = sanitize_name(name)
        if cleaned is None:
            raise ValueError("Имя сессии пусто после очистки")
        return self._root / cleaned

    def add(self, session_dir: Path, name: str) -> None:
        """Добавляет сессию в начало списка недавних (без дубликатов)."""
        record = {"name": name, "path": str(Path(session_dir).resolve())}
        self._recent = [rec for rec in self._recent if rec["path"] != record["path"]]
        self._recent.insert(0, record)
        del self._recent[MAX_RECENT:]
        self._save()

    def remove(self, path: Path) -> None:
        """Удаляет сессию из списка недавних."""
        resolved = str(Path(path).resolve())
        self._recent = [rec for rec in self._recent if rec["path"] != resolved]
        self._save()

    def _load(self) -> None:
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        root = data.get("session_root")
        if isinstance(root, str) and root:
            self._root = Path(root).resolve()
        recent = data.get("recent_sessions")
        if isinstance(recent, list):
            self._recent = [
                {"name": str(rec.get("name", "")), "path": str(rec.get("path", ""))}
                for rec in recent
                if isinstance(rec, dict) and rec.get("path")
            ][:MAX_RECENT]

    def _save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_root": str(self._root),
            "recent_sessions": self._recent,
        }
        self.config_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
