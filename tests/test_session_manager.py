from pathlib import Path

import pytest

from labeling.session_manager import (
    DEFAULT_SESSION_ROOT,
    SESSION_SUBDIRS,
    SessionManager,
    ensure_session_structure,
    sanitize_name,
)


@pytest.fixture
def config_path(tmp_path):
    return tmp_path / "cfg" / "bacteria_analyzer.json"


@pytest.fixture
def manager(config_path):
    return SessionManager(config_path)


class TestSanitizeName:
    def test_keeps_valid_characters(self):
        assert sanitize_name("Чашка #1 — эксперимент") == "Чашка #1 — эксперимент"

    def test_removes_invalid_windows_characters(self):
        assert sanitize_name("фото:эксперимент/31") == "фотоэксперимент31"

    def test_empty_returns_none(self):
        assert sanitize_name("") is None

    def test_none_returns_none(self):
        assert sanitize_name(None) is None

    def test_only_invalid_returns_none(self):
        assert sanitize_name('<>:"/\\|?*') is None

    def test_truncates_to_max_length(self):
        assert len(sanitize_name("a" * 100)) == 60


class TestEnsureSessionStructure:
    def test_creates_root_and_subdirs(self, tmp_path):
        session_dir = tmp_path / "session"
        ensure_session_structure(session_dir)
        assert session_dir.is_dir()
        for sub in SESSION_SUBDIRS:
            assert (session_dir / sub).is_dir()

    def test_idempotent(self, tmp_path):
        session_dir = tmp_path / "session"
        ensure_session_structure(session_dir)
        ensure_session_structure(session_dir)
        assert session_dir.is_dir()


class TestSessionManager:
    def test_default_root_when_config_missing(self, config_path):
        assert not config_path.exists()
        manager = SessionManager(config_path)
        assert manager.current_root == DEFAULT_SESSION_ROOT
        assert manager.recent_sessions == []

    def test_default_root_used_for_missing_file(self):
        manager = SessionManager(Path("/nonexistent/path/cfg.json"))
        assert manager.current_root == DEFAULT_SESSION_ROOT

    def test_corrupt_json_ignored(self, config_path):
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{ not json", encoding="utf-8")
        manager = SessionManager(config_path)
        assert manager.current_root == DEFAULT_SESSION_ROOT
        assert manager.recent_sessions == []

    def test_set_root_persists_across_instances(self, manager, tmp_path):
        new_root = tmp_path / "custom_root"
        manager.set_root(new_root)
        reloaded = SessionManager(manager.config_path)
        assert reloaded.current_root == new_root

    def test_session_path_uses_sanitized_name(self, manager):
        assert manager.session_path("фото:31") == manager.current_root / "фото31"

    def test_session_path_raises_on_empty_name(self, manager):
        with pytest.raises(ValueError):
            manager.session_path("<>:")

    def test_add_and_list_order(self, manager, tmp_path):
        s1, s2 = tmp_path / "s1", tmp_path / "s2"
        ensure_session_structure(s1)
        ensure_session_structure(s2)
        manager.add(s1, "Один")
        manager.add(s2, "Два")
        assert [r["name"] for r in manager.recent_sessions] == ["Два", "Один"]

    def test_add_dedup_moves_to_front(self, manager, tmp_path):
        s1, s2 = tmp_path / "s1", tmp_path / "s2"
        manager.add(s1, "Один")
        manager.add(s2, "Два")
        manager.add(s1, "Один")
        assert [r["name"] for r in manager.recent_sessions] == ["Один", "Два"]

    def test_caps_at_max_recents(self, manager, tmp_path):
        for i in range(15):
            session_dir = tmp_path / f"s{i}"
            session_dir.mkdir()
            manager.add(session_dir, f"Сессия {i}")
        recents = manager.recent_sessions
        assert len(recents) == 10
        assert recents[0]["name"] == "Сессия 14"

    def test_remove(self, manager, tmp_path):
        session_dir = tmp_path / "s1"
        session_dir.mkdir()
        manager.add(session_dir, "Один")
        manager.remove(session_dir)
        assert manager.recent_sessions == []

    def test_recents_persist_across_instances(self, manager, tmp_path):
        session_dir = tmp_path / "s1"
        session_dir.mkdir()
        manager.add(session_dir, "Один")
        reloaded = SessionManager(manager.config_path)
        assert [r["name"] for r in reloaded.recent_sessions] == ["Один"]
