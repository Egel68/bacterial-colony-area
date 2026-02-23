"""UI модуль приложения."""

from .analysis_window import AnalysisWindow
from .main_window import MainWindow
from .styles import get_application_style

__all__ = ["MainWindow", "AnalysisWindow", "get_application_style"]
