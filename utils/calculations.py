"""
Модуль для вычисления площадей и статистик.
"""

import cv2
import numpy as np
from typing import Dict, Optional


class AreaCalculator:
    """
    Класс для вычисления площадей чашки Петри и колоний.
    
    Для перевода в реальные единицы измерения необходимо знать
    реальный диаметр чашки Петри (обычно 90 мм или 55 мм).
    """
    
    # Стандартные размеры чашек Петри (мм)
    STANDARD_PETRI_DIAMETERS = {
        'small': 55,
        'medium': 90,
        'large': 150
    }
    
    def __init__(self, petri_diameter_mm: float = 90.0):
        """
        Инициализация калькулятора.
        
        Args:
            petri_diameter_mm: Реальный диаметр чашки Петри в мм
        """
        self.petri_diameter_mm = petri_diameter_mm
    
    def calculate_areas(
        self,
        petri_mask: np.ndarray,
        colony_mask: np.ndarray,
        petri_info: Optional[Dict] = None
    ) -> Dict:
        """
        Вычисление площадей и статистик.
        
        Args:
            petri_mask: Маска чашки Петри
            colony_mask: Маска колоний
            petri_info: Информация о чашке Петри (опционально)
            
        Returns:
            Словарь с результатами анализа
        """
        # Площадь чашки Петри в пикселях
        petri_area_px = np.count_nonzero(petri_mask)
        
        # Площадь колоний в пикселях
        colony_area_px = np.count_nonzero(colony_mask)
        
        # Вычисляем коэффициент перевода пикселей в мм²
        if petri_info and 'radius' in petri_info:
            radius_px = petri_info['radius']
            petri_area_px_from_radius = np.pi * radius_px ** 2
            
            # Реальная площадь чашки Петри в мм²
            real_radius_mm = self.petri_diameter_mm / 2
            petri_area_mm2 = np.pi * real_radius_mm ** 2
            
            # Коэффициент: мм²/пиксель
            px_to_mm2 = petri_area_mm2 / petri_area_px_from_radius
        else:
            # Если нет информации о радиусе, используем приближение
            real_radius_mm = self.petri_diameter_mm / 2
            petri_area_mm2 = np.pi * real_radius_mm ** 2
            px_to_mm2 = petri_area_mm2 / petri_area_px if petri_area_px > 0 else 0
        
        # Переводим площадь колоний в мм²
        colony_area_mm2 = colony_area_px * px_to_mm2
        
        # Процент покрытия (площадь колоний относительно площади чашки)
        coverage_percent = (colony_area_px / petri_area_px * 100) if petri_area_px > 0 else 0
        
        # Соотношение площадей (колонии / чашка)
        area_ratio = colony_area_px / petri_area_px if petri_area_px > 0 else 0
        
        # Количество колоний
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            colony_mask, connectivity=8
        )
        colony_count = num_labels - 1  # Вычитаем фон
        
        # Статистика по отдельным колониям
        colony_stats = []
        for i in range(1, num_labels):
            area_px = stats[i, cv2.CC_STAT_AREA]
            area_mm2 = area_px * px_to_mm2
            cx, cy = centroids[i]
            
            colony_stats.append({
                'id': i,
                'area_px': area_px,
                'area_mm2': area_mm2,
                'centroid': (cx, cy),
                'bbox': {
                    'x': stats[i, cv2.CC_STAT_LEFT],
                    'y': stats[i, cv2.CC_STAT_TOP],
                    'width': stats[i, cv2.CC_STAT_WIDTH],
                    'height': stats[i, cv2.CC_STAT_HEIGHT]
                }
            })
        
        # Сортируем по размеру
        colony_stats.sort(key=lambda x: x['area_px'], reverse=True)
        
        # Средняя площадь колонии
        avg_colony_area_mm2 = colony_area_mm2 / colony_count if colony_count > 0 else 0
        
        return {
            'petri_area_px': petri_area_px,
            'petri_area_mm2': petri_area_mm2,
            'colony_area_px': colony_area_px,
            'colony_area_mm2': colony_area_mm2,
            'coverage_percent': coverage_percent,
            'area_ratio': area_ratio,
            'colony_count': colony_count,
            'avg_colony_area_mm2': avg_colony_area_mm2,
            'px_to_mm2': px_to_mm2,
            'colonies': colony_stats
        }
    
    def calculate_colony_statistics(
        self,
        colony_mask: np.ndarray,
        px_to_mm2: float
    ) -> Dict:
        """
        Вычисление статистик по колониям.
        
        Args:
            colony_mask: Маска колоний
            px_to_mm2: Коэффициент перевода пикселей в мм²
            
        Returns:
            Словарь со статистиками
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            colony_mask, connectivity=8
        )
        
        if num_labels <= 1:
            return {
                'count': 0,
                'mean_area_mm2': 0,
                'std_area_mm2': 0,
                'min_area_mm2': 0,
                'max_area_mm2': 0
            }
        
        areas = [stats[i, cv2.CC_STAT_AREA] * px_to_mm2 for i in range(1, num_labels)]
        
        return {
            'count': num_labels - 1,
            'mean_area_mm2': np.mean(areas),
            'std_area_mm2': np.std(areas),
            'min_area_mm2': np.min(areas),
            'max_area_mm2': np.max(areas)
        }
    
    def get_colony_distribution(
        self,
        colony_mask: np.ndarray,
        petri_info: Dict,
        num_rings: int = 5
    ) -> Dict:
        """
        Анализ распределения колоний по радиусу от центра чашки.
        
        Args:
            colony_mask: Маска колоний
            petri_info: Информация о чашке Петри
            num_rings: Количество кольцевых зон
            
        Returns:
            Словарь с распределением
        """
        center = petri_info['center']
        radius = petri_info['radius']
        
        h, w = colony_mask.shape
        Y, X = np.ogrid[:h, :w]
        
        # Расстояние от центра для каждого пикселя
        distances = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        
        distribution = []
        ring_width = radius / num_rings
        
        for i in range(num_rings):
            inner_r = i * ring_width
            outer_r = (i + 1) * ring_width
            
            # Маска кольца
            ring_mask = (distances >= inner_r) & (distances < outer_r)
            
            # Площадь кольца
            ring_area = np.count_nonzero(ring_mask)
            
            # Колонии в кольце
            colonies_in_ring = np.count_nonzero(colony_mask & ring_mask)
            
            # Плотность
            density = colonies_in_ring / ring_area if ring_area > 0 else 0
            
            distribution.append({
                'ring': i + 1,
                'inner_radius': inner_r,
                'outer_radius': outer_r,
                'ring_area_px': ring_area,
                'colony_area_px': colonies_in_ring,
                'density': density
            })
        
        return {'rings': distribution}
