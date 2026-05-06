# Copyright 2024 Meng WANG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Data analysis module for extracting statistical insights from data."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple


class DataAnalyzer:
    """Analyzes data to extract statistical properties for LLM prompting.

    Provides insights such as monotonicity, nonlinearity, correlation, and
    residual patterns to help LLM generate better equation candidates.
    """

    @staticmethod
    def analyze(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze data and return statistical insights.

        Args:
            X: Input features of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
            feature_names: Optional list of feature names.

        Returns:
            Dictionary containing statistical insights.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        insights = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_names': feature_names or [f'x{i}' for i in range(X.shape[1])],
            'target_name': 'y',
            'correlations': {},
            'monotonicity': {},
            'nonlinearity': {},
        }

        # Compute correlations between each feature and target
        for i, name in enumerate(insights['feature_names']):
            # Pearson correlation
            corr = np.corrcoef(X[:, i], y)[0, 1]
            insights['correlations'][name] = round(corr, 4) if not np.isnan(corr) else 0.0

            # Check monotonicity using Spearman correlation
            spearman_corr = np.corrcoef(np.argsort(X[:, i]), np.argsort(y))[0, 1]
            insights['monotonicity'][name] = {
                'spearman': round(spearman_corr, 4) if not np.isnan(spearman_corr) else 0.0,
                'is_monotonic': abs(spearman_corr) > 0.7 if not np.isnan(spearman_corr) else False
            }

            # Check linearity using residual analysis of linear fit
            try:
                coeff = np.polyfit(X[:, i], y, 1)
                linear_pred = np.polyval(coeff, X[:, i])
                residuals = y - linear_pred
                residual_std = np.std(residuals)
                data_std = np.std(y)
                nonlinear_ratio = residual_std / data_std if data_std > 0 else 0
                insights['nonlinearity'][name] = {
                    'linear_residual_ratio': round(nonlinear_ratio, 4),
                    'is_highly_nonlinear': nonlinear_ratio > 0.2
                }
            except (ValueError, np.linalg.LinAlgError):
                insights['nonlinearity'][name] = {
                    'linear_residual_ratio': 1.0,
                    'is_highly_nonlinear': True
                }

        # Overall dataset statistics
        insights['statistics'] = {
            'y_range': (float(np.min(y)), float(np.max(y))),
            'y_mean': float(np.mean(y)),
            'y_std': float(np.std(y)),
            'has_negative': np.any(y < 0),
            'has_positive': np.any(y > 0),
            'has_zero_crossing': np.any(y[:-1] * y[1:] <= 0) if len(y) > 1 else False,
        }

        return insights

    @staticmethod
    def format_insights_for_prompt(insights: Dict[str, Any]) -> str:
        """Format insights as a human-readable string for LLM prompt.

        Args:
            insights: Dictionary from analyze() method.

        Returns:
            Formatted string with insights.
        """
        lines = []
        lines.append(f"Dataset has {insights['n_samples']} samples and {insights['n_features']} features.")

        lines.append("\nFeature-target relationships:")
        for name in insights['feature_names']:
            corr = insights['correlations'][name]
            mono = insights['monotonicity'][name]
            nonlin = insights['nonlinearity'][name]

            lines.append(f"  - {name}: correlation={corr}, monotonic={mono['is_monotonic']}, "
                        f"highly_nonlinear={nonlin['is_highly_nonlinear']}")

        stats = insights['statistics']
        lines.append("\nTarget variable statistics:")
        lines.append(f"  - Range: [{stats['y_range'][0]:.4f}, {stats['y_range'][1]:.4f}]")
        lines.append(f"  - Mean: {stats['y_mean']:.4f}, Std: {stats['y_std']:.4f}")
        lines.append(f"  - Contains negative values: {stats['has_negative']}")
        lines.append(f"  - Crosses zero: {stats['has_zero_crossing']}")

        return '\n'.join(lines)