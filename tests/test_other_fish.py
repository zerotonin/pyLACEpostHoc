# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — tests/test_other_fish                           ║
# ║  « cholesterol assay statistics »                                ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import numpy as np
import pytest

from other_fish_related_analysis import cholesterolAna


def test_get_ci_drops_nans_and_brackets_mean():
    lo, hi = cholesterolAna.get_ci(np.array([1.0, 2, 3, 4, 5, np.nan]))
    assert lo < 3.0 < hi


def test_get_ci_from_2d_one_per_column():
    data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    assert len(cholesterolAna.get_ci_from_2d(data, axis=0)) == 2


def test_fisher_test_returns_oddsratio_and_p():
    oddsratio, pvalue = cholesterolAna.fisher_test_for_fluorescence((3, 10), (5, 10))
    assert 0.0 <= pvalue <= 1.0
    assert oddsratio >= 0.0


def test_read_file_transposes(tmp_path):
    csv = tmp_path / "assay.csv"
    csv.write_text("groupA,1,2,3\ngroupB,4,5,6\n", encoding="utf-8")
    data, labels = cholesterolAna.read_file(csv)
    assert labels == ["groupA", "groupB"]
    assert data.shape == (3, 2)


def test_deprecated_alias_warns():
    with pytest.warns(DeprecationWarning, match="get_ci"):
        cholesterolAna.getCI(np.array([1.0, 2.0, 3.0]))
