# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — tests/test_fish_data_base                       ║
# ║  « result frames + file-folder sorting »                         ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import numpy as np
import pytest

from fish_data_base import result_frames
from fish_data_base.counterCurrentAna import SortMultiFileFolder, sortMultiFileFolder


# ── result_frames ───────────────────────────────────────────────────
def test_prep_df_3d_interleaves_labels():
    _, labels = result_frames.prep_df_3d("x", "y", 2)
    assert labels == ["x_0", "y_0", "x_1", "y_1"]


def test_make_df_2d_with_time_index():
    df = result_frames.make_df_2d([[1, 2], [3, 4]], "x", "y", fps=2.0, index="Time")
    assert list(df.columns) == ["x", "y"]
    assert df.index.name == "time sec"
    np.testing.assert_allclose(df.index.to_numpy(), [0.0, 0.5])


def test_make_df_3d_flattens_detections():
    data = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    df = result_frames.make_df_3d(data, "x", "y")
    assert list(df.columns) == ["x_0", "y_0", "x_1", "y_1"]
    np.testing.assert_array_equal(df.iloc[0].to_numpy(), [1, 2, 3, 4])


def test_make_df_hist_prepends_ortho_index():
    data = np.ones((2, 3))
    df = result_frames.make_df_hist(data)
    assert df.columns[0] == "orthoIndexMM"
    assert df.shape == (2, 4)


# ── SortMultiFileFolder ─────────────────────────────────────────────
def test_classify_file_parses_homozygous_female():
    sorter = SortMultiFileFolder("/tmp", "CCur")
    assert sorter.classify_file("ABhmf001", ".mat") == ("HM", 1, "F", "mat")


def test_classify_file_handles_internal_wildtype():
    sorter = SortMultiFileFolder("/tmp", "CCur")
    genotype, number, sex, ftype = sorter.classify_file("intwf012", ".seq")
    assert (genotype, number, sex, ftype) == ("INT", 12, "F", "seq")


def test_full_experiment_name_and_unknown():
    assert SortMultiFileFolder("/tmp", "cst").get_full_experiment_name() == "c-start"
    with pytest.raises(ValueError, match="unknown experiment"):
        SortMultiFileFolder("/tmp", "ZZZ").get_full_experiment_name()


def test_run_groups_files(tmp_path):
    (tmp_path / "hmf001_results_ana.mat").write_text("x", encoding="utf-8")
    (tmp_path / "hmf001.seq").write_text("x", encoding="utf-8")
    result = SortMultiFileFolder(tmp_path, "CCur").run()
    assert "HMF1" in result
    entry = result["HMF1"]
    assert entry["anaMat"].endswith("hmf001_results_ana.mat")
    assert entry["seq"].endswith("hmf001.seq")


def test_deprecated_class_and_main_alias(tmp_path):
    with pytest.warns(DeprecationWarning, match="SortMultiFileFolder"):
        sorter = sortMultiFileFolder(tmp_path, "CCur")
    assert isinstance(sorter, SortMultiFileFolder)
    with pytest.warns(DeprecationWarning, match="run"):
        assert sorter.__main__() == {}
