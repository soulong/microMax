"""SQL identifier quoting tests.

Class names become column names (prob_<class>); they may contain spaces,
dots or quotes, so both CREATE and SELECT must quote every identifier.
"""

import sqlite3

import numpy as np

from microModel.reduction import _load_inference_features
from microModel.utils import sql_ident


def test_sql_ident_escapes_embedded_quotes():
    assert sql_ident("a") == '"a"'
    assert sql_ident('a"b') == '"a""b"'
    assert sql_ident("prob_nuclear punctae") == '"prob_nuclear punctae"'


def test_load_inference_features_quotes_special_columns(tmp_path):
    """The reader previously joined raw column names: a class name with a
    space (legal; microModel itself writes such a column quoted) made the
    SELECT invalid SQL."""
    db = tmp_path / "infer.db"
    feats = np.arange(6, dtype=np.float32).reshape(2, 3)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE inference ("
        "uid INTEGER PRIMARY KEY AUTOINCREMENT, "
        "directory TEXT NOT NULL, filename TEXT NOT NULL, "
        '"prob_nuclear punctae" REAL, '
        '"weird""quote" REAL, '
        "features BLOB)"
    )
    for i in range(2):
        conn.execute(
            'INSERT INTO inference '
            '(directory, filename, "prob_nuclear punctae", "weird""quote", features) '
            "VALUES (?, ?, ?, ?, ?)",
            (".", f"c{i}.tiff", 0.5, 0.1, feats[i].tobytes()),
        )
    conn.commit()
    conn.close()

    out_feats, dicts = _load_inference_features(str(db), raise_on_error=True)
    np.testing.assert_allclose(out_feats, feats)
    assert "prob_nuclear punctae" in dicts[0]
    assert 'weird"quote' in dicts[0]
