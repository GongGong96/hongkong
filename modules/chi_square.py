"""
chi_square.py
Uji keacakan dengan metode Chi-Square Goodness of Fit.
Hipotesis nol (H0): distribusi nomor adalah acak seragam (uniform).
"""

from modules import stats_utils as stats
from database import get_conn
from modules.preprocessing import hitung_frekuensi_digit
from datetime import date
import numpy as np


def uji_chi_square_digit(kategori: str):
    """
    Uji Chi-square pada distribusi digit per posisi.
    Jika distribusi seragam, tiap digit (0-9) harusnya muncul N/10 kali.

    Returns:
        dict berisi hasil uji per posisi + kesimpulan keseluruhan
    """
    freq_map = hitung_frekuensi_digit(kategori)
    hasil_per_posisi = {}

    for posisi, freq_dict in freq_map.items():
        observed = np.array([freq_dict[str(d)] for d in range(10)])
        n = observed.sum()
        if n == 0:
            continue
        expected = np.full(10, n / 10)  # distribusi seragam (Ei = N/10)

        chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        df = 9  # k-1 = 10-1

        kesimpulan_posisi = "Acak (H0 diterima)" if p_value >= 0.05 else "Tidak Acak (H0 ditolak)"

        hasil_per_posisi[posisi] = {
            "chi2": round(float(chi2), 4),
            "p_value": round(float(p_value), 6),
            "df": df,
            "n": int(n),
            "observed": observed.tolist(),
            "expected": [round(e, 2) for e in expected.tolist()],
            "kesimpulan": kesimpulan_posisi,
        }

    # Kesimpulan keseluruhan: mayoritas posisi acak?
    acak_count = sum(1 for v in hasil_per_posisi.values() if "diterima" in v["kesimpulan"])
    total = len(hasil_per_posisi)
    kesimpulan_global = "Acak (H0 diterima)" if acak_count >= total / 2 else "Tidak Acak (H0 ditolak)"

    return {
        "kategori": kategori,
        "per_posisi": hasil_per_posisi,
        "kesimpulan_global": kesimpulan_global,
        "acak_count": acak_count,
        "total_posisi": total,
    }


def simpan_hasil_uji(kategori: str, hasil: dict):
    """Simpan ringkasan hasil uji ke tabel uji_keacakan."""
    conn = get_conn()
    today = date.today().isoformat()

    # Ambil nilai dari posisi 'ribuan' sebagai representasi
    pos = hasil["per_posisi"].get("ribuan", {})
    chi2 = pos.get("chi2", 0)
    p_val = pos.get("p_value", 0)
    df = pos.get("df", 9)

    conn.execute(
        """INSERT INTO uji_keacakan (tanggal_uji, kategori, chi_square_value, p_value, df, kesimpulan)
           VALUES (?,?,?,?,?,?)""",
        (today, kategori, chi2, p_val, df, hasil["kesimpulan_global"]),
    )
    conn.commit()
    conn.close()


def get_riwayat_uji(kategori: str = None, limit: int = 20):
    """Ambil riwayat uji keacakan dari database."""
    conn = get_conn()
    if kategori:
        rows = conn.execute(
            "SELECT * FROM uji_keacakan WHERE kategori=? ORDER BY created_at DESC LIMIT ?",
            (kategori, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM uji_keacakan ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
