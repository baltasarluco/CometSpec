from __future__ import annotations

"""Helper utilities for :mod:`pyfluor`.

- load/open tables 
- locate and load spectra
- load ephemeris summary
- load pumping spectra
- get packaged example CN line list path
- get line lists
- etc...
"""

import os
import re
import io

from pathlib import Path
from typing import Dict, Any, Optional, Literal, Union

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import units as u
from specutils.utils.wcs_utils import air_to_vac


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"



def open_table(
    file_path: os.PathLike | str,
    *,
    header_row: int = 0,
    units_row: Optional[int] = 1,
    data_start: int = 2,
    fmt: str = "ascii.csv",
) -> Table:
    """Read a CSV/ASCII table with optional units row."""
    file_path = Path(file_path)

    t = Table.read(
        file_path,
        format=fmt,
        header_start=header_row,
        data_start=data_start,
    )

    # --- If units_row is None, skip unit parsing ---
    if units_row is None:
        for col in t.colnames:
            t[col].unit = None
        return t

    # Read the units row (0-based indexing)
    with file_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=0):
            if i == units_row:
                units_line = line.strip().split(",")
                break
        else:
            raise ValueError(f"units_row={units_row} exceeds number of rows")

    for col, unit_str in zip(t.colnames, units_line):
        unit_str = unit_str.strip()
        if unit_str and unit_str not in {"-", "None"}:
            try:
                t[col].unit = u.Unit(unit_str)
            except Exception:
                t[col].unit = None
        else:
            t[col].unit = None

    return t



def find_spectrum_file(
    dir_path: os.PathLike | str,
    *,
    night: str,
    fibre: str,
    suffix: str = ".csv",
) -> Optional[Path]:
    """Return first file under dir_path that matches night & fibre."""
    dir_path = Path(dir_path)
    fibre = str(fibre)
    for fname in sorted(dir_path.iterdir()):
        if not fname.name.endswith(suffix):
            continue
        if night in fname.name and fibre in fname.name:
            return fname
    return None


def load_spectrum(
    dir_path: os.PathLike | str,
    *,
    night: str,
    fibre: str,
    header_row: int = 0,
    units_row: int = 1,
    data_start: int = 2,
) -> Table:
    """Load stacked spectrum for a given (night, fibre)."""
    path = find_spectrum_file(dir_path, night=night, fibre=fibre)
    if path is None:
        raise FileNotFoundError(
            f"No spectrum file found in {dir_path!s} for night={night!r}, fibre={fibre!r}"
        )
    return open_table(path, header_row=header_row, units_row=units_row, data_start=data_start)


def load_ephemeris_summary(
    path: os.PathLike | str,
    *,
    key_column: str = "date_obs",
) -> Dict[str, Dict[str, Any]]:
    """Read ephemeris_means_by_observation.csv into dict-of-dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ephemeris summary file not found at {path!s}.")

    table = Table.read(path, format="ascii.csv")
    epi: Dict[str, Dict[str, Any]] = {}

    for idx, row in enumerate(table):
        if key_column in table.colnames:
            key = str(row[key_column])
        else:
            key = str(idx)

        record: Dict[str, Any] = {}
        for col in table.colnames:
            val = row[col]
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:
                    pass
            record[col] = val
        epi[key] = record

    return epi


def get_ephemeris_for_night(
    ephemeris: Dict[str, Dict[str, Any]],
    night: str,
) -> Dict[str, Any]:
    """Return ephemeris[night] or empty dict."""
    return dict(ephemeris.get(str(night), {}))


def load_pumping_file(
    night: str,
    *,
    directory: os.PathLike | str | None = None,
    pattern: str = "pumping_{night}.txt",
    wavelength_col: str = "WAVE",   # NEW
    flux_col: str = "FLUX",         # NEW
    scale_by_r_au: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load pumping spectrum for a given night.

    Expects at least columns:
    - wavelength_col : Angstrom
    - flux_col       : e.g. erg s^-1 cm^-2 Å^-1 at comet (or to be scaled)
    """
    if directory is None:
        directory = DATA_DIR
    directory = Path(directory)

    fname = pattern.format(night=night)
    path = directory / fname
    if not path.exists():
        raise FileNotFoundError(f"Pumping file not found: {path!s}")

    df = pd.read_csv(path, sep=None, engine="python")

    if wavelength_col not in df.columns or flux_col not in df.columns:
        raise ValueError(
            f"Pumping file must contain columns "
            f"'{wavelength_col}' and '{flux_col}'."
        )

    if scale_by_r_au is not None:
        df[flux_col] = df[flux_col] * (1.0 / float(scale_by_r_au)) ** 2

    return df

# GET KURUCZ FROM DATA FOLDER:
def get_kurucz_irradiance_path() -> Path:
    """
    Return the path to the packaged high res Kurucz solar irradiance file.

    By default this expects `kurucz_solar_irradiance.txt` to live in `fluo_cn/data/`.
    """
    candidate = DATA_DIR / "kurucz_irradiance.txt"
    if not candidate.exists():
        raise FileNotFoundError(
            "Kurucz solar irradiance file not found in fluo_cn/data. "
            "Place the file there or provide your own path when loading pumping spectra."
        )
    return candidate

def open_kurucz_irradiance() -> pd.DataFrame:
    """
    Load the high res Kurucz solar irradiance file as a DataFrame. in AA and cgs at 1au

    Expects at least columns:
    - WAVE : Angstrom
    - FLUX : e.g. erg s^-1 cm^-2 Å^-1 at 1 AU
    """
    path = get_kurucz_irradiance_path()
    df = pd.read_csv(path, sep='\s+', names=['nm', 'flux'])
    wave = np.asarray(df['nm']*10, dtype=float)
    flux = np.asarray(df['flux'], dtype=float)
    flux = flux * u.W / (u.m**2 * u.nm)
    flux = flux.to(u.erg / (u.s * u.cm**2 * u.AA))
    flux = flux.value

    solar = pd.DataFrame({'WAVE': wave, 'FLUX': flux})

    return solar

Mol = Literal["CN", "C2", "C3", "NH", "CH"]
CNIso = Literal["12C14N", "13C14N", "12C15N"]


def get_default_mol_linelist_path(
    mol: Mol = "CN",
    *,
    isotope: CNIso = "12C14N",
) -> Path:
    """
    Return the path to the packaged default molecular line list.

    Layout expected:
      - data/CN/<isotope>.txt for CN (e.g., data/CN/12C14N.txt)
      - data/neowise_lines.txt for other molecules (C2, C3, NH, CH)
    """
    mol = mol.upper()

    if mol == "CN":
        candidate = DATA_DIR / "CN" / f"{isotope}.txt"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Default CN line list not found: {candidate!s}. "
                "Place the file in data/CN/ or pass an explicit path."
            )
        return candidate

    if mol in {"C2", "C3", "NH", "CH"}:
        candidate = DATA_DIR / "neowise_lines.txt"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Default NEOWISE molecular line list not found: {candidate!s}. "
                "Place neowise_lines.txt in data/ or pass an explicit path."
            )
        return candidate

    raise ValueError(f"Unknown mol={mol!r}. Expected one of: CN, C2, C3, NH, CH.")


# --- Atomic line lists ---

# Minimal name->symbol map from Hydrogen through Yttrium (inclusive).
# Accepts either full name ("hydrogen") or symbol ("H").
_NAME_TO_SYMBOL = {
    "hydrogen": "H", "helium": "He",
    "lithium": "Li", "beryllium": "Be", "boron": "B", "carbon": "C", "nitrogen": "N", "oxygen": "O",
    "fluorine": "F", "neon": "Ne",
    "sodium": "Na", "magnesium": "Mg", "aluminium": "Al", "aluminum": "Al", "silicon": "Si",
    "phosphorus": "P", "sulfur": "S", "sulphur": "S", "chlorine": "Cl", "argon": "Ar",
    "potassium": "K", "calcium": "Ca", "scandium": "Sc", "titanium": "Ti", "vanadium": "V",
    "chromium": "Cr", "manganese": "Mn", "iron": "Fe", "cobalt": "Co", "nickel": "Ni",
    "copper": "Cu", "zinc": "Zn", "gallium": "Ga", "germanium": "Ge", "arsenic": "As",
    "selenium": "Se", "bromine": "Br", "krypton": "Kr",
    "rubidium": "Rb", "strontium": "Sr", "yttrium": "Y", "itrium": "Y",
}

_SYMBOL_RE = re.compile(r"^[A-Z][a-z]?$")


def _normalize_element(element: str) -> str:
    s = element.strip()
    if _SYMBOL_RE.match(s):
        return s  # already a symbol like "H", "Fe", "Y"
    key = s.lower()
    if key in _NAME_TO_SYMBOL:
        return _NAME_TO_SYMBOL[key]
    raise ValueError(
        f"Unknown element={element!r}. Provide a symbol (e.g., 'Fe') or a name from hydrogen..yttrium."
    )


def get_default_atomic_linelist_path(element: str) -> Path:
    """
    Return the path to the packaged default atomic line list.

    Layout expected:
      - data/Element_lines/<SYMBOL>.txt (e.g., data/Element_lines/Fe.txt)
    """
    symbol = _normalize_element(element)
    candidate = DATA_DIR / "Element_lines" / f"{symbol}.txt"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Default atomic line list not found: {candidate!s}. "
            "Place the file in data/Element_lines/ or pass an explicit path."
        )
    return candidate


def print_available_linelist_inventory() -> None:
    """
    Print available line lists found in the data directory.

    Distinction:
      - Default available for fluorescence: CN isotopologues in data/CN/
      - Available for line plotting: all molecular and atomic line lists

    Expected layout:
      - data/CN/*.txt
      - data/neowise_lines.txt
      - data/Element_lines/*.txt
    """
    data_dir = Path(DATA_DIR)

    # --- Fluorescence defaults (currently CN only) ---
    cn_dir = data_dir / "CN"
    cn_files = sorted(cn_dir.glob("*.txt")) if cn_dir.exists() else []

    print("Default available for fluorescence:")
    if cn_files:
        for p in cn_files:
            print(f"  - CN {p.stem}")
    else:
        print("  (none found)")

    print('Comment: for the fluorescence modeling you could also provide your own line list')
    # --- Line plotting (all) ---
    print("\nAvailable for line plotting:")

    # Molecular
    if cn_files:
        for p in cn_files:
            print(f"  - [mol] CN {p.stem}")

    neowise = data_dir / "neowise_lines.txt"
    if neowise.exists():
        print("  - [mol] NEOWISE (C2, C3, NH, CH)")

    # Atomic
    elem_dir = data_dir / "Element_lines"
    elem_files = sorted(elem_dir.glob("*.txt")) if elem_dir.exists() else []
    if elem_files:
        for p in elem_files:
            print(f"  - [atom] {p.stem}")

    if not (cn_files or neowise.exists() or elem_files):
        print("  (none found)")

def load_cn_linelist(path_or_text: Union[str, os.PathLike]) -> pd.DataFrame:
    """Load a Brooke/PGOPHER-style CN line list into a DataFrame.

    Parameters
    ----------
    path_or_text : str or PathLike or full file text
        - If it's a path (str or PathLike), the file is read from disk.
        - If it's a string containing newlines or starting with 'Title:',
          it's interpreted as the content itself.
    """
    colspecs = [
        (0, 1),  (2, 3),  (4, 6),  (7, 9),
        (10, 15), (16, 21),
        (22, 23), (24, 25),
        (26, 27), (28, 29),
        (30, 33), (34, 37),
        (38, 49), (50, 60), (61, 69),
        (70, 80), (81, 93), (94, 106), (107, 118),
    ]
    names = [
        "eS'", "eS''", "v'", "v''", "J'", "J''",
        "F'", "F''", "p'", "p''",
        "N'", "N''",
        "Obs", "Cal", "Res",
        "E''", "A", "f", "Des",
    ]

    # Normalize to something we can reason about
    if isinstance(path_or_text, (os.PathLike,)):
        # definitely a path
        path_str = os.fspath(path_or_text)
        is_text = False
    elif isinstance(path_or_text, str):
        # could be either: text payload or a filesystem path
        # heuristic: if it clearly looks like multiline content or Brooke header,
        # treat as text; otherwise assume it's a path.
        if "\n" in path_or_text or path_or_text.lstrip().startswith("Title:"):
            is_text = True
            text = path_or_text
        else:
            path_str = path_or_text
            is_text = False
    else:
        # Fallback: try to interpret as path-like
        path_str = os.fspath(path_or_text)
        is_text = False

    if not is_text:
        with open(path_str, "r", encoding="utf-8") as f:
            text = f.read()

    data_lines = [
        ln for ln in text.splitlines()
        if len(ln) > 2 and ln[0] in "ABX" and ln[1].isspace() and ln[2] in "ABX"
    ]
    if not data_lines:
        raise ValueError("No Brooke-style data lines found in input.")

    df = pd.read_fwf(io.StringIO("\n".join(data_lines)),
                     colspecs=colspecs, names=names)

    num_cols = ["v'", "v''", "J'", "J''", "F'", "F''",
                "N'", "N''", "Obs", "Cal", "Res", "E''", "A", "f"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["eS'", "eS''", "p'", "p''", "Des"]:
        df[c] = df[c].astype(str).str.strip()

    wn_cal = df["Cal"].to_numpy(float)
    wn_obs = df["Obs"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_cal_A = 1e8 / wn_cal
        lam_obs_A = 1e8 / wn_obs

    df["lambda_vac_A_from_Cal"] = lam_cal_A
    df["lambda_vac_A_from_Obs"] = lam_obs_A

    return df

_NEOWISE_MOL_MAP = {
    # user-facing -> file token
    "CN": "CN",
    "CH": "CH",
    "NH2": "NH2",
    "C3": "C_3_",
    "C2": "C_2_",
}


def _airA_to_vacA(wave_air_A: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Convert wavelength(s) from air Å to vacuum Å."""
    return air_to_vac(np.asarray(wave_air_A) * u.AA).to(u.AA).value


def load_neowise_linelist(path: Union[str, os.PathLike]) -> pd.DataFrame:
    """
    Load NEOWISE molecular line list into a DataFrame with columns:
      - Wave (air Å)
      - REL_Intensity
      - MOL
      - identifier
    """
    df = {"Wave": [], "REL_Intensity": [], "MOL": [], "identifier": []}
    path = os.fspath(path)

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            parts = [x for x in raw.strip().split(" ") if x != ""]
            if len(parts) < 3:
                continue
            wave = float(parts[0])
            rel = float(parts[1])
            mol = parts[2]
            ident = " ".join(parts[3:]) if len(parts) > 3 else ""
            df["Wave"].append(wave)
            df["REL_Intensity"].append(rel)
            df["MOL"].append(mol)
            df["identifier"].append(ident)

    return pd.DataFrame(df)


def get_linelist_wavelengths_vacuum(
    species: str,
    *,
    # CN options
    isotope: CNIso = "12C14N",
    upper_state: Optional[str] = "B",
    lower_state: Optional[str] = "X",
    v_upper: Optional[int] = 0,
    v_lower: Optional[int] = 0,
    # selection options
    n_lines: Optional[int] = None,
    path: Optional[Union[str, os.PathLike]] = None,
) -> list[float]:

    sp = species.strip()
    mol_up = sp.upper()

    # -----------------
    # CN (Brooke-style)
    # -----------------
    if mol_up == "CN":
        if path is None:
            path = get_default_mol_linelist_path("CN", isotope=isotope)

        df = load_cn_linelist(path)

        # --- configurable filters ---
        if upper_state is not None:
            df = df[df["eS'"] == upper_state]
        if lower_state is not None:
            df = df[df["eS''"] == lower_state]
        if v_upper is not None:
            df = df[df["v'"] == v_upper]
        if v_lower is not None:
            df = df[df["v''"] == v_lower]

        if df.empty:
            raise ValueError(
                "CN selection produced no lines. "
                "Check electronic states or vibrational numbers."
            )

        # Sort by Einstein A
        df = df.sort_values(by="A", ascending=False)

        if n_lines is not None:
            df = df.head(int(n_lines))

        waves_vac = df["lambda_vac_A_from_Cal"].astype(float).to_list()

        print(
            "CN lines: sorted by Einstein A coefficient (descending). "
            f"Filters: eS'={upper_state}, eS''={lower_state}, "
            f"v'={v_upper}, v''={v_lower}"
        )

        return waves_vac

    # -----------------
    # NEOWISE multi-molecule file -> MUST filter to requested molecule
    # -----------------
    if mol_up in {"C2", "C3", "NH", "CH", "NH2"}:
        if path is None:
            # any of these returns neowise_lines.txt with your helper
            path = get_default_mol_linelist_path("C2")

        neowise = load_neowise_linelist(path)

        token = _NEOWISE_MOL_MAP.get(mol_up)
        if token is None:
            raise ValueError(f"Unsupported molecule {mol_up!r}. Try CN, CH, NH2, C2, C3.")

        sub = neowise[neowise["MOL"] == token].copy()  # <-- key: only requested species
        if sub.empty:
            raise ValueError(f"No lines found for {mol_up} (token={token!r}) in {path!s}.")

        sub = sub.sort_values(by="REL_Intensity", ascending=False)
        if n_lines is not None:
            sub = sub.head(int(n_lines))

        waves_vac = _airA_to_vacA(sub["Wave"].astype(float).to_numpy()).tolist()
        print("NEOWISE lines: sorted by REL_Intensity in the NEOWISE comet (descending).")

        return waves_vac

    # -----------------
    # Atomic element (your atomic files are per-element; so already “only that”)
    # -----------------
    if path is None:
        path = get_default_atomic_linelist_path(sp)

    atom = pd.read_csv(os.fspath(path), sep='\s+', names=["WAVE", "name", "ion"])
    waves_vac = _airA_to_vacA(atom["WAVE"].astype(float).to_numpy()).tolist()

    print("Atomic lines: lines from NIST (not completed lists, just for quick plotting).")
    return waves_vac