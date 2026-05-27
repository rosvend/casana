"""One-off ETL: normalize per-city neighborhood/upper-division geospatial data
into a single lightweight Source of Truth (data/normalized_zones.json).

Run manually after data updates:

    uv run --group dev python scripts/normalize_geodata.py

The output JSON is the runtime contract — ``src/utils/geography.py`` reads it
without needing geopandas. Re-run only when source files in ``data/<city>/``
change.

Field mappings discovered during exploration:

================================================================================
Source file                                Neighborhood field    Upper-division
================================================================================
bogota/bogota_barrios.geojson              SCANOMBRE             (sjoin)
bogota/bogota_localidades.json (Esri)      LocNombre             —
medellin/medellin_barrios.geojson          nombre                limitecomuna…id
medellin/medellin_comunas.geojson          —                     name + ref
barranquilla/barranquilla_barrios          nombre_barrio         localidad (id)
barranquilla/barranquilla_localidades      —                     nombre + ident.
cali/cali_barrios/mc_barrios.shp           barrio                comuna (str)
cali/cali_comunas/mc_comunas.shp           —                     nombre + comuna
cartagena/cartagena_barrios/Barrios_Ctg    NOMBRE                LOC + ZONA
================================================================================

Cali and Cartagena shapefiles use projected CRSes (MAGNA-Cali EPSG:6249 and a
local Cartagena PCS); geopandas reads the ``.prj`` automatically, so a
``to_crs(WGS84)`` call is enough. Centroids are computed in a metric CRS
(EPSG:3857) to avoid the latitude-distortion artefact that comes with
naive lat/lon centroids.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = DATA / "normalized_zones.json"

WGS84 = "EPSG:4326"
COLOMBIA_BBOX = (-82.0, -5.0, -66.0, 14.0)  # lon_min, lat_min, lon_max, lat_max


def _normalize(s: str | None) -> str:
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(s).strip().lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _centroid_lonlat(geom) -> tuple[float, float]:
    """Reproject to a metric CRS for accurate centroid, then back to WGS84."""
    gs = gpd.GeoSeries([geom], crs=WGS84).to_crs("EPSG:3857")
    c = gs.centroid.to_crs(WGS84).iloc[0]
    return (round(c.x, 6), round(c.y, 6))


def load_bogota() -> list[dict]:
    barrios = gpd.read_file(DATA / "bogota" / "bogota_barrios.geojson").to_crs(WGS84)
    localidades = gpd.read_file(
        DATA / "bogota" / "bogota_localidades.json", driver="ESRIJSON"
    )
    if localidades.crs is None:
        localidades = localidades.set_crs(WGS84)
    else:
        localidades = localidades.to_crs(WGS84)
    barrios["_rep"] = barrios.geometry.representative_point()
    rep = barrios.set_geometry("_rep")
    joined = gpd.sjoin(
        rep, localidades[["LocNombre", "geometry"]], how="left", predicate="within"
    )
    records = []
    for _, row in joined.iterrows():
        name = row.get("SCANOMBRE")
        loc = row.get("LocNombre")
        if not name:
            continue
        geom = barrios.loc[row.name, "geometry"]
        lon, lat = _centroid_lonlat(geom)
        records.append({
            "city": "bogota",
            "upper_division": _normalize(loc) if loc else None,
            "neighborhood": _normalize(name),
            "centroid": {"lat": lat, "lon": lon},
        })
    return records


def _medellin_comuna_label(raw_name: str) -> str:
    parts = str(raw_name).split(" - ", 1)
    return _normalize(parts[1] if len(parts) == 2 else raw_name)


def load_medellin() -> list[dict]:
    barrios = gpd.read_file(DATA / "medellin" / "medellin_barrios.geojson").to_crs(WGS84)
    comunas = gpd.read_file(DATA / "medellin" / "medellin_comunas.geojson").to_crs(WGS84)
    comuna_by_ref = {
        str(r["ref"]).lstrip("0"): _medellin_comuna_label(r["name"])
        for _, r in comunas.iterrows()
        if r.get("ref") is not None
    }
    records = []
    for _, row in barrios.iterrows():
        name = row.get("nombre")
        comuna_id = str(row.get("limitecomunacorregimientoid") or "").lstrip("0")
        if not name:
            continue
        lon, lat = _centroid_lonlat(row.geometry)
        records.append({
            "city": "medellin",
            "upper_division": comuna_by_ref.get(comuna_id),
            "neighborhood": _normalize(name),
            "centroid": {"lat": lat, "lon": lon},
        })
    return records


def load_barranquilla() -> list[dict]:
    barrios = gpd.read_file(
        DATA / "barranquilla" / "barranquilla_barrios.geojson"
    ).to_crs(WGS84)
    localidades = gpd.read_file(
        DATA / "barranquilla" / "barranquilla_localidades.geojson"
    ).to_crs(WGS84)
    loc_by_id = {
        str(r["identificador"]).lstrip("0"): _normalize(r["nombre"])
        for _, r in localidades.iterrows()
    }
    records = []
    for _, row in barrios.iterrows():
        name = row.get("nombre_barrio")
        loc_id = str(row.get("localidad") or "").lstrip("0")
        if not name:
            continue
        lon, lat = _centroid_lonlat(row.geometry)
        records.append({
            "city": "barranquilla",
            "upper_division": loc_by_id.get(loc_id),
            "neighborhood": _normalize(name),
            "centroid": {"lat": lat, "lon": lon},
        })
    return records


def load_cali() -> list[dict]:
    barrios = gpd.read_file(
        DATA / "cali" / "cali_barrios" / "mc_barrios.shp"
    ).to_crs(WGS84)
    records = []
    for _, row in barrios.iterrows():
        name = row.get("barrio")
        comuna_id = str(row.get("comuna") or "").strip()
        if not name:
            continue
        lon, lat = _centroid_lonlat(row.geometry)
        records.append({
            "city": "cali",
            "upper_division": f"comuna {comuna_id}" if comuna_id else None,
            "neighborhood": _normalize(name),
            "centroid": {"lat": lat, "lon": lon},
        })
    return records


def load_cartagena() -> list[dict]:
    barrios = gpd.read_file(
        DATA / "cartagena" / "cartagena_barrios" / "Barrios_Ctg.shp"
    ).to_crs(WGS84)
    records = []
    for _, row in barrios.iterrows():
        name = row.get("NOMBRE")
        loc = str(row.get("LOC") or "").strip()
        if not name:
            continue
        lon, lat = _centroid_lonlat(row.geometry)
        records.append({
            "city": "cartagena",
            "upper_division": f"localidad {loc}" if loc else None,
            "neighborhood": _normalize(name),
            "centroid": {"lat": lat, "lon": lon},
        })
    return records


def _validate(records: list[dict]) -> None:
    lon_min, lat_min, lon_max, lat_max = COLOMBIA_BBOX
    valid_cities = {"bogota", "medellin", "cali", "barranquilla", "cartagena"}
    for r in records:
        c = r["centroid"]
        assert lon_min <= c["lon"] <= lon_max, f"out-of-bbox lon: {r}"
        assert lat_min <= c["lat"] <= lat_max, f"out-of-bbox lat: {r}"
        assert r["neighborhood"], f"empty neighborhood: {r}"
        assert r["city"] in valid_cities, f"unknown city: {r}"


def main() -> None:
    records: list[dict] = []
    for loader in (
        load_bogota,
        load_medellin,
        load_barranquilla,
        load_cali,
        load_cartagena,
    ):
        chunk = loader()
        print(f"{loader.__name__}: {len(chunk)} records")
        records.extend(chunk)
    _validate(records)
    payload = {
        "version": 1,
        "cities": ["bogota", "medellin", "cali", "barranquilla", "cartagena"],
        "neighborhoods": records,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB, {len(records)} records)")


if __name__ == "__main__":
    main()
