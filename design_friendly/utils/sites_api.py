"""
Extract and visualize all available layers for Hollandse Kust Noord
from the Rijkswaterstaat public ArcGIS FeatureServer.
"""

import matplotlib.pyplot as plt
import requests
import numpy as np

BASE_URL = (
    "https://geo.rijkswaterstaat.nl/arcgis/rest/services"
    "/GDR/windenergiegebieden/FeatureServer"
)

LAYERS = {
    "turbines": 0,
    "boundary": 1,
    # "safety_zones": 2,
    "passage": 4,
    # "shared_use": 5,  # fishing etc.
    "designated_area": 7,
}

CRS = {
    "wgs": {"epsg": 4326, "x": "lon", "y": "lat"},  # wgs84
    "utm": {"epsg": 25831, "x": "easting", "y": "northing"},  # utm31n
}


def get_layer(layer, farm_name="HKN", crs="wgs84"):
    """Query a FeatureServer layer by farm name.

    Parameters
    ----------
    layer : str
        Key in LAYERS.
    farm_name : str
        Substring to match in the 'naam' field.
    crs : str
        Key in CRS: 'wgs84' or 'utm31n'.

    Returns
    -------
    dict
        GeoJSON FeatureCollection.
    """
    url = f"{BASE_URL}/{LAYERS[layer]}/query"
    params = {
        "where": f"naam LIKE '%{farm_name}%'",
        "outFields": "*",
        "outSR": CRS[crs]["epsg"],
        "f": "geojson",
        "returnGeometry": "true",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_all_layers(farm_name="HKN", crs="wgs84"):
    """Fetch all layers, returning {layer_name: geojson} for successful ones.

    Parameters
    ----------
    farm_name : str
    crs : str

    Returns
    -------
    dict[str, dict]
        layer_name > GeoJSON FeatureCollection (only non-empty).
    """
    results = {}
    for name in LAYERS:
        try:
            data = get_layer(name, farm_name, crs)
            feats = data.get("features", [])
            if len(feats) > 0:
                if name == "turbines":
                    turbine_feats = feats
                    # move substation out of turbines layer!
                    # substation labeled OHVS or others
                    other_feats = [
                        f
                        for f in turbine_feats
                        if f["properties"].get("opmerking") != "Turbine"
                    ]
                    if len(other_feats) > 0:
                        results[f"{name}_other"] = {**data, "features": other_feats}
                    # HKN storage floater is returned here as turbine!
                    # move it out of turbine layer
                    is_storage = lambda f: (f["id"] == 616) and (
                        f["properties"].get("turbine_nr") == "HNA6"
                    )
                    storage_feats = [
                        {**f, "properties": {**f["properties"], "type": "storage"}}
                        for f in turbine_feats
                        if is_storage(f)
                    ]
                    if len(storage_feats) > 0:
                        results["storage"] = {**data, "features": storage_feats}
                    turbine_feats = [f for f in turbine_feats if not is_storage(f)]
                    data["features"] = [
                        f
                        for f in turbine_feats
                        if f["properties"].get("opmerking") == "Turbine"
                    ]
                results[name] = data
                print(f"  {name:20s} > {len(feats)} feature(s)")
            else:
                print(f"  {name:20s} > empty")
        except Exception as e:
            print(f"  {name:20s} > error ({e})")
    return results


def plot_polygon_loop(ax, geom, **kwargs):
    """Plot closed loop json Polygon or MultiPolygon geometry."""
    gtype = geom["type"]
    coords = geom["coordinates"]
    # normalize to list of polygons
    polys = coords if gtype == "MultiPolygon" else [coords]
    for poly in polys:
        for segs in poly:
            xs, ys = zip(*segs)
            ax.plot(xs, ys, **kwargs)


def plot_all(results, crs="wgs84", disp_name=True):
    """Plot all fetched layers on a single figure.

    Parameters
    ----------
    results : dict[str, dict]
        Output of fetch_all_layers.
    crs : str
        Key in CRS.
    """
    xname, yname = CRS[crs]["x"], CRS[crs]["y"]
    fig, ax = plt.subplots(figsize=(8, 8))

    for name, geojson in results.items():
        features = geojson["features"]
        gtype = features[0]["geometry"]["type"]
        if gtype == "Point":
            xs = [f["geometry"]["coordinates"][0] for f in features]
            # xs = [f["properties"]["utm_x"] for f in features]  # not compatible with boundary
            ys = [f["geometry"]["coordinates"][1] for f in features]
            # ys = [f["properties"]["utm_y"] for f in features]
            wt_name = [f["properties"]["turbine_nr"] for f in features]
            ax.scatter(xs, ys, zorder=5, marker="2", label=f"{name} ({len(features)})")
            for i, n in enumerate(wt_name):
                ax.text(xs[i], ys[i], n)
        else:
            for feat in features:
                plot_polygon_loop(ax, feat["geometry"])
    ax.legend(loc="upper left")
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title("Hollandse Kust Noord — all available layers")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()


def get_nlfarm_xyb(farm="HKN", wgs_coords=False):
    results = fetch_all_layers(farm, crs="utm")
    boundaries = np.array(
        results["boundary"]["features"][0]["geometry"]["coordinates"]
    ).squeeze()  # only the utm boundary is relevant for optimization
    x = [f["properties"]["utm_x"] for f in results["turbines"]["features"]]
    x = np.array(x)  # always utm
    y = [f["properties"]["utm_y"] for f in results["turbines"]["features"]]
    y = np.array(y)  # always utm
    if wgs_coords:
        results_wgs = fetch_all_layers(farm, crs="wgs")
        x_wgs = [
            f["geometry"]["coordinates"][0] for f in results_wgs["turbines"]["features"]
        ]
        x_wgs = np.array(x_wgs)
        y_wgs = [
            f["geometry"]["coordinates"][1] for f in results_wgs["turbines"]["features"]
        ]
        y_wgs = np.array(y_wgs)
        return x, y, boundaries, x_wgs, y_wgs
    return x, y, boundaries


if __name__ == "__main__":
    farm = "HKN"
    crs = "utm"
    print(f"Fetching layers for '{farm}':")
    results = fetch_all_layers(farm, crs)
    print(f"\n{len(results)} layer(s) with data\n")
    plot_all(results, crs)
    boundaries = np.array(
        results["boundary"]["features"][0]["geometry"]["coordinates"]
    ).squeeze()
    x = [f["properties"]["utm_x"] for f in results["turbines"]["features"]]
    x = np.array(x)
    y = [f["properties"]["utm_y"] for f in results["turbines"]["features"]]
    y = np.array(y)
    plt.scatter(x, y, marker="2")
    x, y, boundaries, x_wgs, y_wgs = get_nlfarm_xyb(farm, wgs_coords=True)
    plt.scatter(x, y, marker="o")
    plt.axis("equal")
    plt.show()
