"""Build the purpose-made 3.5 km SUMO motorway network."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET


PACKAGE_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = PACKAGE_DIR / "scenario"
NET_FILE = SCENARIO_DIR / "highway.net.xml"


def _write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="    ")
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _find_sumo_binary(name: str) -> str:
    try:
        import sumo

        candidate = Path(sumo.SUMO_HOME) / "bin" / f"{name}.exe"
        if candidate.exists():
            return str(candidate)
    except ImportError:
        pass

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / f"{name}.exe"
        if candidate.exists():
            return str(candidate)

    binary = shutil.which(name)
    if binary:
        return binary
    raise FileNotFoundError(f"Could not locate {name}; install SUMO or set SUMO_HOME")


def _build_nodes(path: Path) -> None:
    root = ET.Element("nodes")
    for node_id, x, y, node_type in (
        ("n0", 0, 0, None),
        ("n600", 600, 0, "priority"),
        ("n1400", 1400, 0, "priority"),
        ("n1600", 1600, 0, "priority"),
        ("n1717", 1716.667, 0, "priority"),
        ("n2100", 2100, 0, "priority"),
        ("n2300", 2300, 0, "priority"),
        ("n3100", 3100, 0, "priority"),
        ("n3500", 3500, 0, None),
        ("r0", 1200, -180, None),
        ("r1600", 1600, -7, "priority"),
    ):
        attributes = {"id": node_id, "x": str(x), "y": str(y)}
        if node_type is not None:
            attributes["type"] = node_type
        ET.SubElement(root, "node", attributes)
    _write_xml(root, path)


def _build_edges(path: Path) -> None:
    root = ET.Element("edges")
    common = {
        "priority": "3",
        "numLanes": "3",
        "speed": "33.33",
        "spreadType": "center",
        "width": "3.5",
    }
    ET.SubElement(root, "edge", id="main0", **{"from": "n0", "to": "n600", "length": "600", **common})

    curve_points = []
    control_y = (0.0, 22.0, 0.0, -22.0, 0.0)
    for segment in range(4):
        for offset in range(0, 200, 5):
            u = offset / 200.0
            # Quintic smoothstep gives zero tangent and curvature jumps where
            # each left/right bend meets the short transition between bends.
            blend = u**3 * (u * (u * 6.0 - 15.0) + 10.0)
            x = 600 + segment * 200 + offset
            y = control_y[segment] + (control_y[segment + 1] - control_y[segment]) * blend
            curve_points.append(f"{x:.2f},{y:.2f}")
    curve_points.append("1400.00,0.00")
    ET.SubElement(
        root,
        "edge",
        id="curve",
        **{
            "from": "n600",
            "to": "n1400",
            "length": "800",
            "shape": " ".join(curve_points),
            **common,
        },
    )
    ET.SubElement(root, "edge", id="mergePre", **{"from": "n1400", "to": "n1600", "length": "200", **common})
    ET.SubElement(
        root,
        "edge",
        id="mainAlongRamp",
        **{
            "from": "n1600",
            "to": "n1717",
            "length": "116.667",
            **common,
        },
    )

    # The first 116.7 m remains a physically separate solid-line acceleration
    # section.  The remaining 383.3 m is a shared four-lane edge, so SUMO can
    # perform real lane changes from the auxiliary lane into the three
    # unchanged mainline lanes before the terminal taper.
    open_edge = ET.SubElement(
        root,
        "edge",
        id="mergeOpen",
        **{
            "from": "n1717",
            "to": "n2100",
            "priority": "3",
            "numLanes": "4",
            "speed": "33.33",
            "length": "383.333",
            "shape": "1716.667,-1.75 2100,-1.75",
            "spreadType": "center",
            "width": "3.5",
        },
    )
    auxiliary_points = ["1716.667,-7.00", "1950.00,-7.00"]
    for offset in range(5, 151, 5):
        u = offset / 150.0
        blend = u**3 * (u * (u * 6.0 - 15.0) + 10.0)
        auxiliary_points.append(f"{1950.0 + offset:.2f},{-7.0 + 3.5 * blend:.2f}")
    lane_shapes = (
        " ".join(auxiliary_points),
        "1716.667,-3.50 2100,-3.50",
        "1716.667,0.00 2100,0.00",
        "1716.667,3.50 2100,3.50",
    )
    for index, shape in enumerate(lane_shapes):
        ET.SubElement(
            open_edge,
            "lane",
            index=str(index),
            speed="33.33",
            width="3.5",
            shape=shape,
        )
    ET.SubElement(root, "edge", id="mergePost", **{"from": "n2100", "to": "n2300", "length": "200", **common})
    ET.SubElement(root, "edge", id="wave", **{"from": "n2300", "to": "n3100", "length": "800", **common})
    ET.SubElement(root, "edge", id="recovery", **{"from": "n3100", "to": "n3500", "length": "400", **common})
    ramp_points = []
    for sample in range(81):
        u = sample / 80.0
        inverse = 1.0 - u
        x = (
            inverse**3 * 1200.0
            + 3.0 * inverse**2 * u * 1325.0
            + 3.0 * inverse * u**2 * 1475.0
            + u**3 * 1600.0
        )
        y = (
            inverse**3 * -180.0
            + 3.0 * inverse**2 * u * -180.0
            + 3.0 * inverse * u**2 * -12.0
            + u**3 * -7.0
        )
        ramp_points.append(f"{x:.2f},{y:.2f}")
    ET.SubElement(
        root,
        "edge",
        id="ramp",
        **{
            "from": "r0",
            "to": "r1600",
            "priority": "1",
            "numLanes": "1",
            "speed": "27.78",
            "length": "400",
            "shape": " ".join(ramp_points),
            "spreadType": "center",
            "width": "3.5",
        },
    )
    ET.SubElement(
        root,
        "edge",
        id="rampAccel",
        **{
            "from": "r1600",
            "to": "n1717",
            "priority": "2",
            "numLanes": "1",
            "speed": "33.33",
            "length": "116.667",
            "shape": "1600,-7 1716.667,-7",
            "spreadType": "center",
            "width": "3.5",
        },
    )
    _write_xml(root, path)


def _build_connections(path: Path) -> None:
    root = ET.Element("connections")
    chain = (
        ("main0", "curve"),
        ("curve", "mergePre"),
        ("mergePre", "mainAlongRamp"),
        ("mergePost", "wave"),
        ("wave", "recovery"),
    )
    for source, target in chain:
        for lane in range(3):
            ET.SubElement(
                root,
                "connection",
                **{
                    "from": source,
                    "to": target,
                    "fromLane": str(lane),
                    "toLane": str(lane),
                },
            )
    for lane in range(3):
        ET.SubElement(
            root,
            "connection",
            **{
                "from": "mainAlongRamp",
                "to": "mergeOpen",
                "fromLane": str(lane),
                "toLane": str(lane + 1),
            },
        )
        ET.SubElement(
            root,
            "connection",
            **{
                "from": "mergeOpen",
                "to": "mergePost",
                "fromLane": str(lane + 1),
                "toLane": str(lane),
            },
        )
    ET.SubElement(
        root,
        "connection",
        **{
            "from": "mergeOpen",
            "to": "mergePost",
            "fromLane": "0",
            "toLane": "0",
        },
    )
    ET.SubElement(
        root,
        "connection",
        **{"from": "ramp", "to": "rampAccel", "fromLane": "0", "toLane": "0"},
    )
    ET.SubElement(
        root,
        "connection",
        **{
            "from": "rampAccel",
            "to": "mergeOpen",
            "fromLane": "0",
            "toLane": "0",
        },
    )
    _write_xml(root, path)


def build_network(force: bool = True) -> Path:
    """Generate the plain XML sources and compile ``highway.net.xml``."""

    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    if NET_FILE.exists() and not force:
        return NET_FILE

    nodes = SCENARIO_DIR / "highway.nod.xml"
    edges = SCENARIO_DIR / "highway.edg.xml"
    connections = SCENARIO_DIR / "highway.con.xml"
    _build_nodes(nodes)
    _build_edges(edges)
    _build_connections(connections)

    command = [
        _find_sumo_binary("netconvert"),
        "--node-files",
        str(nodes),
        "--edge-files",
        str(edges),
        "--connection-files",
        str(connections),
        "--output-file",
        str(NET_FILE),
        "--no-turnarounds",
        "true",
        "--junctions.corner-detail",
        "5",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"netconvert failed:\n{result.stdout}\n{result.stderr}")
    return NET_FILE


if __name__ == "__main__":
    print(build_network())
