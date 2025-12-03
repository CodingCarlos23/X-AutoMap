import os
import re
import unittest

import numpy as np
from PIL import Image

# Avoid GUI requirements
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from viewers.DetailBeta import ScansGroupedViewer  # noqa: E402


def load_channels(coarse_id, fine_id, elements="CuCaFe"):
    """Load channels for a fine scan, honoring a requested element set when present."""

    def parse_suffix_elements(dirname):
        if "_" not in dirname:
            return []
        suffix = dirname.split("_", 1)[1]
        return re.findall(r"[A-Z][a-z]?", suffix)

    def order_elements(available, suffix_order, requested_order):
        preferred = requested_order or ["Cu", "Ca", "Fe"]
        if set(preferred).issubset(available):
            return preferred
        if suffix_order and all(el in available for el in suffix_order):
            return suffix_order
        return sorted(available)

    requested_order = re.findall(r"[A-Z][a-z]?", elements) if elements else []
    requested_set = set(requested_order)

    base_root = os.path.join("data", "scans_grouped")
    prefix = f"scan_{coarse_id}_" if fine_id == coarse_id else f"detsum_{fine_id}_"

    best = None
    for dirname in sorted(os.listdir(base_root)):
        if not (dirname == str(coarse_id) or dirname.startswith(f"{coarse_id}_")):
            continue

        if elements and "_" in dirname and elements not in dirname and dirname != str(coarse_id):
            continue

        dirpath = os.path.join(base_root, dirname)
        files = [f for f in os.listdir(dirpath) if f.startswith(prefix) and f.endswith(".tiff")]
        if not files:
            continue

        available_elements = {f[len(prefix) : -5] for f in files}
        ordered_elements = order_elements(available_elements, parse_suffix_elements(dirname), requested_order)
        specified_match = bool(requested_set) and requested_set.issubset(available_elements)
        candidate = {
            "dir": dirpath,
            "elements": ordered_elements,
            "count": len(available_elements),
            "preferred_order": ordered_elements == requested_order or ordered_elements == ["Cu", "Ca", "Fe"],
            "specified_match": specified_match,
        }

        if best is None:
            best = candidate
        elif candidate["specified_match"] and not best["specified_match"]:
            best = candidate
        elif candidate["specified_match"] == best["specified_match"] and candidate["count"] > best["count"]:
            best = candidate
        elif (
            candidate["specified_match"] == best["specified_match"]
            and candidate["count"] == best["count"]
            and candidate["preferred_order"]
            and not best["preferred_order"]
        ):
            best = candidate

    if best is None:
        raise FileNotFoundError(f"No matching scan data for coarse {coarse_id}, fine {fine_id} (elements {elements})")

    if fine_id == coarse_id:
        filenames = [f"scan_{coarse_id}_{el}.tiff" for el in best["elements"]]
    else:
        filenames = [f"detsum_{fine_id}_{el}.tiff" for el in best["elements"]]

    return [np.array(Image.open(os.path.join(best["dir"], name))) for name in filenames]


class ClassificationRegressionTests(unittest.TestCase):
    def setUp(self):
        self.viewer = ScansGroupedViewer.__new__(ScansGroupedViewer)

    def test_expected_labels(self):
        def rec(coarse, fine, elements="CuCaFe"):
            return {"coarse": coarse, "fine": fine, "elements": elements}

        expected = {
            "Together": [
                rec(367592, 367593),
                rec(367592, 367594),
                rec(367592, 367595),
                rec(367596, 367597),
                rec(367596, 367598),
                rec(367596, 367599),
                # Reclassified from viewers/clicked.txt overrides (CrFeMn set)
                rec(367915, 367918, "CrFeMn"),
                rec(367903, 367905, "CrFeMn"),
                rec(367862, 367866, "CrFeMn"),
                rec(367862, 367864, "CrFeMn"),
                rec(367760, 367765, "CrFeMn"),
                rec(367786, 367788, "CrFeMn"),
                rec(367675, 367677, "CrFeMn"),
                rec(367720, 367721, "CrFeMn"),
            ],
            "Partial": [
                rec(367899, 367902),
                rec(367600, 367604),
                rec(367600, 367608),
                rec(367582, 367588),
                rec(367609, 367613),
                rec(367622, 367625),
                rec(367630, 367633),
                rec(367658, 367662),
                rec(367686, 367687),
                rec(367675, 367678),
                rec(367692, 367697),
                rec(367703, 367707),
                rec(367726, 367730),
                # From viewers/clicked.txt; visually partial but classified separate
                rec(367921, 367926, "CrFeMn"),
                rec(367915, 367916, "CrFeMn"),
                rec(367857, 367859, "CrFeMn"),
                rec(367816, 367817, "CrFeMn"),
                rec(367786, 367787, "CrFeMn"),
                rec(369155, 369157, "FeCaSi"),
            ],
            "Separate": [
                # False partial/together fine scans
                rec(367582, 367583),
                rec(367703, 367704),
                rec(367741, 367742),
                rec(367748, 367750),
                rec(367748, 367752),
                rec(367803, 367804),
                rec(367803, 367806),
                rec(367638, 367638),
                rec(367638, 367639),
                rec(367614, 367616),
                # Overrides near 367900
                rec(367899, 367900),
                rec(367899, 367901),
                # Coarse-level checks for false Together/Partial
                rec(367592, 367592),
                rec(367596, 367596),
                rec(367614, 367614),
                rec(367582, 367582),
                rec(367703, 367703),
                rec(367741, 367741),
                rec(367748, 367748),
                rec(367803, 367803),
            ],
        }

        total = 0
        failures = []
        per_label = {label: {"total": len(pairs), "passed": 0, "failed": 0} for label, pairs in expected.items()}

        per_label_results = {label: [] for label in expected}

        for expected_label, pairs in expected.items():
            for entry in pairs:
                coarse_id, fine_id, elements = entry["coarse"], entry["fine"], entry["elements"]
                channels = load_channels(coarse_id, fine_id, elements=elements)
                total += 1
                result = self.viewer.typeDetector(channels, fine_id=fine_id, elements=elements)
                per_label_results[expected_label].append((coarse_id, fine_id, elements, result))
                if result != expected_label:
                    failures.append((coarse_id, fine_id, elements, expected_label, result))
                    per_label[expected_label]["failed"] += 1
                else:
                    per_label[expected_label]["passed"] += 1

        passed = total - len(failures)
        print(f"Classification check: passed {passed}/{total}, failed {len(failures)}")
        for label, stats in per_label.items():
            print(
                f"  {label}: passed {stats['passed']}/{stats['total']}, failed {stats['failed']}"
            )

        for label, results in per_label_results.items():
            print(f"\nResults for category '{label}':")
            for coarse_id, fine_id, elements, predicted in results:
                status = "OK" if predicted == label else f"FAIL (got {predicted})"
                print(f"  coarse {coarse_id}, fine {fine_id} [{elements}]: {status}")

        if failures:
            details = "\n".join(
                f"coarse {c}, fine {f} (elements {els}): expected {exp}, got {got}" for c, f, els, exp, got in failures
            )
            self.fail(f"Classification mismatches:\n{details}")


if __name__ == "__main__":
    unittest.main()
