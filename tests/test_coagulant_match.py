"""Coagulant jar-test matching: settled water quality must drive ranking."""
import unittest

from aquaflow.api import AquaFlowService
from aquaflow.treatment.coagulant import Jar, choose, recommend

TARGET = {"ntu": 50.0, "alkalinity": 100.0, "temp": 20.0}


def run(jars):
    return AquaFlowService().run_workflow(
        "coagulant_match", {"jars": jars, "target": TARGET}
    )


def jar(raw_ntu, alkalinity, temp, dose, settled_ntu):
    return {
        "raw_ntu": raw_ntu,
        "alkalinity": alkalinity,
        "temp": temp,
        "dose": dose,
        "settled_ntu": settled_ntu,
    }


class CoagulantMatchWorkflowTest(unittest.TestCase):
    def test_raw_similar_but_poorly_settled_loses(self):
        # Perfect raw-water match but terrible settled turbidity must not win
        # over a slightly less similar jar with far better settled quality.
        result = run([
            jar(50.0, 100.0, 20.0, 30.0, 8.0),   # raw twin, settled 8.0 NTU
            jar(56.0, 100.0, 21.0, 25.0, 1.2),   # raw gap, settled 1.2 NTU
        ])
        self.assertEqual(result["decision"], "dose")
        self.assertEqual(result["value"], 25.0)
        # Published evidence carries the winning settled water quality.
        self.assertEqual(result["evidence"]["settled_ntu"], 1.2)
        self.assertEqual(result["evidence"]["candidate_count"], 2)

    def test_clearly_better_settled_wins_from_further_raw_water(self):
        rows = [
            Jar(52.0, 98.0, 19.0, 30.0, 6.5),    # close raw, settled 6.5
            Jar(58.0, 105.0, 22.0, 18.0, 0.9),   # farther raw, settled 0.9
        ]
        self.assertEqual(recommend(rows, 50.0, 100.0, 20.0), 18.0)
        self.assertEqual(choose(rows, 50.0, 100.0, 20.0).settled_ntu, 0.9)

    def test_full_tie_breaks_to_lower_dose_deterministically(self):
        rows = [
            Jar(50.0, 100.0, 20.0, 20.0, 1.5),
            Jar(50.0, 100.0, 20.0, 15.0, 1.5),   # identical score, lower dose
        ]
        self.assertEqual(recommend(rows, 50.0, 100.0, 20.0), 15.0)
        # Order-independent: reversing the input cannot change the pick.
        self.assertEqual(recommend(rows[::-1], 50.0, 100.0, 20.0), 15.0)
        # Score tie with different settled quality prefers the clearer jar.
        tied_score = [
            Jar(50.0, 100.0, 20.0, 10.0, 3.0),   # score 3.0
            Jar(51.0, 100.0, 20.0, 40.0, 2.3),   # score 0.7 + 2.3 = 3.0
        ]
        self.assertEqual(recommend(tied_score, 50.0, 100.0, 20.0), 40.0)

    def test_empty_jar_set_rejected(self):
        with self.assertRaises(ValueError):
            recommend([], 50.0, 100.0, 20.0)
        with self.assertRaises(ValueError):
            run([])

    def test_normalization_preserves_settled_evidence(self):
        result = run([jar(50.0, 100.0, 20.0, 30.0, 4.4)])
        # No legacy downgrade and no settled override injected upstream.
        self.assertEqual(result["audit"]["contract"], "v1")
        self.assertEqual(result["evidence"]["settled_ntu"], 4.4)
        self.assertEqual(result["value"], 30.0)


if __name__ == "__main__":
    unittest.main()
