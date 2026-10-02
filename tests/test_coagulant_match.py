import unittest

from aquaflow.treatment.coagulant import (
    Jar,
    Match,
    rank_candidates,
    recommend,
    RAW_WEIGHT,
    SETTLED_WEIGHT,
)
from aquaflow.workflow_models import normalize
from aquaflow.workflow_service import WorkflowService


def jar(raw=10.0, alk=50.0, temp=20.0, dose=30.0, settled=3.0):
    return Jar(raw, alk, temp, dose, settled)


def payload(jars, target=None):
    return {
        "jars": [
            {"raw_ntu": j.raw_ntu, "alkalinity": j.alkalinity, "temp": j.temp,
             "dose": j.dose, "settled_ntu": j.settled_ntu}
            for j in jars
        ],
        "target": target or {"ntu": 10.0, "alkalinity": 50.0, "temp": 20.0},
    }


class CoagulantRankingTests(unittest.TestCase):
    def test_raw_close_but_settled_worse_loses(self):
        # Raw water nearly identical but it settled poorly; the farther raw
        # water jar produced far better settled water and must win.
        near_raw_bad = jar(raw=10.1, dose=30.0, settled=10.0)   # sim .07
        far_raw_good = jar(raw=11.0, dose=25.0, settled=1.0)   # sim .70
        self.assertEqual(recommend([near_raw_bad, far_raw_good], 10, 50, 20), 25.0)
        # and ordering must not depend on input order
        self.assertEqual(recommend([far_raw_good, near_raw_bad], 10, 50, 20), 25.0)

    def test_clearly_better_settled_result_wins(self):
        excellent = jar(raw=13.0, dose=20.0, settled=0.5)
        mediocre = jar(raw=10.5, dose=40.0, settled=5.0)
        ranked = rank_candidates([excellent, mediocre], 10, 50, 20)
        self.assertEqual(ranked[0].jar.dose, 20.0)
        self.assertLess(ranked[0].jar.settled_ntu, ranked[1].jar.settled_ntu)

    def test_score_blends_raw_and_settled_weights(self):
        m = rank_candidates([jar(raw=12.0, settled=2.0)], 10, 50, 20)[0]
        # similarity = |12-10|*.7 = 1.4; score = .6*1.4 + .4*2 = 1.64
        self.assertAlmostEqual(m.raw_similarity, 1.4)
        self.assertAlmostEqual(m.score, RAW_WEIGHT * 1.4 + SETTLED_WEIGHT * 2.0)
        self.assertEqual((RAW_WEIGHT, SETTLED_WEIGHT), (0.6, 0.4))

    def test_exact_blended_tie_lower_settled_wins(self):
        p = jar(raw=10.0, dose=40.0, settled=3.0)                       # sim 0
        q = jar(raw=10.0 + 2.0 / 0.7, dose=50.0, settled=0.0)           # sim 2.0
        ranked = rank_candidates([p, q], 10, 50, 20)
        self.assertAlmostEqual(ranked[0].score, ranked[1].score)
        self.assertEqual(ranked[0].jar.settled_ntu, 0.0)
        self.assertEqual(ranked[0].index, 1)

    def test_tie_on_score_and_settled_lower_dose_wins(self):
        high_dose = jar(raw=10.0, dose=40.0, settled=3.0)
        low_dose = jar(raw=10.0, dose=30.0, settled=3.0)
        self.assertEqual(recommend([high_dose, low_dose], 10, 50, 20), 30.0)

    def test_fully_tied_candidates_keep_input_order(self):
        first = jar(raw=10.1, dose=30.0, settled=3.0)
        second = jar(raw=9.9, dose=30.0, settled=3.0)
        ranked = rank_candidates([first, second], 10, 50, 20)
        self.assertAlmostEqual(ranked[0].score, ranked[1].score)
        self.assertEqual([m.index for m in ranked], [0, 1])
        # swapping inputs swaps which physical row wins: deterministic by order
        swapped = rank_candidates([second, first], 10, 50, 20)
        self.assertEqual(swapped[0].index, 0)

    def test_empty_jar_set_rejected_at_domain_layer(self):
        for fn in (recommend, rank_candidates):
            with self.assertRaises(ValueError):
                fn([], 10, 50, 20)

    def test_ranked_result_is_match_with_index(self):
        m = rank_candidates([jar(dose=15.0)], 10, 50, 20)[0]
        self.assertIsInstance(m, Match)
        self.assertEqual(m.index, 0)
        self.assertEqual(m.jar.dose, 15.0)


class CoagulantWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.svc = WorkflowService()

    def test_full_path_settled_water_quality_drives_published_dose(self):
        jars = [
            jar(raw=10.1, dose=30.0, settled=10.0),
            jar(raw=11.0, dose=25.0, settled=1.0),
        ]
        res = self.svc.run("coagulant_match", payload(jars))
        self.assertEqual(res["decision"], "dose")
        self.assertEqual(res["value"], 25.0)
        ev = res["evidence"]
        self.assertEqual(ev["candidate_count"], 2)
        self.assertEqual(ev["winning"]["dose"], 25.0)
        self.assertEqual(ev["winning"]["settled_ntu"], 1.0)
        # full ordered ranking is published as evidence
        self.assertEqual([r["dose"] for r in ev["ranking"]], [25.0, 30.0])
        self.assertAlmostEqual(ev["runner_up"]["settled_ntu"], 10.0)
        self.assertEqual(ev["weights"], {"raw_similarity": 0.6, "settled_turbidity": 0.4})

    def test_normalization_preserves_settled_turbidity(self):
        jars = [jar(settled=7.5)]
        env = normalize("coagulant_match", payload(jars))
        self.assertEqual(env.payload["jars"][0]["settled_ntu"], 7.5)
        self.assertNotIn("_settled_override", env.payload)
        self.assertEqual(env.payload["_contract"], "v1")

    def test_empty_jar_set_rejected_end_to_end(self):
        with self.assertRaises(ValueError):
            self.svc.run("coagulant_match", payload([]))

    def test_missing_settled_turbidity_rejected(self):
        bad = {"jars": [{"raw_ntu": 10, "alkalinity": 50, "temp": 20, "dose": 30}],
               "target": {"ntu": 10, "alkalinity": 50, "temp": 20}}
        with self.assertRaises(ValueError):
            self.svc.run("coagulant_match", bad)

    def test_exact_tie_end_to_end_prefers_lower_dose(self):
        jars = [jar(raw=10, dose=40, settled=3), jar(raw=10, dose=30, settled=3)]
        res = self.svc.run("coagulant_match", payload(jars))
        self.assertEqual(res["value"], 30.0)
        scores = [r["score"] for r in res["evidence"]["ranking"]]
        self.assertAlmostEqual(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
