import json

import numpy as np
import pytest

from benchmarks import modeling
from sniptext.pipelines import DEFAULT, Pipeline
from sniptext.router import (
    CONF_STAT_NAMES,
    FORMAT,
    Router,
    RouterModel,
    choose_actions,
    conf_stats,
    load_model,
)

A = Pipeline("light_psm6", ("light",), "6")
B = Pipeline("light_up2_psm6", ("light", "up2"), "6")
GBR = {"kind": "gbr", "n_estimators": 30, "max_depth": 2, "learning_rate": 0.1}
NAMES = ["f0", "f1", "f2"]


def data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 3))
    # action A is good when f0 is small, action B when it is large
    y = np.column_stack([X[:, 0], 1.0 - X[:, 0]]) * 0.8 + rng.normal(0, 0.01, (n, 2))
    return X, np.clip(y, 0, 1)


def write(tmp_path, policy="pre_ocr", spec=GBR, time_weight=0.0, costs=(0.1, 0.2), names=NAMES):
    X, y = data()
    if policy == "cascade":
        X = np.hstack([X, np.zeros((len(X), len(CONF_STAT_NAMES)))])
        names = names + list(CONF_STAT_NAMES)
    models = modeling.fit(spec, X, y)
    model = modeling.export_model(policy, names, [A, B], list(costs), time_weight, spec, models)
    path = tmp_path / "router_model.json"
    modeling.write_model(path, model)
    return path, models, X


class TestConfStats:
    def test_no_confidences(self):
        assert conf_stats(None).tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
        assert conf_stats([[]]).tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]

    def test_values(self):
        stats = conf_stats([[0.9, 0.5], [0.4]])
        assert stats[0] == pytest.approx(0.6) and stats[1] == 0.4
        assert stats[3] == pytest.approx(2 / 3) and stats[4] == 0.03


class TestChoice:
    def test_lowest_predicted_cer_wins_at_zero_weight(self):
        assert choose_actions(np.array([[0.3, 0.1]]), np.array([0.1, 0.5]), 0.0).tolist() == [1]

    def test_time_weight_buys_the_cheaper_action(self):
        assert choose_actions(np.array([[0.3, 0.1]]), np.array([0.1, 0.5]), 1.0).tolist() == [0]

    def test_ties_go_to_the_cheaper_action(self):
        assert choose_actions(np.array([[0.2, 0.2]]), np.array([0.5, 0.1]), 0.0).tolist() == [1]


class TestTrees:
    def stump(self, threshold=0.5):
        return {"kind": "gbr", "init": 10.0, "rate": 0.5, "depth": 1,
                "feature": [[0, -1, -1]], "threshold": [[threshold, 0.0, 0.0]],
                "left": [[1, 0, 0]], "right": [[2, 0, 0]], "value": [[0.0, 1.0, 3.0]]}  # fmt: skip

    def model(self, regressor):
        return RouterModel.from_dict(
            {
                "format": FORMAT,
                "policy": "pre_ocr",
                "feature_names": ["f0"],
                "actions": [A.to_dict(), B.to_dict()],
                "costs": [0.1, 0.1],
                "time_weight": 0.0,
                "regressors": [regressor, regressor],
            }  # fmt: skip
        )

    def test_a_hand_made_stump(self):
        trees = self.model(self.stump()).regressors[0]
        assert trees.predict(np.array([[0.2], [0.9]])).tolist() == [10.5, 11.5]

    def test_inputs_are_compared_as_float32_like_scikit_learn(self):
        # float32(0.1) is slightly above the float64 threshold 0.1, so the row goes right
        trees = self.model(self.stump(threshold=0.1)).regressors[0]
        assert trees.predict(np.array([[0.1]])).tolist() == [11.5]

    def test_predictions_are_clipped_to_a_valid_cer(self):
        assert self.model(self.stump()).predict_cer([[0.2]]).tolist() == [[1.0, 1.0]]


class TestParityWithScikitLearn:
    @pytest.mark.parametrize(
        "spec",
        [
            GBR,
            {"kind": "gbr", "n_estimators": 60, "max_depth": 3, "learning_rate": 0.05},
            {"kind": "ridge", "alpha": 1.0},
        ],
    )
    def test_numpy_inference_matches(self, tmp_path, spec):
        path, models, X = write(tmp_path, spec=spec)
        fresh = np.random.default_rng(5).random((300, 3))
        for rows in (X, fresh):
            assert np.allclose(load_model(path).predict_cer(rows), modeling.predict(models, rows),
                               atol=1e-9, rtol=0)  # fmt: skip

    def test_the_file_is_plain_json_and_stable(self, tmp_path):
        path, _, _ = write(tmp_path)
        first = path.read_bytes()
        write(tmp_path)
        assert path.read_bytes() == first
        assert json.loads(first)["format"] == FORMAT


class TestRouter:
    def test_routes_by_the_feature_that_separates_the_actions(self, tmp_path):
        path, _, _ = write(tmp_path)
        router = Router(path)
        assert router.available and router.policy == "pre_ocr"
        assert router.actions == (A, B) and router.default == A
        assert router.choose([0.05, 0.5, 0.5]) == A
        assert router.choose([0.95, 0.5, 0.5]) == B
        assert router.choose_vector([0.95, 0.5, 0.5]) == 1

    def test_time_weight_override(self, tmp_path):
        path, _, _ = write(tmp_path)
        assert Router(path, time_weight=100.0).choose([0.95, 0.5, 0.5]) == A

    def test_shipped_time_weight_is_used_by_default(self, tmp_path):
        path, _, _ = write(tmp_path, time_weight=100.0)
        assert Router(path).choose([0.95, 0.5, 0.5]) == A

    def test_cascade_appends_the_confidence_statistics(self, tmp_path):
        path, _, _ = write(tmp_path, policy="cascade")
        router = Router(path)
        assert router.policy == "cascade"
        assert router.choose([0.95, 0.5, 0.5], [[0.9, 0.2]]) == B

    def test_static_model_never_routes(self, tmp_path):
        model = modeling.export_model("static", NAMES, [B], [0.2], 0.0, None, [])
        path = tmp_path / "m.json"
        modeling.write_model(path, model)
        router = Router(path)
        assert not router.available and router.policy == "static"
        assert router.default == B and router.choose([0.1, 0.1, 0.1]) == B

    @pytest.mark.parametrize(
        "content", [None, "not json", "{}", json.dumps({"format": FORMAT + 1})]
    )
    def test_missing_or_malformed_file_falls_back_to_the_default(self, tmp_path, content):
        path = tmp_path / "m.json"
        if content is not None:
            path.write_text(content)
        router = Router(path)
        assert not router.available
        assert router.default == DEFAULT and router.actions == (DEFAULT,)
        assert router.choose([0.1, 0.2, 0.3]) == DEFAULT

    def test_a_model_whose_parts_disagree_is_rejected(self, tmp_path):
        path, _, _ = write(tmp_path)
        broken = json.loads(path.read_text())
        broken["costs"] = [0.1]
        path.write_text(json.dumps(broken))
        assert not Router(path).available

    @pytest.mark.parametrize("vector", [[0.9, 0.5], [0.9, float("nan"), 0.5]])
    def test_a_bad_feature_vector_gets_the_default(self, tmp_path, vector):
        path, _, _ = write(tmp_path)
        assert Router(path).choose(vector) == A

    def test_the_file_is_read_once(self, tmp_path, monkeypatch):
        path, _, _ = write(tmp_path)
        router = Router(path)
        router.choose([0.1, 0.1, 0.1])
        path.unlink()
        assert router.choose([0.95, 0.5, 0.5]) == B


class TestShippedModel:
    def test_it_loads_and_its_default_is_the_package_default(self):
        model = load_model()
        assert model.actions[0] == DEFAULT
        assert 1 <= len(model.actions) <= 4
        assert len({action.name for action in model.actions}) == len(model.actions)

    def test_the_router_answers_for_a_plain_feature_vector(self):
        router = Router()
        model = load_model()
        x = np.full(len(model.feature_names), 0.5)
        assert router.actions[router.choose_vector(x)] in model.actions
