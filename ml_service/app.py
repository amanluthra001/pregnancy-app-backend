import os
import sys
import json
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

# Ensure current directory and ML folder are on path so ML modules can be imported
ROOT = os.path.abspath(os.path.dirname(__file__))
ML_ROOT = os.path.join(ROOT, 'ML')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if ML_ROOT not in sys.path:
    sys.path.insert(0, ML_ROOT)

# Try to import custom fetal module (defines ManualGaussianNB) so unpickling can resolve the class
try:
    # Attempt both package-style and direct import depending on how the model was pickled
    import ML.fetal.gaussian_naive_final as _gf
except Exception:
    try:
        import gaussian_naive_final as _gf
    except Exception:
        _gf = None
# Also attempt to import maternal custom trees (defines KLDecisionTree etc.)
try:
    import ML.mother.custom_trees as _mt
except Exception:
    try:
        import custom_trees as _mt
    except Exception:
        _mt = None

APP = Flask(__name__)
CORS(APP)

# Paths to models (relative to current directory)
FETAL_MODEL_PATH = os.path.join(ROOT, 'ML', 'fetal', 'manual_gaussian_nb_with_purity.pkl')
MATERNAL_MODEL_PATH = os.path.join(ROOT, 'ML', 'mother', 'boosted_tree_model.pkl')

# Define maternal feature order expected by the model
MATERNAL_FEATURES = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']

def load_pickle(path):
    class SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Map known custom classes to their modules so unpickling succeeds
            if name in ('KLDecisionTree', 'TsallisDecisionTree'):
                try:
                    import ML.mother.custom_trees as _mt2
                    return getattr(_mt2, name)
                except Exception:
                    pass
            if name == 'ManualGaussianNB':
                try:
                    import ML.fetal.gaussian_naive_final as _gf2
                    return getattr(_gf2, name)
                except Exception:
                    pass
            return super().find_class(module, name)

    with open(path, 'rb') as f:
        return SafeUnpickler(f).load()


def normalize_input(data, feature_order):
    """
    Accepts either a list/tuple of values (in correct order) or a dict mapping feature names -> values.
    Returns a 2D list [[v1, v2, ...]] ready for model.predict.
    """
    if isinstance(data, (list, tuple)):
        return [list(map(float, data))]
    if isinstance(data, dict):
        # try to match keys case-insensitively
        lowered = {k.lower(): v for k, v in data.items()}
        row = []
        for feat in feature_order:
            v = None
            # prefer exact key
            if feat in data:
                v = data[feat]
            else:
                v = lowered.get(feat.lower())
            if v is None:
                raise ValueError(f"Missing feature '{feat}' in input data")
            row.append(float(v))
        return [row]
    raise ValueError('Input data must be a list or an object/dict')


@APP.route('/predict', methods=['POST'])
def predict():
    body = request.get_json(force=True)
    model_type = body.get('model')
    data = body.get('data')

    if model_type not in ('fetal', 'maternal'):
        return jsonify({'error': 'model must be "fetal" or "maternal"'}), 400

    try:
        if model_type == 'maternal':
            model = load_pickle(MATERNAL_MODEL_PATH)
            X = normalize_input(data, MATERNAL_FEATURES)
            # Some saved maternal artifacts are dicts containing an 'ensemble' of
            # (tree, weight) tuples plus a label_encoder. Handle that format here.
            try:
                import numpy as _np
                if isinstance(model, dict) and 'ensemble' in model:
                    ens = model.get('ensemble', [])
                    n_classes = int(model.get('n_classes', 3))
                    votes = _np.zeros((len(X), n_classes), dtype=float)
                    for tree, weight in ens:
                        preds = tree.predict(X)
                        for i, p in enumerate(preds):
                            votes[i, int(p)] += float(weight)
                    final = votes.argmax(axis=1)
                    le = model.get('label_encoder')
                    if le is not None:
                        try:
                            labels = le.inverse_transform(final)
                        except Exception:
                            labels = [str(int(v)) for v in final]
                    else:
                        labels = [str(int(v)) for v in final]
                    return jsonify({ 'riskLabel': labels[0], 'riskLevel': int(final[0]) })
                else:
                    pred = model.predict(X)
                    label = pred[0]
                    return jsonify({ 'riskLabel': label, 'riskLevel': label })
            except Exception as _e:
                return jsonify({'error': 'Prediction failed', 'details': str(_e)}), 500

        # fetal
        # fetal model was trained with many features; try to infer order from CSV header
        fetal_csv = os.path.join(ROOT, 'ML', 'fetal', 'fetal_health.csv')
        if not os.path.exists(fetal_csv):
            return jsonify({'error': 'Fetal CSV not found; cannot determine feature order.'}), 500
        # read header line
        with open(fetal_csv, 'r', encoding='utf-8') as fh:
            header = fh.readline().strip()
        cols = [c.strip() for c in header.split(',')]
        # remove target column if present
        if 'fetal_health' in cols:
            feature_order = [c for c in cols if c != 'fetal_health']
        else:
            feature_order = cols

        model = load_pickle(FETAL_MODEL_PATH)
        X = normalize_input(data, feature_order)
        
        # Fetal model is stored as a dict with 'model' key
        if isinstance(model, dict) and 'model' in model:
            actual_model = model['model']
            pred = actual_model.predict(X)
        else:
            pred = model.predict(X)
            
        # fetal labels are numeric (1/2/3). Return both numeric and mapped text
        label = pred[0]
        label_map = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
        text = label_map.get(int(label), str(label))
        return jsonify({ 'riskLabel': text, 'riskLevel': int(label) })

    except Exception as e:
        return jsonify({ 'error': 'Prediction failed', 'details': str(e) }), 500


if __name__ == '__main__':
    # Run on port from environment or default 5001
    port = int(os.environ.get('PORT', 5001))
    APP.run(host='0.0.0.0', port=port, debug=False)
