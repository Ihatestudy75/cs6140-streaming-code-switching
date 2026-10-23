"""Predict from already-tokenized, observed prefix IDs only."""
import argparse
import json
from pathlib import Path
import joblib
from model import predict_prefixes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path(__file__).resolve().parent/'runs/logistic-v1/logistic_model.joblib')
    parser.add_argument('--prefix-ids',required=True,help='JSON list of observed token IDs, including the current token.')
    args=parser.parse_args()
    ids=json.loads(args.prefix_ids)
    if not isinstance(ids,list) or not ids or any(type(x) is not int or x < 0 for x in ids):
        parser.error('prefix-ids must be a nonempty JSON list of nonnegative integers.')
    bundle=joblib.load(args.model)
    scores,durations=predict_prefixes(bundle,[ids])
    decision=bool(scores[0]>=bundle['switch_config']['threshold'])
    print(json.dumps({'switch_probability':float(scores[0]),'predicted_switch':decision,
                      'predicted_duration_if_switch':bundle['duration_classes'][int(durations[0])] if decision else None},indent=2))
