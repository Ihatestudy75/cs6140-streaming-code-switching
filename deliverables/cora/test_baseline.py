import unittest
from run import Counts, choose_threshold, evaluate, score_group


def sample(ids, y, d=-1, sid='a', pair='p'):
    return {'inputs': {'input_ids': ids}, 'targets': {'switch': y, 'duration': d},
            'provenance': {'sequence_id': sid, 'language_pair': pair, 'position': len(ids)-1}}


class BaselineTests(unittest.TestCase):
    def test_current_token_and_unseen_fallback(self):
        m = Counts([sample([1], 1, 2), sample([2], 0)])
        self.assertEqual(m.predict([9, 1]), (1.0, 2))
        self.assertEqual(m.predict([99]), (0.5, 2))

    def test_prefix_changes_only_with_past_and_backs_off(self):
        m = Counts([sample([1, 2], 1, 0), sample([3, 2], 0)], order=2)
        self.assertEqual(m.predict([1, 2])[0], 1.0)
        self.assertEqual(m.predict([3, 2])[0], 0.0)
        self.assertEqual(m.predict([9, 2])[0], 0.5)
        m.min_count = 2
        self.assertEqual(m.predict([1, 2])[0], 0.5)

    def test_validation_sweep_matches_exhaustive_macro(self):
        ss = [sample([1], 1), sample([2], 0), sample([3], 1, sid='b'), sample([4], 0, sid='b')]
        scores = [0.6, 0.6, 0.3, 0.1]
        types = {'a': 'A', 'b': 'B'}
        chosen = choose_threshold(ss, scores, types)
        pred = [(x, 0) for x in scores]
        exhaustive = max(evaluate(ss, pred, t, types)['macro_pair_type_switch_f1'] for t in [0.7, 0.6, 0.3, 0.1])
        self.assertAlmostEqual(chosen['validation_macro_pair_type_switch_f1'], exhaustive)

    def test_duration_masks_and_joint_penalties(self):
        ss = [sample([1], 1, 0), sample([2], 1, -1), sample([3], 0)]
        # Unknown duration switch excluded only from duration, not switch score.
        result = score_group([(ss[0], 0.1, 0), (ss[1], 0.8, 2), (ss[2], 0.8, 0)], 0.5)
        self.assertEqual(result['duration_support'], 1)
        self.assertEqual(result['duration_accuracy_oracle_switch'], 1.0)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 1))
        self.assertEqual(result['joint_duration_macro_f1'], 0.0)

    def test_predict_does_not_use_metadata_or_target(self):
        ss = [sample([1], 1, 1), sample([2], 0)]
        m = Counts(ss, 2)
        before = m.predict(ss[0]['inputs']['input_ids'])
        ss[0]['targets'] = {'switch': 0, 'duration': 2}
        ss[0]['provenance'] = {'sequence_id': 'changed'}
        self.assertEqual(m.predict(ss[0]['inputs']['input_ids']), before)


if __name__ == '__main__':
    unittest.main()
