import unittest
from pathlib import Path
import numpy as np
from metrics import choose_threshold, evaluate
from model import token_features, at_position, vectorize_prefixes, predict_prefixes


def sample(ids, y, d=-1, sid='a', pair='p'):
    return {'inputs':{'input_ids':ids},'targets':{'switch':y,'duration':d},
            'provenance':{'sequence_id':sid,'language_pair':pair,'position':len(ids)-1}}


class StreamingTests(unittest.TestCase):
    def test_future_changes_cannot_change_features(self):
        a=[11,12,13,14,15];b=[11,12,999,998,997]
        self.assertEqual(at_position(a,1,20),at_position(b,1,20))
        self.assertNotEqual(at_position(a,2,20),at_position(b,2,20))

    def test_window_keeps_recent_context_and_token_order(self):
        self.assertEqual(token_features([99,1,2],2),token_features([1,2],2))
        self.assertNotEqual(token_features([1,2],2),token_features([2,1],2))

    def test_targets_and_provenance_are_not_inputs(self):
        a=sample([11,12],1,2)
        before=vectorize_prefixes([a['inputs']['input_ids']],5)
        a['targets']={'switch':0,'duration':0}
        a['provenance']={'sequence_id':'different','language_pair':'other','position':999}
        after=vectorize_prefixes([a['inputs']['input_ids']],5)
        self.assertEqual((before!=after).nnz,0)

    def test_joint_metric_excludes_unknown_durations_only(self):
        ss=[sample([1],1,0),sample([2],1,-1),sample([3],0)]
        result=evaluate(ss,[(0.1,0),(0.8,2),(0.8,0)],0.5,{'a':'A'})['pooled']
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,1,1))
        self.assertEqual(result['duration_support'],1)
        self.assertEqual(result['joint_duration_macro_f1'],0)

    def test_threshold_search_matches_exhaustive(self):
        ss=[sample([1],1),sample([2],0),sample([3],1,sid='b'),sample([4],0,sid='b')]
        scores=[0.6,0.6,0.3,0.1];types={'a':'A','b':'B'}
        chosen=choose_threshold(ss,scores,types)
        expected=max(evaluate(ss,list(zip(scores,[0]*4)),t,types)['macro_pair_type_switch_f1']
                     for t in [0.7,0.6,0.3,0.1])
        self.assertAlmostEqual(chosen['validation_macro_pair_type_switch_f1'],expected)

    def test_serialized_bundle_reproduces_probabilities(self):
        import joblib,tempfile
        from sklearn.linear_model import LogisticRegression
        ids=[[1,2],[1,3],[4,2],[4,3],[5,2],[5,3]]
        x=vectorize_prefixes(ids,5)
        switch=LogisticRegression(solver='liblinear',random_state=6140).fit(x,[0,1,0,1,0,1])
        duration=LogisticRegression(solver='liblinear',random_state=6140).fit(x,[0,1,2,0,1,2])
        bundle={'switch_model':switch,'duration_model':duration,
                'switch_config':{'window':5},'duration_config':{'window':5}}
        before=predict_prefixes(bundle,ids)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.joblib';joblib.dump(bundle,path)
            after=predict_prefixes(joblib.load(path),ids)
        np.testing.assert_allclose(before[0],after[0],rtol=0,atol=0)
        np.testing.assert_array_equal(before[1],after[1])

if __name__=='__main__':
    unittest.main()
