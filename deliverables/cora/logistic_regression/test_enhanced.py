import unittest
from model import vectorize_prefixes
from enhanced_features import describe, extra_features, vectorize

class EnhancedTests(unittest.TestCase):
    vocab=['hello','##ing',',','世界','12','Bonjour','。','привет']
    def test_causal_window(self):
        a=[0,1,2,3,4,5];b=[0,1,2,7,7,7]
        for mode in ('morph','script','all'):
            self.assertEqual(extra_features(a[:3],3,self.vocab,mode),extra_features(b[:3],3,self.vocab,mode))
            self.assertEqual(extra_features(a,3,self.vocab,mode),extra_features([7]*3+a[-3:],3,self.vocab,mode))
    def test_original_block(self):
        prefixes=[[0,1,2],[3,6]];original=vectorize_prefixes(prefixes,5)
        for mode in ('base','morph','script','all'):
            full=vectorize(prefixes,5,self.vocab,mode)
            self.assertEqual((original!=full[:,:original.shape[1]]).nnz,0)
    def test_descriptors(self):
        self.assertTrue(describe('##ing')['continuation'])
        self.assertTrue(describe('。')['punct'])
        self.assertTrue(describe('12')['digit'])
        self.assertEqual(describe('世界')['script'],'HAN')
        self.assertEqual(describe('привет')['script'],'CYRILLIC')
    def test_ablation(self):
        args=([0,2,3,4],5,self.vocab)
        both=extra_features(*args,'all')
        self.assertIn('morph:punct_distance=2',both)
        self.assertEqual(both,dict(extra_features(*args,'morph'),**extra_features(*args,'script')))

if __name__=='__main__':unittest.main()
