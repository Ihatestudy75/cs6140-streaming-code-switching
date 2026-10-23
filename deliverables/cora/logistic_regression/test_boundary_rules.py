import unittest
from boundary_rules import boundary_kind, boundary_group

class BoundaryTests(unittest.TestCase):
    def test_multiple_scripts_and_punctuation(self):
        for text,kind in [('word,','clause_punctuation'),('好。','sentence_punctuation'),
                          ('مرحبا؟','sentence_punctuation'),('word）','other_punctuation'),
                          ('好','non_punctuation'),('word','non_punctuation')]:
            self.assertEqual(boundary_kind(text,[[0,len(text)]],0),kind)

    def test_future_text_is_not_used(self):
        self.assertEqual(boundary_kind('好，hello',[[0,1],[1,2],[2,7]],1),
                         boundary_kind('好，different words',[[0,1],[1,2],[2,17]],1))
        self.assertEqual(boundary_kind('word,next',[[0,4],[4,5],[5,9]],0),'non_punctuation')

    def test_whitespace_and_categories(self):
        self.assertEqual(boundary_kind('word,  ',[[0,7]],0),'clause_punctuation')
        self.assertEqual(boundary_group('sentence_punctuation'),'punctuation')
        self.assertEqual(boundary_group('non_punctuation'),'within_clause_proxy')
        self.assertEqual(boundary_kind('',[[0,0]],0),'unclassified')

if __name__=='__main__':unittest.main()
