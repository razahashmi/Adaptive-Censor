"""Mask candidate selection must preserve prompt ownership and image geometry."""
import unittest
import numpy as np
from mask_prompt_experiment import combine,envelope


class PromptTests(unittest.TestCase):
    def test_selects_quality_per_prompt_and_never_fills_the_detector_box(self):
        boxes=[[1,1,3,3],[7,2,9,4]]
        masks=np.zeros((2,2,6,10),dtype=bool)
        masks[0,1,2,2]=True;masks[1,0,3,8]=True
        masks[0,0]=True;masks[1,1]=True
        result=combine(masks,[[.1,.9],[.9,.1]],boxes,(6,10))
        self.assertEqual(int(result.sum()),2)
        self.assertTrue(result[2,2] and result[3,8])

    def test_each_outline_is_clipped_to_its_own_box_on_a_non_square_image(self):
        boxes=[[0,0,2,3],[8,1,10,4]]
        masks=np.zeros((2,1,4,10),dtype=bool)
        masks[0,0,2,9]=True;masks[1,0,1,1]=True
        self.assertFalse(combine(masks,[[1],[1]],boxes,(4,10)).any())
        self.assertEqual(envelope((4,10),[0,0,10,4]).shape,(4,10))

    def test_rejects_misaligned_or_invalid_quality_outputs(self):
        masks=np.zeros((1,2,4,10),dtype=bool)
        for scores in ([[1]],[[float('nan'),1]]):
            with self.assertRaises(ValueError):combine(masks,scores,[[0,0,2,2]],(4,10))
        with self.assertRaises(ValueError):combine(masks,[[1,1]],[],(4,10))


if __name__=='__main__':unittest.main()
