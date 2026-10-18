import unittest
import numpy as np
from nudenet.nudenet import _postprocess
from anatomy_runtime import postprocess


class AnatomyRuntimeTests(unittest.TestCase):
    def compare(self,rows,width=640,height=359,size=640):
        output=[np.asarray(rows,dtype=np.float32).T[None]]
        args=(output,0,width-height,1,width/height,width,height,size,size)
        self.assertEqual(postprocess(*args),_postprocess(*args))

    def test_vectorized_result_matches_reference_on_clipping_and_cross_class_nms(self):
        rows=np.zeros((6,22),dtype=np.float32)
        rows[:,:4]=[[-10,10,40,50],[639,358,60,50],[200,150,50,40],
                    [200,150,50,40],[300,250,20,20],[500,300,20,20]]
        for i,(category,score) in enumerate([(3,.9),(2,.7),(3,.8),(5,.85),(3,.25),(3,.2)]):
            rows[i,4+category]=score
        self.compare(rows)

    def test_equivalence_on_dense_anchor_outputs_and_ties(self):
        rng=np.random.default_rng(13)
        for width,height,size in [(640,359,640),(320,320,320),(640,266,640)]:
            rows=np.zeros((8400,22),dtype=np.float32)
            rows[:,:2]=rng.uniform(0,size,(8400,2));rows[:,2:4]=rng.uniform(2,200,(8400,2))
            rows[:,4:]=rng.uniform(0,.18,(8400,18))
            for i in range(0,8400,113):rows[i,4+i%18]=rng.choice([.25,.55,.9])
            self.compare(rows,width,height,size)

    def test_empty_detection_output_remains_empty(self):
        self.compare(np.zeros((100,22),dtype=np.float32))


if __name__=='__main__':unittest.main()
