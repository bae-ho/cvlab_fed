import unittest

import numpy as np

from scripts.localization_demo import (binary_to_segments, frame_iou, scores_to_segments,
                                       segments_to_mask, smooth, video_name_from_path)


class LocalizationDemoTests(unittest.TestCase):
    def test_binary_to_segments(self):
        self.assertEqual(binary_to_segments([0, 1, 1, 0, 1]), [(1, 3), (4, 5)])
        self.assertEqual(binary_to_segments([]), [])

    def test_scores_to_segments_merge_and_min_len(self):
        s = [0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9]
        self.assertEqual(scores_to_segments(s, 0.5, min_len=1, merge_gap=0), [(0, 2), (3, 5), (8, 9)])
        self.assertEqual(scores_to_segments(s, 0.5, min_len=2, merge_gap=1), [(0, 5)])

    def test_iou(self):
        gt = segments_to_mask([(2, 6)], 10)
        pred = segments_to_mask([(4, 8)], 10)
        self.assertAlmostEqual(frame_iou(pred, gt), 2 / 6)
        self.assertIsNone(frame_iou(np.zeros(5, bool), np.zeros(5, bool)))

    def test_smooth_keeps_length(self):
        x = np.random.rand(37)
        for w in (1, 2, 5):
            self.assertEqual(len(smooth(x, w)), 37)

    def test_video_name(self):
        self.assertEqual(video_name_from_path('/a/b/Arson011_x264__0.npy'), 'Arson011_x264')
        self.assertEqual(video_name_from_path('Fight_01.npy'), 'Fight_01')


if __name__ == '__main__':
    unittest.main()
