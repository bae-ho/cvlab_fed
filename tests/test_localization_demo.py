import unittest

import numpy as np

from scripts.localization_demo import (best_f1_threshold, binary_to_segments, normalize_video, frame_iou, scores_to_segments,
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

    def test_best_f1_threshold_separates_low_scores(self):
        # 이상 구간 점수가 0.3 근처라 0.5로는 못 잡는 경우에도 적절한 기준을 찾아야 함
        scores = np.r_[np.full(80, 0.05), np.full(20, 0.3)]
        gt = np.r_[np.zeros(80, bool), np.ones(20, bool)]
        t, f1 = best_f1_threshold(scores, gt)
        self.assertTrue(0.05 < t <= 0.3)
        self.assertAlmostEqual(f1, 1.0)

    def test_normalize_video(self):
        n = normalize_video([0.01, 0.03, 0.02])
        self.assertAlmostEqual(n.min(), 0.0)
        self.assertAlmostEqual(n.max(), 1.0)
        self.assertTrue(np.all(normalize_video([0.2, 0.2]) == 0))


if __name__ == '__main__':
    unittest.main()
