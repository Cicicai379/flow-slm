import unittest

import torch

from blockwise.blocks import pack_blocks, shift_blocks_right


class PackBlocksTest(unittest.TestCase):
    def test_partial_blocks_keep_frame_level_mask(self):
        frames = torch.arange(2 * 10 * 3).reshape(2, 10, 3).float()
        packed = pack_blocks(frames, torch.tensor([10, 5]), block_size=4)

        self.assertEqual(tuple(packed.values.shape), (2, 3, 4, 3))
        self.assertEqual(packed.frame_mask.sum(dim=(1, 2)).tolist(), [10, 5])
        self.assertEqual(packed.block_mask.tolist(), [[True, True, True], [True, True, False]])
        self.assertFalse(packed.frame_mask[0, 2, 2:].any())
        self.assertFalse(packed.frame_mask[1, 1, 1:].any())

    def test_lengths_are_validated(self):
        with self.assertRaises(ValueError):
            pack_blocks(torch.zeros(1, 3, 2), torch.tensor([4]), block_size=2)

    def test_shift_has_no_same_block_target_leakage(self):
        targets = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        inputs = shift_blocks_right(targets, torch.tensor([-1.0, -2.0]))

        self.assertTrue(torch.equal(inputs[:, 0], torch.tensor([[-1.0, -2.0]])))
        self.assertTrue(torch.equal(inputs[:, 1:], targets[:, :-1]))
        self.assertFalse(torch.equal(inputs, targets))


if __name__ == "__main__":
    unittest.main()
