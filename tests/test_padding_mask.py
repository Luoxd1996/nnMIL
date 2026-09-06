import unittest

import torch

from nnMIL.network_architecture.models.simple_mil import SimpleMIL
from nnMIL.utilities.masking import valid_mask_from_bag_sizes


class TestSimpleMILPaddingMask(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = SimpleMIL(input_dim=8, hidden_dim=4, pred_num=2, dropout=False).eval()
        self.features = torch.randn(1, 3, 8)
        self.padded = torch.cat([self.features, torch.zeros(1, 2, 8)], dim=1)
        self.mask = torch.tensor([[True, True, True, False, False]])

    def test_masked_padding_matches_unpadded_evaluation(self):
        raw_logits = self.model(self.features)["logits"]
        padded_logits = self.model(self.padded, valid_mask=self.mask)["logits"]
        self.assertTrue(torch.allclose(raw_logits, padded_logits, atol=1e-6))

    def test_unmasked_padding_changes_prediction(self):
        raw_logits = self.model(self.features)["logits"]
        padded_logits = self.model(self.padded)["logits"]
        self.assertFalse(torch.allclose(raw_logits, padded_logits, atol=1e-6))

    def test_training_masked_padding_matches_unpadded(self):
        self.model.train()
        torch.manual_seed(23)
        raw_logits = self.model(self.features)["logits"]
        torch.manual_seed(23)
        padded_logits = self.model(self.padded, valid_mask=self.mask)["logits"]
        self.assertTrue(torch.allclose(raw_logits, padded_logits, atol=1e-6))

    def test_mask_builder_uses_true_bag_lengths(self):
        mask = valid_mask_from_bag_sizes(torch.randn(2, 5, 8), torch.tensor([5, 3]))
        self.assertEqual(mask.tolist(), [[True, True, True, True, True], [True, True, True, False, False]])


if __name__ == "__main__":
    unittest.main()
