"""Numerical and causal contracts for the hand-written decoder."""
import math
import unittest

import torch


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_rmsnorm_and_split_half_rotation(self):
        from qwen3_translation.model import RMSNorm, RotaryEmbedding

        norm = RMSNorm(2, eps=0)
        torch.testing.assert_close(norm(torch.tensor([[3., 4.]])),
                                   torch.tensor([[0.84852814, 1.13137085]]))
        self.assertEqual(norm(torch.ones(1, 2, dtype=torch.bfloat16)).dtype, torch.bfloat16)
        q = torch.tensor([[[[1., 2., 3., 4.], [1., 2., 3., 4.]]]])
        rotated, _ = RotaryEmbedding(4, theta=1)(q, q, torch.tensor([[0, 1]]))
        torch.testing.assert_close(rotated[:, :, 0], q[:, :, 0])
        expected = torch.tensor([math.cos(1)-3*math.sin(1), 2*math.cos(1)-4*math.sin(1),
                                 3*math.cos(1)+math.sin(1), 4*math.cos(1)+2*math.sin(1)])
        torch.testing.assert_close(rotated[0, 0, 1], expected)
        torch.testing.assert_close(rotated.square().sum(-1), q.square().sum(-1))

    def test_consecutive_kv_grouping_and_causal_average(self):
        from qwen3_translation.model import GroupedQueryAttention, QwenDenseConfig

        cfg = QwenDenseConfig(vocab_size=19, hidden_size=4, intermediate_size=8,
                              num_hidden_layers=1, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=2)
        attn = GroupedQueryAttention(cfg)
        with torch.no_grad():
            attn.q_proj.weight.zero_()
            attn.k_proj.weight.zero_()
            attn.v_proj.weight.copy_(torch.eye(4))
            attn.o_proj.weight.zero_()
            for row in range(4):
                attn.o_proj.weight[row, 2 * row] = 1
        values = torch.tensor([[[1., 2., 3., 4.], [5., 6., 7., 8.]]])
        actual = attn(values, torch.ones(1, 2), torch.tensor([[0, 1]]))
        torch.testing.assert_close(actual, torch.tensor([[[1., 1., 3., 3.], [3., 3., 5., 5.]]]))

    def test_future_and_masked_keys_cannot_change_earlier_predictions(self):
        from qwen3_translation.model import QwenDenseConfig, QwenDenseForCausalLM

        # Attention width is 16 while hidden width is 12, deliberately unequal.
        cfg = QwenDenseConfig(vocab_size=19, hidden_size=12, intermediate_size=24,
                              num_hidden_layers=2, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=4, max_position_embeddings=16)
        torch.manual_seed(19)
        model = QwenDenseForCausalLM(cfg).eval()
        ids = torch.tensor([[2, 4, 6, 8, 10]])
        changed = ids.clone()
        changed[:, 3:] = torch.tensor([11, 12])
        torch.testing.assert_close(model(ids)[:, :3], model(changed)[:, :3])
        mask = torch.tensor([[1, 0, 1, 1, 1]])
        changed = ids.clone()
        changed[0, 1] = 18
        torch.testing.assert_close(model(ids, mask)[:, 2:], model(changed, mask)[:, 2:])
        self.assertIs(model.lm_head.weight, model.embed_tokens.weight)
        self.assertIsNot(model.layers[0].attention.q_proj.weight, model.layers[1].attention.q_proj.weight)
        with self.assertRaises(ValueError):
            model(torch.ones(1, 17, dtype=torch.long))

    def test_decoder_can_learn_a_tiny_continuation(self):
        from qwen3_translation.model import QwenDenseConfig, QwenDenseForCausalLM

        torch.manual_seed(19)
        cfg = QwenDenseConfig(vocab_size=19, hidden_size=12, intermediate_size=24,
                              num_hidden_layers=2, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=4)
        model = QwenDenseForCausalLM(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        ids = torch.tensor([[2, 4, 6, 8, 10]])
        labels = torch.tensor([[-100, -100, 8, 10, 3]])
        losses = []
        for _ in range(60):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(ids).reshape(-1, 19), labels.reshape(-1))
            losses.append(float(loss.detach()))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            optimizer.step()
        self.assertLess(losses[-1], 0.1)
        self.assertLess(losses[-1], losses[0] / 10)


if __name__ == "__main__":
    unittest.main()
