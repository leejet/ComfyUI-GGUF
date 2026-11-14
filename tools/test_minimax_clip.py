"""Run with the host ComfyUI Python: python tools/test_minimax_clip.py."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
import comfy.sd
import comfy.utils

spec = importlib.util.spec_from_file_location(
    '_gguf_regression', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
Loader = package.NODE_CLASS_MAPPINGS['CLIPLoaderGGUF']


class BackendReached(Exception):
    pass


class MiniMaxClipTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            'model.visual.deepstack_merger_list.0.norm.weight': torch.empty(1280, device='meta'),
            'model.visual.merger.linear_fc2.weight': torch.empty((5120, 1280), device='meta'),
            'model.layers.49.self_attn.q_proj.weight': torch.empty((8192, 5120), device='meta'),
        }

    def capture(self, clip_type, state):
        captured = []

        def backend(**kwargs):
            captured.extend(kwargs['state_dicts'])
            raise BackendReached

        with mock.patch.object(comfy.sd, 'load_text_encoder_state_dicts', side_effect=backend):
            with self.assertRaises(BackendReached):
                Loader().load_patcher([], clip_type, [state])
        return captured[0]

    def test_minimax_32b_is_not_detected_as_8b(self):
        state = self.capture(comfy.sd.CLIPType.MINIMAX, self.state)
        self.assertEqual(comfy.sd.detect_te_model(state), comfy.sd.TEModel.QWEN3VL_32B)
        self.assertNotIn('model.visual.deepstack_merger_list.0.norm.weight', state)

    def test_other_clip_type_keeps_its_visual_namespace(self):
        state = self.capture(comfy.sd.CLIPType.QWEN_IMAGE, self.state)
        self.assertIn('model.visual.deepstack_merger_list.0.norm.weight', state)

    def test_short_minimax_encoder_is_not_rewritten(self):
        del self.state['model.layers.49.self_attn.q_proj.weight']
        state = self.capture(comfy.sd.CLIPType.MINIMAX, self.state)
        self.assertIn('model.visual.deepstack_merger_list.0.norm.weight', state)

    def test_already_canonical_minimax_encoder_stays_valid(self):
        state = comfy.utils.state_dict_prefix_replace(self.state, {'model.visual.': 'visual.'})
        state = self.capture(comfy.sd.CLIPType.MINIMAX, state)
        self.assertEqual(comfy.sd.detect_te_model(state), comfy.sd.TEModel.QWEN3VL_32B)


if __name__ == '__main__':
    unittest.main()
