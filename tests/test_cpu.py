"""CPU-only wrapper tests; model and network boundaries are mocked.

Run directly with `python -m unittest discover -s tests -v`.
"""
import builtins
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


# Import the real Evo2 Python wrappers without importing Vortex's CUDA kernels.
# These are boundaries, not alternate implementations of the code under test.
class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def import_wrappers():
    modules = {
        name: types.ModuleType(name)
        for name in (
            'vortex', 'vortex.model', 'vortex.model.model',
            'vortex.model.generation', 'vortex.model.tokenizer', 'vortex.model.utils',
        )
    }
    modules['vortex.model.model'].StripedHyena = mock.Mock()
    modules['vortex.model.generation'].generate = mock.Mock()
    modules['vortex.model.tokenizer'].CharLevelTokenizer = mock.Mock()
    utils = modules['vortex.model.utils']
    utils.dotdict = DotDict
    utils.print_rank_0 = mock.Mock()
    utils.load_checkpoint = mock.Mock()
    with mock.patch.dict(sys.modules, {**modules, 'transformer_engine': None}):
        loaded = importlib.import_module('evo2.models')
        package_modules = {k: v for k, v in sys.modules.items() if k == 'evo2' or k.startswith('evo2.')}
    # patch.dict restores sys.modules; keep the imported real package available to
    # pkgutil's resource lookup, without keeping the fake dependency modules.
    sys.modules.update(package_modules)
    return loaded


models = import_wrappers()


class CheckpointMergeTests(unittest.TestCase):
    def test_failed_merge_is_not_reused_on_retry(self):
        """An I/O failure after part0 must not publish a reusable checkpoint."""
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            filename = 'evo2_7b.pt'
            part0 = snapshot / (filename + '.part0')
            part1 = snapshot / (filename + '.part1')
            part0.write_bytes(b'first shard\x00')
            part1.write_bytes(b'second shard\xff')
            final = root / filename
            real_open = builtins.open

            def fail_second_part(path, *args, **kwargs):
                if Path(path) == part1:
                    raise OSError('simulated read failure on part1')
                return real_open(path, *args, **kwargs)

            # No requests, model construction, checkpoint reads, or shard deletion.
            with (
                mock.patch.object(models.constants, 'HF_HUB_CACHE', str(root / 'hub')),
                mock.patch.object(models, 'snapshot_download', return_value=str(snapshot)) as download,
                mock.patch.object(models, 'hf_hub_download') as tracking,
                mock.patch.object(models, 'StripedHyena'),
                mock.patch.object(models, 'load_checkpoint') as load,
            ):
                loader = models.Evo2.__new__(models.Evo2)
                kwargs = dict(model_name='evo2_7b', config_path='configs/evo2-7b-1m.yml', remove_shards=False)
                with mock.patch('builtins.open', side_effect=fail_second_part):
                    with self.assertRaisesRegex(OSError, 'simulated read failure'):
                        loader.load_evo2_model(**kwargs)
                load.assert_not_called()
                # Retry the same documented load with the filesystem healthy again.
                loader.load_evo2_model(**kwargs)
                self.assertEqual(final.read_bytes(), part0.read_bytes() + part1.read_bytes())
                self.assertEqual(download.call_count, 2)
                tracking.assert_not_called()
                load.assert_called_once()
                self.assertEqual(set(root.iterdir()), {snapshot, final})

    def test_complete_merge_keeps_existing_shard_order(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            chunks = [b'part0\x00', b'part1\xff', b'part2\n']
            for i, chunk in enumerate(chunks):
                (snapshot / f'evo2_7b.pt.part{i}').write_bytes(chunk)
            with (
                mock.patch.object(models.constants, 'HF_HUB_CACHE', str(root / 'hub')),
                mock.patch.object(models, 'snapshot_download', return_value=str(snapshot)),
                mock.patch.object(models, 'StripedHyena'),
                mock.patch.object(models, 'load_checkpoint') as load,
            ):
                models.Evo2.__new__(models.Evo2).load_evo2_model(
                    model_name='evo2_7b', config_path='configs/evo2-7b-1m.yml', remove_shards=False,
                )
                self.assertEqual((root / 'evo2_7b.pt').read_bytes(), b''.join(chunks))
                self.assertEqual(load.call_args.args[1], str(root / 'evo2_7b.pt'))
                self.assertEqual(set(root.iterdir()), {snapshot, root / 'evo2_7b.pt'})


if __name__ == '__main__':
    unittest.main()
