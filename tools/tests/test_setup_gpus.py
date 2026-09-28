"""CPU-only setup/serving GPU-selection regressions. Run with unittest discovery in tools/tests/."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_module("strata_setup_gpu_tests", ROOT / "setup.py")
# These tests exercise process launch/environment plumbing, not templates. Stub the frontend so Jinja is
# not required on a clean Python installation (no downloaded packages, model files or CUDA needed).
frontend = types.ModuleType("serve.frontend")
for symbol in ("ChatTemplate", "Event", "OutputParser", "anthropic_to_messages", "images_of", "openai_to_messages"):
    setattr(frontend, symbol, Mock())
with patch.dict(sys.modules, {"serve.frontend": frontend}):
    server = load_module("strata_server_gpu_tests", ROOT / "serve" / "server.py")

U0 = "GPU-11111111-1111-1111-1111-111111111111"
U1 = "GPU-22222222-2222-2222-2222-222222222222"
U2 = "GPU-33333333-3333-3333-3333-333333333333"
SMI = (f"0, {U0}, NVIDIA Test A, 24576, 8.6, 580.88\n"
       f"1, {U1}, NVIDIA Test B, 32768, 12.0, 580.88\n"
       f"2, {U2}, NVIDIA Test C, 16384, 8.6, 580.88\n")


def gpu(index, arch="86", vram=24):
    return {"index": index, "device": index, "uuid": (U0, U1, U2)[index], "name": f"NVIDIA Test {index}",
            "arch": arch, "vram_gb": vram, "driver": "580.88"}


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))

    def detect(self, uuids, mask=None):
        if mask is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = mask
        with patch.object(setup, "out", return_value=SMI), patch.object(setup, "cuda_visible_uuids", return_value=uuids):
            return setup.gpu_infos()

    def test_smi_queries_all_devices_with_stable_ids_and_legacy_fields(self):
        with patch.object(setup.subprocess, "run", return_value=Mock(stdout=SMI)) as run:
            found = setup.physical_gpu_infos()
        self.assertEqual(len(found), 3)
        self.assertEqual(found[1], {k: v for k, v in gpu(1, "120", 32).items() if k != "device"} |
                         {"name": "NVIDIA Test B"})
        self.assertIn("--query-gpu=index,uuid,name,memory.total,compute_cap,driver_version", run.call_args.args[0])

    def test_csv_quoted_gpu_name(self):
        with patch.object(setup, "out", return_value=SMI.replace("NVIDIA Test A", '"NVIDIA, Test A"')):
            self.assertEqual(setup.physical_gpu_infos()[0]["name"], "NVIDIA, Test A")

    def test_smi_unavailable(self):
        with patch.object(setup, "out", return_value=""), patch.object(setup, "cuda_visible_uuids") as probe:
            self.assertEqual(setup.gpu_infos(), [])
            self.assertIsNone(setup.gpu_info())
        probe.assert_not_called()

    def test_reject_malformed_or_duplicate_smi_information(self):
        bad = ["not a GPU", SMI.replace(U0, "GPU-unknown"), SMI.replace("24576", "nan"),
               SMI.replace("24576", "-1"), SMI.replace("8.6", "N/A"), SMI.replace("580.88", "N/A"),
               SMI + SMI.splitlines()[0], SMI.replace(U1, U0), SMI.replace("1, GPU-", "0, GPU-")]
        for text in bad:
            with self.subTest(text=text), patch.object(setup, "out", return_value=text), self.assertRaises(ValueError):
                setup.physical_gpu_infos()

    def test_default_order_comes_from_cuda_not_smi(self):
        found = self.detect([U2, U0, U1])
        self.assertEqual([g["index"] for g in found], [2, 0, 1])
        self.assertEqual([g["device"] for g in found], [0, 1, 2])
        self.assertEqual(setup.select_gpus(found, "1,0")[0]["uuid"], U0)
        self.assertEqual(setup.select_gpus(found, None), [found[0]])

    def test_numeric_mask_is_interpreted_by_cuda_not_as_smi_indices(self):
        found = self.detect([U1, U2], "2,0")
        self.assertEqual([g["index"] for g in found], [1, 2])
        self.assertEqual(setup.select_gpus(found, "1,0"), [found[1], found[0]])
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,0")

    def test_full_and_abbreviated_uuid_masks(self):
        for mask in (f"{U1},{U0}", "GPU-2222,GPU-1111", f"{U1},0"):
            with self.subTest(mask=mask):
                self.assertEqual([g["uuid"] for g in self.detect([U1, U0], mask)], [U1, U0])

    def test_empty_mask_hides_every_gpu_without_probing(self):
        for mask in ("", "-1"):
            with self.subTest(mask=mask), patch.object(setup, "out", return_value=SMI), \
                    patch.object(setup, "cuda_visible_uuids") as probe:
                os.environ["CUDA_VISIBLE_DEVICES"] = mask
                self.assertEqual(setup.gpu_infos(), [])
                probe.assert_not_called()

    def test_invalid_ambiguous_mig_and_truncated_masks_are_rejected(self):
        for mask in ("0,", "0,-1", "0,nope", "GPU-", "GPU-deadbeef", "MIG-123", " 0", "0,0", "9"):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.detect([], mask)
        ambiguous = SMI.replace(U1, U0[:-1] + "2")
        os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-1111"
        with patch.object(setup, "out", return_value=ambiguous), self.assertRaisesRegex(ValueError, "ambiguous"):
            setup.gpu_infos()

    def test_uuid_mapping_must_match_driver(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.detect([U0, U1], f"{U1},{U0}")
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        with self.assertRaisesRegex(ValueError, "cannot be matched"):
            self.detect(["GPU-44444444-4444-4444-4444-444444444444"])

    def test_legacy_gpu_info_returns_first_visible(self):
        with patch.object(setup, "gpu_infos", return_value=[gpu(2)]):
            self.assertEqual(setup.gpu_info(), gpu(2))

    def test_runtime_probe_uses_fresh_inheriting_subprocess(self):
        os.environ.update(CUDA_VISIBLE_DEVICES=f"{U2},{U0}", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        with patch.object(setup.subprocess, "run", return_value=Mock(stdout=json.dumps([U2, U0]))) as run:
            self.assertEqual(setup.cuda_visible_uuids(), [U2, U0])
        self.assertEqual(run.call_args.args[0][:2], [sys.executable, "-c"])
        self.assertNotIn("env", run.call_args.kwargs)  # full inherited environment, not a replaced mask
        self.assertTrue(run.call_args.kwargs["check"])
        compile(run.call_args.args[0][2], "cuda-probe", "exec")

    def test_probe_failures_never_fall_back_to_guessed_indices(self):
        for response in ("", "{}", '"GPU"', '[null]', json.dumps([U0, U0])):
            with self.subTest(response=response), patch.object(setup.subprocess, "run", return_value=Mock(stdout=response)), \
                    self.assertRaisesRegex(ValueError, "refusing to guess"):
                setup.cuda_visible_uuids()
        for error in (OSError("no driver"), subprocess.TimeoutExpired("probe", 60),
                      subprocess.CalledProcessError(1, "probe")):
            with self.subTest(error=error), patch.object(setup.subprocess, "run", side_effect=error), \
                    self.assertRaises(ValueError):
                setup.cuda_visible_uuids()


class ArgumentAndContextTests(unittest.TestCase):
    def test_default_gpu_options_are_not_forwarded_implicitly(self):
        args = setup.argument_parser().parse_args([])
        self.assertIsNone(args.devices)
        self.assertIsNone(args.expert_vram_mib)
        self.assertIsNone(args.expert_reserve_mib)
        self.assertFalse(args.allow_high_context)

    def test_order_and_zero_cache_limits_are_preserved(self):
        args = setup.argument_parser().parse_args(["--devices", " 2, 00,1 ", "--expert-vram-mib", "0",
                                                  "--expert-reserve-mib", "1024"])
        self.assertEqual(args.devices, "2,0,1")
        self.assertEqual(args.expert_vram_mib, 0)
        self.assertEqual(args.expert_reserve_mib, 1024)

    def test_invalid_device_lists(self):
        for value in ("", "0,", ",0", "1,,0", "-1", "1,1", "0,00", "GPU-1111", "1.0", "+1", "2147483648"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                setup.device_list(value)

    def test_invalid_cache_limits(self):
        for value in ("-1", "1.5", "auto", "", "2147483648"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                setup.nonnegative_int(value)

    def test_unavailable_cuda_ordinal_is_not_physical_index(self):
        with self.assertRaisesRegex(ValueError, "after CUDA_VISIBLE_DEVICES"):
            setup.select_gpus([gpu(2)], "2")

    def test_context_safeguard_and_explicit_override(self):
        with patch.object(setup, "warn") as warn:
            self.assertEqual(setup.context_limit("IQ3_XXS", 64, 262144), 131072)
            self.assertIn("--allow-high-context", warn.call_args.args[0])
            self.assertEqual(setup.context_limit("IQ3_XXS", 64, 262144, True), 262144)
            self.assertIn("at your own risk", warn.call_args.args[0])
        with patch.object(setup, "warn"):
            self.assertEqual(setup.context_limit("IQ3_S", 64, 262144), 131072)
            self.assertEqual(setup.context_limit("IQ3_S", 64, 262144, True), 262144)
        for model, ram, context in (("IQ3_XXS", 90, 262144), ("IQ3_XXS", 64, 131072), ("Q2_0", 64, 262144)):
            with self.subTest(model=model, ram=ram, context=context), patch.object(setup, "warn") as warn:
                self.assertEqual(setup.context_limit(model, ram, context), context)
                warn.assert_not_called()

    def test_installed_config_no_setup_fast_path_and_port_override(self):
        for args, port in (([], None), (["--yes"], None), (["--port", "9090"], 9090)):
            with self.subTest(args=args), patch.object(setup, "installed_configs", return_value=[Path("saved.json")]), \
                    patch.object(setup, "start", return_value=7) as start, patch.object(setup, "gpu_infos") as detect, \
                    patch.object(setup, "say"), patch.object(setup, "update_installed_engine") as update:
                self.assertEqual(setup.main(args), 7)
                update.assert_called_once()
                start.assert_called_once_with(Path("saved.json"), port)
                detect.assert_not_called()

    def test_explicit_choices_do_not_launch_old_config(self):
        choices = (["--devices", "0"], ["--expert-vram-mib", "0"], ["--expert-reserve-mib", "1024"],
                   ["--context", "262144"], ["--allow-high-context"], ["--vision", "no"], ["--build"],
                   ["--prebuilt", "local"], ["--models-dir", "models"], ["--gguf-dir", "ggufs"],
                   ["--model", "IQ3_XXS"], ["--family", "swift"], ["--setup"], ["--check"], ["--no-start"],
                   ["--kv", "q4_0"], ["--kv-resident", "0"], ["--host", "0.0.0.0"], ["--api-key", "test"],
                   ["--experimental-speed-projection", "off"])
        for args in choices:
            with self.subTest(args=args), patch.object(setup, "installed_configs", return_value=[Path("saved.json")]), \
                    patch.object(setup, "start") as start, patch.object(setup, "gpu_infos", side_effect=RuntimeError("setup")), \
                    patch.object(setup, "say"), self.assertRaisesRegex(RuntimeError, "setup"):
                setup.main(args)
            start.assert_not_called()


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(setup, "ROOT", self.root))
        self.stack.enter_context(patch.object(setup, "say"))
        self.tools = self.stack.enter_context(patch.object(setup, "install_build_tools", return_value=("/cuda/bin/nvcc", None)))
        self.build = self.stack.enter_context(patch.object(setup, "cmake_build"))
        self.stack.enter_context(patch.object(setup.shutil, "copy2"))
        (self.root / "engine").mkdir()

    def stamp(self, archs=(86,), vision="none", source="local"):
        (self.root / "engine" / "BUILD.json").write_text(json.dumps({"source": source, "archs": list(archs),
                                                                     "vision": vision, "version": ".".join(map(str, setup.MIN_ENGINE)),
                                                                      "src": setup.source_hash(setup.ENGINE_SOURCES),
                                                                      "vision_src": setup.source_hash(setup.VISION_SOURCES)}))
        (self.root / "engine" / setup.EXE).touch()

    def test_mixed_architectures_deduplicated_and_toolkit_follows_newest(self):
        selected = [gpu(0), gpu(1, "120"), gpu(2)]
        setup.build_engine(selected[0], "gpu", True, self.root / "llama", gpus=selected)
        self.tools.assert_called_once_with(selected[1], True)
        self.assertEqual(self.build.call_count, 2)
        for call in self.build.call_args_list:
            self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86;120", call.args[3])
            self.assertNotEqual(call.args[1], self.root / "build")
        meta = json.loads((self.root / "engine" / "BUILD.json").read_text())
        self.assertEqual(meta["archs"], [86, 120])

    def test_multigpu_rebuilds_even_with_matching_local_stamp(self):
        self.stamp()
        setup.build_engine(gpu(0), "none", True, self.root / "llama", gpus=[gpu(0), gpu(2)])
        self.build.assert_called_once()

    def test_explicit_force_rebuilds_single_gpu(self):
        self.stamp()
        setup.build_engine(gpu(0), "none", True, self.root / "llama", force=True)
        self.build.assert_called_once()

    def test_default_single_gpu_can_reuse_compatible_build(self):
        self.stamp()
        self.assertEqual(setup.build_engine(gpu(0), "none", True, self.root / "llama"), self.root / "engine")
        self.build.assert_not_called()
        self.tools.assert_not_called()

    def test_new_gpu_architecture_rebuilds_stale_local_engine(self):
        self.stamp()
        setup.build_engine(gpu(1, "120"), "none", True, self.root / "llama")
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=120", self.build.call_args.args[3])

    def test_cpu_to_gpu_vision_rebuilds_encoder(self):
        self.stamp(vision="cpu")
        (self.root / "engine" / setup.VEXE).touch()
        setup.build_engine(gpu(0), "gpu", True, self.root / "llama")
        self.build.assert_called_once()
        self.assertEqual(self.build.call_args.args[2], "strata-vision")
        self.assertIn("-DSTRATA_VISION_CUDA=ON", self.build.call_args.args[3])

    def test_setup_does_not_overwrite_user_cmake_architectures(self):
        user_cache = self.root / "build" / "CMakeCache.txt"
        user_cache.parent.mkdir()
        user_cache.write_text("CMAKE_CUDA_ARCHITECTURES:STRING=89-real;120-virtual\n")
        setup.build_engine(gpu(0), "none", True, self.root / "llama", gpus=[gpu(0), gpu(2)])
        self.assertEqual(user_cache.read_text(), "CMAKE_CUDA_ARCHITECTURES:STRING=89-real;120-virtual\n")
        self.assertEqual(self.build.call_args.args[1], self.root / "build-setup")

    def test_cached_engine_with_missing_vision_helper_rebuilds(self):
        self.stamp(vision="gpu")
        self.assertIsNone(setup.get_prebuilt("", gpu(0), "gpu"))

    def test_source_change_rebuilds_and_refreshes_fingerprint(self):
        self.stamp()
        source = self.root / "src" / "test.cpp"
        source.parent.mkdir()
        source.write_text("// new source\n")
        self.assertIsNone(setup.get_prebuilt("", gpu(0), "none"))
        setup.build_engine(gpu(0), "none", True, self.root / "llama")
        self.build.assert_called_once()
        stamp = json.loads((self.root / "engine" / "BUILD.json").read_text())
        self.assertEqual(stamp["src"], setup.source_hash(setup.ENGINE_SOURCES))

    def test_automatic_rebuild_preserves_all_previously_built_architectures(self):
        self.stamp(archs=(86, 120))
        source = self.root / "src" / "test.cpp"
        source.parent.mkdir()
        source.write_text("// source changed after git pull\n")
        with patch.object(setup, "gpu_info", return_value=gpu(0)), \
                patch.object(setup, "get_llama_cpp", return_value=self.root / "llama"), \
                patch.object(setup, "get_prebuilt") as download:
            setup.update_installed_engine("unused")
        download.assert_not_called()
        self.build.assert_called_once()
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86;120", self.build.call_args.args[3])
        stamp = json.loads((self.root / "engine" / "BUILD.json").read_text())
        self.assertEqual(stamp["archs"], [86, 120])

    def test_unchanged_multigpu_source_build_is_not_updated_on_plain_restart(self):
        self.stamp(archs=(86, 120))
        with patch.object(setup, "gpu_info") as detect, patch.object(setup, "get_prebuilt") as download:
            setup.update_installed_engine("unused")
        detect.assert_not_called()
        download.assert_not_called()
        self.build.assert_not_called()

    def test_prebuilt_cache_requires_matching_architecture(self):
        self.stamp(source="release")
        self.assertEqual(setup.get_prebuilt("", gpu(0), "none"), self.root / "engine")
        self.assertIsNone(setup.get_prebuilt("", gpu(1, "120"), "none"))
        self.stamp(archs=(), source="release")
        self.assertIsNone(setup.get_prebuilt("", gpu(0), "none"))


class CmakeInvocationTests(unittest.TestCase):
    def test_multi_arch_argument_survives_linux_and_windows_commands(self):
        for windows in (False, True):
            with self.subTest(windows=windows), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                definition = "-DCMAKE_CUDA_ARCHITECTURES=86;120"
                with patch.object(setup, "ROOT", root), patch.object(setup, "WIN", windows), \
                        patch.object(setup, "find_tool", side_effect=lambda name: f"/Build Tools/{name}"), \
                        patch.object(setup, "run") as run:
                    setup.cmake_build(root, root / "build-setup", "strata", [definition], "vcvars64.bat", "build-strata.bat")
                if windows:
                    self.assertEqual(run.call_args.args[0][:2], ["cmd", "/c"])
                    script = (root / "build-strata.bat").read_text()
                    self.assertIn(definition, script)
                    self.assertIn('"/Build Tools/cmake"', script)
                else:
                    self.assertIn(definition, run.call_args_list[0].args[0])
                    self.assertEqual(run.call_count, 2)


class SetupConfigTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(setup, "ROOT", self.root))
        self.stack.enter_context(patch.object(setup, "say"))
        self.stack.enter_context(patch.object(setup, "gpu_infos", return_value=[gpu(0), gpu(1, "120", 32), gpu(2)]))
        self.stack.enter_context(patch.object(setup, "ram_gb", return_value=64))
        self.stack.enter_context(patch.object(setup, "cpu_info", return_value=("Test CPU", True, False)))
        self.stack.enter_context(patch.object(setup, "free_gb", return_value=1000))
        self.stack.enter_context(patch.object(setup, "pip_install"))
        self.stack.enter_context(patch.object(setup, "get_llama_cpp", return_value=self.root / "llama"))
        self.stack.enter_context(patch.object(setup, "cuda_lib_dirs", return_value=[]))
        self.stack.enter_context(patch.object(setup, "run"))
        self.stack.enter_context(patch.object(setup, "download"))
        self.stack.enter_context(patch.object(setup, "write_run_script", return_value=self.root / "run.sh"))
        self.eng = self.root / "engine"
        self.eng.mkdir()
        (self.eng / "BUILD.json").write_text('{"source": "local", "archs": [86, 120]}')
        self.build = self.stack.enter_context(patch.object(setup, "build_engine", return_value=self.eng))
        self.prebuilt = self.stack.enter_context(patch.object(setup, "get_prebuilt", return_value=self.eng))
        self.start = self.stack.enter_context(patch.object(setup, "start"))
        self.ggufs = self.root / "ggufs"
        self.ggufs.mkdir()
        for i in (1, 2):
            (self.ggufs / setup.FAMILIES["qwen"]["file"].format(q="IQ3_XXS", i=i)).touch()
        rt = self.root / "mtp" / "rt"
        rt.mkdir(parents=True)
        (rt / "experts.bin").touch()
        (rt / "draft_vocab.bin").touch()
        reader = types.ModuleType("gguf_reader")
        reader.GGUFFile = Mock(return_value=types.SimpleNamespace(tensors=[types.SimpleNamespace(name="per_layer_token_embd.weight")]))
        self.stack.enter_context(patch.dict(sys.modules, {"gguf_reader": reader}))

    def configure(self, *args):
        code = setup.main(["--family", "qwen", "--model", "IQ3_XXS", "--context", "262144", "--vision", "gpu",
                           "--gguf-dir", str(self.ggufs), "--no-start", "--yes", *args])
        self.assertEqual(code, 0)
        self.start.assert_not_called()
        return json.loads((self.root / "strata-iq3_xxs.json").read_text())

    def test_multigpu_config_preserves_order_cache_options_spec_and_primary_vision_uuid(self):
        cfg = self.configure("--devices", "2,1,0", "--expert-vram-mib", "4096", "--expert-reserve-mib", "2048",
                             "--allow-high-context")
        for name, value in (("--devices", "2,1,0"), ("--expert-vram-mib", "4096"), ("--expert-reserve-mib", "2048"),
                            ("--max-context", "262144"), ("--spec", "4")):
            self.assertEqual(cfg["args"][cfg["args"].index(name) + 1], value)
        self.assertIn("--native", cfg["args"])
        self.assertEqual(cfg["vision"]["gpu_uuid"], U2)
        self.prebuilt.assert_not_called()
        self.assertTrue(self.build.call_args.kwargs["force"])
        self.assertEqual([g["device"] for g in self.build.call_args.kwargs["gpus"]], [2, 1, 0])

    def test_single_gpu_default_keeps_prebuilt_path_and_old_engine_cli(self):
        cfg = self.configure()
        for name in ("--devices", "--expert-vram-mib", "--expert-reserve-mib"):
            self.assertNotIn(name, cfg["args"])
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "131072")
        self.assertEqual(cfg["vision"]["gpu_uuid"], U0)
        self.prebuilt.assert_called_once()
        self.build.assert_not_called()

    def test_explicit_single_gpu_or_zero_cap_requires_new_source_cli(self):
        for args in (("--devices", "1"), ("--expert-vram-mib", "0"), ("--expert-reserve-mib", "0")):
            with self.subTest(args=args):
                self.prebuilt.reset_mock()
                self.build.reset_mock()
                cfg = self.configure(*args)
                self.assertEqual(cfg["args"][cfg["args"].index(args[0]) + 1], args[1])
                self.prebuilt.assert_not_called()
                self.assertTrue(self.build.call_args.kwargs["force"])

    def test_multigpu_defaults_keep_kv_on_primary_and_projection_off(self):
        cfg = self.configure("--devices", "0,1")
        self.assertNotIn("--kv-resident", cfg["args"])
        self.assertEqual(cfg["args"][cfg["args"].index("--kv") + 1], "int8")
        self.assertNotIn("--control-vector-scaled", cfg["args"])

    def test_explicit_kv_streaming_and_precision_are_preserved(self):
        for cells in ("0", "32768"):
            with self.subTest(cells=cells):
                cfg = self.configure("--devices", "0,1", "--kv", "q4_0", "--kv-resident", cells)
                self.assertEqual(cfg["args"][cfg["args"].index("--kv-resident") + 1], cells)
                self.assertEqual(cfg["args"][cfg["args"].index("--kv") + 1], "q4_0")

    def test_upstream_iq3_s_uses_native_pack_and_retains_ram_safeguard(self):
        for i in (1, 2):
            (self.ggufs / setup.FAMILIES["qwen"]["file"].format(q="IQ3_S", i=i)).touch()
        code = setup.main(["--family", "qwen", "--model", "IQ3_S", "--context", "262144",
                           "--devices", "0,1", "--vision", "no", "--gguf-dir", str(self.ggufs),
                           "--no-start", "--yes"])
        self.assertEqual(code, 0)
        cfg = json.loads((self.root / "strata-iq3_s.json").read_text())
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "131072")
        self.assertIn("--native", cfg["args"])
        self.assertNotIn("--kv-resident", cfg["args"])
        self.assertNotIn("--control-vector-scaled", cfg["args"])

    def test_upstream_host_and_api_key_survive_explicit_reconfiguration(self):
        cfg = self.configure("--devices", "1,0", "--host", "0.0.0.0", "--api-key", "test-secret")
        self.assertEqual(cfg["host"], "0.0.0.0")
        self.assertEqual(cfg["api_key"], "test-secret")

    def test_cpu_vision_does_not_pin_cuda_mask(self):
        cfg = self.configure("--devices", "1,0", "--vision", "cpu")
        self.assertNotIn("gpu_uuid", cfg["vision"])
        self.assertFalse(cfg["vision"]["gpu"])


class ServingEnvironmentTests(unittest.TestCase):
    def test_only_gpu_vision_is_remapped_and_environment_is_not_mutated(self):
        cfg = {"vision": {"gpu": True, "gpu_uuid": U1}}
        original = {"CUDA_VISIBLE_DEVICES": "2,0,1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUSTOM": "keep"}
        with patch.dict(os.environ, original, clear=True):
            engine_env = server.child_env(cfg)
            vision_env = server.child_env(cfg, vision=True)
            self.assertEqual(dict(os.environ), original)
        self.assertEqual(engine_env, original)
        self.assertEqual(vision_env, dict(original, CUDA_VISIBLE_DEVICES=U1))

    def test_legacy_and_cpu_configs_inherit_mask(self):
        for cfg in ({}, {"vision": {"gpu": True}}, {"vision": {"gpu": False, "gpu_uuid": U1}}):
            with self.subTest(cfg=cfg), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": U0}, clear=True):
                self.assertEqual(server.child_env(cfg, vision=True)["CUDA_VISIBLE_DEVICES"], U0)

    def test_engine_does_not_invent_mask(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("CUDA_VISIBLE_DEVICES", server.child_env({"vision": {"gpu": True, "gpu_uuid": U1}}))

    def test_invalid_vision_uuid_is_not_used_as_mask(self):
        for value in ("0", "GPU-1111", f"{U0},{U1}", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                server.child_env({"vision": {"gpu": True, "gpu_uuid": value}}, vision=True)

    def test_library_paths_are_preserved_for_each_child_on_linux_and_windows(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = {"lib_dirs": [temp, str(Path(temp) / "missing")], "vision": {"gpu": True, "gpu_uuid": U2}}
            native_path = type(Path(temp))
            for platform, variable in (("posix", "LD_LIBRARY_PATH"), ("nt", "PATH")):
                with self.subTest(platform=platform), patch.object(server.os, "name", platform), \
                        patch.object(server, "Path", native_path), patch.dict(os.environ, {variable: "existing"}, clear=True):
                    self.assertEqual(server.child_env(cfg)[variable], temp + os.pathsep + "existing")
                    self.assertEqual(server.child_env(cfg, vision=True)[variable], temp + os.pathsep + "existing")

    def test_engine_process_keeps_native_serve_spec_and_device_arguments(self):
        args = ["--native", "model.gguf", "--spec", "4", "--devices", "1,0"]
        process = Mock(stdout=io.StringIO(f"INFO devices=1,0 gpu_uuids={U1},{U0}\nREADY 131072 stop\n"))
        with patch.object(server.subprocess, "Popen", return_value=process) as popen, \
                patch.object(server.threading, "Thread"), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,0"}, clear=True):
            env = server.child_env({"vision": {"gpu": True, "gpu_uuid": U1}})
            engine = server.StrataEngine("strata", args, env=env)
        self.assertEqual(popen.call_args.args[0], ["strata", "--serve", *args])
        self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2,0")
        self.assertEqual(engine.max_context, 131072)
        self.assertEqual(engine.info["gpu_uuids"], f"{U1},{U0}")
        with patch("serve.telemetry.Telemetry") as sampler:
            svc = server.Service(engine, server.ByteTokenizer(), Mock())
            svc.start_telemetry()
        self.assertEqual(sampler.call_args.kwargs["gpu_uuids"], [U1, U0])

    def test_server_startup_passes_separate_engine_and_vision_environments(self):
        with tempfile.TemporaryDirectory() as temp:
            tpath = Path(temp)
            (tpath / "vocab.json").write_text("{}")
            (tpath / "merges.txt").write_text("")
            (tpath / "token_type.json").write_text("[]")
            cfg = {"exe": "strata", "args": ["--devices", "1,0", "--spec", "4"], "tokenizer": temp,
                   "vision": {"gpu": True, "gpu_uuid": U2}}
            cpath = tpath / "config.json"
            cpath.write_text(json.dumps(cfg))
            tokenizer = types.ModuleType("strata_tokenizer")
            tokenizer.Tokenizer = Mock(return_value=server.ByteTokenizer())
            with patch.dict(sys.modules, {"strata_tokenizer": tokenizer}), \
                    patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config", str(cpath)]), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,0", "CUSTOM": "keep"}, clear=True), \
                    patch.object(server, "Server"), patch.object(server, "serve"), \
                    patch.object(server, "Service"), patch.object(server, "Vision") as vision, \
                    patch.object(server, "StrataEngine") as engine, patch("builtins.print"), \
                    patch.object(server.threading, "Event", return_value=Mock(wait=Mock(side_effect=KeyboardInterrupt))):
                self.assertEqual(server.main(), 0)
            self.assertEqual(vision.call_args.kwargs["env"], {"CUDA_VISIBLE_DEVICES": U2, "CUSTOM": "keep"})
            self.assertEqual(engine.call_args.kwargs["env"], {"CUDA_VISIBLE_DEVICES": "2,0", "CUSTOM": "keep"})
            self.assertEqual(engine.call_args.args, ("strata", cfg["args"]))

    def test_vision_process_receives_only_primary_uuid(self):
        cfg = {"vision": {"exe": "strata-vision", "model": "model.gguf", "mmproj": "vision.gguf",
                           "gpu": True, "gpu_uuid": U2}}
        with tempfile.TemporaryDirectory() as temp, patch.object(server.tempfile, "mkdtemp", return_value=temp), \
                patch.object(server.subprocess, "Popen", return_value=Mock(stdout=io.StringIO("READY\n"))) as popen, \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,2,0", "CUSTOM": "keep"}, clear=True):
            server.Vision(cfg["vision"], env=server.child_env(cfg, vision=True))
        self.assertIn("--gpu", popen.call_args.args[0])
        self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], U2)
        self.assertEqual(popen.call_args.kwargs["env"]["CUSTOM"], "keep")


if __name__ == "__main__":
    unittest.main()
