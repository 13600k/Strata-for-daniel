"""GPU telemetry selection is driven by engine UUIDs, not physical index zero."""
import unittest
from unittest.mock import Mock, patch
from serve import telemetry

U0 = "GPU-11111111-1111-1111-1111-111111111111"
U1 = "GPU-22222222-2222-2222-2222-222222222222"


class GpuTelemetry(unittest.TestCase):
    def test_nvml_selects_uuid_without_index_fallback(self):
        lib = Mock()
        lib.nvmlInit_v2.return_value = 0

        def select(uid, handle):
            handle._obj.value = 123
            return 0

        lib.nvmlDeviceGetHandleByUUID.side_effect = select
        with patch.object(telemetry.ctypes, "CDLL", return_value=lib):
            gpu = telemetry._Nvml(uuid=U1)
        self.assertTrue(gpu.ok())
        self.assertEqual(lib.nvmlDeviceGetHandleByUUID.call_args.args[0], U1.encode("ascii"))
        lib.nvmlDeviceGetHandleByIndex_v2.assert_not_called()

        lib.nvmlDeviceGetHandleByUUID.side_effect = None
        lib.nvmlDeviceGetHandleByUUID.return_value = 1
        with patch.object(telemetry.ctypes, "CDLL", return_value=lib):
            gpu = telemetry._Nvml(uuid=U1)
        self.assertFalse(gpu.ok())
        lib.nvmlDeviceGetHandleByIndex_v2.assert_not_called()

    def test_readings_preserve_selection_order_and_primary_charts(self):
        first = Mock(ok=Mock(return_value=True), read=Mock(return_value={"util": 70, "mem_used": 100}))
        second = Mock(ok=Mock(return_value=True), read=Mock(return_value={"util": 20, "mem_used": 200}))
        first.name.return_value = "first"
        second.name.return_value = "second"
        with patch.object(telemetry, "_Nvml", side_effect=[first, second]) as nvml, \
                patch.object(telemetry, "_cpu_name", return_value="test CPU"), \
                patch.object(telemetry.threading, "Thread"):
            tel = telemetry.Telemetry(gpu_uuids=[U1, U0])
        self.assertEqual([c.kwargs for c in nvml.call_args_list], [{"uuid": U1}, {"uuid": U0}])
        tel.ps = None
        tel.fallback = Mock(cpu=Mock(return_value=1), ram=Mock(return_value=(2, 3)))
        s = tel.sample()
        self.assertEqual(s["gpu_util"], 70)
        self.assertEqual(s["gpu_mem_used"], 100)
        self.assertEqual([g["uuid"] for g in s["gpus"]], [U1, U0])
        self.assertEqual([g["role"] for g in s["gpus"]], ["primary", "experts"])
        self.assertEqual(s["gpus"][1]["mem_used"], 200)
        self.assertEqual(tel.static["gpu_name"], "first")

    def test_legacy_engine_retains_one_default_monitor(self):
        with patch.object(telemetry, "_Nvml") as nvml, \
                patch.object(telemetry.threading, "Thread"), patch.object(telemetry, "_cpu_name"):
            tel = telemetry.Telemetry()
        nvml.assert_called_once_with(0)
        self.assertEqual(len(tel.gpus), 1)
        self.assertIsNone(tel.static["gpus"][0]["uuid"])


if __name__ == "__main__":
    unittest.main()
