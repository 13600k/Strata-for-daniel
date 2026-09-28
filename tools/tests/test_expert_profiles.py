"""Validate upstream's shipped complete rankings for canonical and pruned models.

These are byte/placement checks, not GPU accuracy or model benchmarks.
"""
import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ShippedExpertProfiles(unittest.TestCase):
    def test_profiles_cover_their_model_geometry(self):
        for name, experts in (("expert-profile.bin", 512), ("expert-profile-coder.bin", 256)):
            with self.subTest(profile=name):
                data = (ROOT / "data" / name).read_bytes()
                self.assertEqual(data[:4], b"STRP")
                version, layers, ne, slots, ranked = struct.unpack_from("<5I", data, 4)
                self.assertEqual((version, layers, ne), (1, 48, experts))
                self.assertEqual(slots, 48 * experts)
                self.assertEqual(ranked, slots)
                pairs = list(struct.iter_unpack("<HH", data[24:24 + ranked * 4]))
                self.assertEqual(len(pairs), ranked)
                self.assertEqual(set(pairs), {(l, e) for l in range(48) for e in range(experts)})
                residency = struct.unpack_from(f"<{ranked}i", data, 24 + ranked * 4)
                for slot, (layer, expert) in enumerate(pairs):
                    self.assertEqual(residency[layer * experts + expert], slot)


if __name__ == "__main__":
    unittest.main()
