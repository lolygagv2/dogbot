#!/usr/bin/env python3
"""The daytime camera profile follows the module fitted RIGHT NOW, not the yaml history.

Morgan swaps standard (IR-cut) and NoIR Camera Module 3 Wide on treatbot4 all the
time. libcamera reports 'imx708_wide' vs 'imx708_wide_noir'; select_day_profile()
must pick the matching yaml block, with the flat legacy keys as fallback.
Run: python3 tests/perception/test_camera_variant_profile.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml  # noqa: E402

from services.perception.camera_detect import (  # noqa: E402
    camera_variant, is_noir_model, select_day_profile,
)

PASS = 0


def check(name, cond):
    global PASS
    print(f"{'PASS' if cond else 'FAIL'}: {name}")
    if cond:
        PASS += 1
    else:
        sys.exit(1)


cfg = {
    'rotation': 180,
    'standard_day_profile': {},
    'noir_day_profile': {'awb_enable': False, 'colour_gains': [0.887, 1.52], 'saturation': 0.85},
}

check("variant strings", camera_variant('imx708_wide') == 'standard'
      and camera_variant('imx708_wide_noir') == 'noir'
      and camera_variant('') == 'unknown'
      and is_noir_model('IMX708_WIDE_NOIR') and not is_noir_model('imx708'))

check("standard module -> no colour overrides (AWB auto)",
      select_day_profile(cfg, 'imx708_wide') == {})

noir = select_day_profile(cfg, 'imx708_wide_noir')
check("noir module -> static gain lock from noir_day_profile",
      noir['awb_enable'] is False and noir['colour_gains'] == [0.887, 1.52] and noir['saturation'] == 0.85)

legacy = {'rotation': 180, 'awb_mode': 'daylight', 'contrast': 1.1}
check("legacy flat keys used when no variant block exists",
      select_day_profile(legacy, 'imx708_wide') == {'awb_mode': 'daylight', 'contrast': 1.1}
      and select_day_profile(legacy, 'imx708_wide_noir') == {'awb_mode': 'daylight', 'contrast': 1.1})

check("rotation/servo keys never leak into camera controls",
      'rotation' not in select_day_profile({'rotation': 180, 'pan_min': -89}, 'imx708_wide'))

tb4 = yaml.safe_load(open(Path(__file__).resolve().parents[2] / 'config/robot_profiles/treatbot4.yaml'))['camera']
check("treatbot4.yaml: standard -> {} , noir -> gain lock",
      select_day_profile(tb4, 'imx708_wide') == {}
      and select_day_profile(tb4, 'imx708_wide_noir')['colour_gains'] == [0.887, 1.52])

print(f"\n{PASS} PASS")
