"""EB-GRPO 熵带控制器的单元测试（纯标准库，无 GPU）。

运行：python tests/test_entropy_controller.py
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from entropy_controller import EntropyBandController  # noqa: E402


def run_steps(c: EntropyBandController, entropies):
    return [c.update(h) for h in entropies]


class TestEntropyBandController(unittest.TestCase):
    def test_deep_below_band_saturates_boost(self):
        """熵远低于带下界：u 饱和到 +1，双执行器都给到上限。"""
        c = EntropyBandController(h_lo=0.2, h_hi=0.4, coeff_max=0.01)
        sig = run_steps(c, [0.02] * 5)[-1]
        self.assertAlmostEqual(sig.u, 1.0)
        self.assertAlmostEqual(sig.entropy_coeff, 0.01)
        self.assertAlmostEqual(sig.clip_higher, 0.28)

    def test_above_band_applies_penalty(self):
        """熵高于带上界：熵系数转负（熵惩罚）；裁剪只单向放宽，故回到 eps_base。"""
        c = EntropyBandController(h_lo=0.2, h_hi=0.4)
        sig = run_steps(c, [0.6] * 5)[-1]
        self.assertLess(sig.u, 0.0)
        self.assertLess(sig.entropy_coeff, 0.0)
        self.assertAlmostEqual(sig.clip_higher, 0.2)

    def test_deep_above_band_saturates_penalty(self):
        """熵远高于带上界：u 饱和到 -1，熵惩罚给到 -coeff_max（与抬升侧同幅度）。"""
        c = EntropyBandController(h_lo=0.2, h_hi=0.4, coeff_max=0.01)
        sig = run_steps(c, [5.0] * 5)[-1]
        self.assertAlmostEqual(sig.u, -1.0)
        self.assertAlmostEqual(sig.entropy_coeff, -0.01)

    def test_deadband_no_action(self):
        """带内且积分已衰减：不动作。"""
        c = EntropyBandController(h_lo=0.2, h_hi=0.4)
        run_steps(c, [0.02] * 20)  # 先制造饱和
        sig = run_steps(c, [0.3] * 400)[-1]  # 回带内，泄漏使积分归零
        self.assertAlmostEqual(sig.u, 0.0, places=6)
        self.assertAlmostEqual(sig.error, 0.0)

    def test_linear_region(self):
        """小偏差处于线性区：u 与 e 成比例（比例项主导）。"""
        c = EntropyBandController(h_lo=0.2, h_hi=0.4, k_p=4.0, k_i=0.0)
        sig = c.update(0.19)  # e = 0.01
        self.assertAlmostEqual(sig.u, 0.04, places=9)

    def test_anti_windup(self):
        """长时间饱和后积分有界；回到带内后 u 在有限步内归零。"""
        c = EntropyBandController(h_lo=0.2, h_hi=0.4, i_clamp=4.0, leak=0.95, k_i=0.5)
        run_steps(c, [0.01] * 500)  # 试图把积分推爆
        self.assertLessEqual(abs(c.integral), 4.0 + 1e-9)
        sig = run_steps(c, [0.3] * 300)[-1]
        self.assertAlmostEqual(sig.u, 0.0, places=6)

    def test_band_config_validation(self):
        with self.assertRaises(ValueError):
            EntropyBandController(h_lo=0.4, h_hi=0.2)
        with self.assertRaises(ValueError):
            EntropyBandController(h_lo=-0.1, h_hi=0.2)
        # 相对带的比例同样要求下界 < 上界且为正
        with self.assertRaises(ValueError):
            EntropyBandController(h_lo=1.0, h_hi=0.5, band_relative=True)
        with self.assertRaises(ValueError):
            EntropyBandController(h_lo=0.5, h_hi=0.9, band_relative=True, h_ref=-1.0)

    def test_relative_band_explicit_reference(self):
        """给定 H_ref：比例带宽立刻折算成绝对带宽。"""
        c = EntropyBandController(h_lo=0.5, h_hi=0.9, band_relative=True, h_ref=2.0)
        self.assertAlmostEqual(c.h_lo, 1.0)
        self.assertAlmostEqual(c.h_hi, 1.8)
        sig = c.update(0.5)  # 远低于带下界 -> 拉满
        self.assertAlmostEqual(sig.u, 1.0)
        self.assertAlmostEqual(sig.entropy_coeff, c.coeff_max)

    def test_relative_band_self_calibration(self):
        """不给 H_ref：用第一个观测熵自标定；首个观测在带上方，故先给惩罚。"""
        c = EntropyBandController(h_lo=0.5, h_hi=0.9, band_relative=True, k_p=4.0, k_i=0.5)
        first = c.update(2.0)
        self.assertAlmostEqual(c.h_lo, 1.0)
        self.assertAlmostEqual(c.h_hi, 1.8)
        self.assertAlmostEqual(first.error, -0.2)  # 2.0 高于带上界 1.8
        self.assertLess(first.entropy_coeff, 0.0)  # 双向：飘出带外就压
        # 熵回落到带下方后，控制器转为抬升
        tail = [c.update(h) for h in [1.6, 1.2, 0.95, 0.9, 0.85]]
        self.assertGreater(tail[-1].entropy_coeff, 0.0)

    def test_trajectory_stabilizes_entropy(self):
        """闭环仿真（确定性被控对象）：熵应保持有界且不塌到地板。

        被控对象模型：h_{t+1} = h_t - d + g * coeff_t
        （d=0.008 模拟 GRPO 自身的熵衰减趋势；g=10 为熵奖励的抬升增益）。
        开环（coeff 恒 0）时熵必然单调坍缩到 0；闭环下断言两条硬性质：
        后 1000 步熵始终有界（0 < h < 0.7），且不再贴地板。
        """
        c = EntropyBandController(
            h_lo=0.2, h_hi=0.4, k_p=4.0, k_i=0.5, leak=0.9, i_clamp=4.0
        )
        h = 0.6
        trace = []
        for t in range(2000):
            sig = c.update(h)
            h = h - 0.008 + sig.entropy_coeff * 10.0
            h = max(0.001, h)
            trace.append(h)
        tail = trace[-1000:]
        self.assertTrue(all(0.0 < x < 0.7 for x in tail), f"熵越界: min={min(tail)}, max={max(tail)}")
        floor_hits = sum(1 for x in tail if x <= 0.002)
        self.assertEqual(floor_hits, 0, "闭环下熵不应贴地板（开环必然坍缩贴地）")


    def test_symmetric_loop_contains_upward_drift(self):
        """预跑实测的另一个方向：熵单调上飘。闭环（双向）应把它压回带附近，
        同一被控对象在开环（coeff 恒 0，等价 vanilla GRPO）下必然一路走高。

        被控对象：h_{t+1} = h_t + 0.01 + 10 * entropy_coeff_t
        （+0.01 是预跑里观察到的熵上升趋势；熵系数为负即惩罚，故能拉回）。
        """
        d, gain, steps = 0.01, 10.0, 200

        def run(use_controller):
            c = EntropyBandController(
                h_lo=0.5, h_hi=1.1, h_ref=1.8, band_relative=True,
                coeff_max=0.01, k_p=4.0, k_i=0.5, leak=0.9,
            )
            h = 1.8
            trace = []
            for _ in range(steps):
                coeff = c.update(h).entropy_coeff if use_controller else 0.0
                h = max(0.001, h + d + gain * coeff)
                trace.append(h)
            return trace

        open_loop, closed = run(False), run(True)
        self.assertGreater(open_loop[-1], 3.5)  # 开环：200 步一路飘到 3.8
        self.assertLess(closed[-1], 2.2)  # 闭环：稳在带上界(1.98)附近
        self.assertLess(sum(closed[-50:]) / 50, sum(open_loop[-50:]) / 50 - 1.0)
        self.assertTrue(all(0.001 < x < 3.0 for x in closed[50:]), "闭环后段应有界")


if __name__ == "__main__":
    unittest.main(verbosity=2)
