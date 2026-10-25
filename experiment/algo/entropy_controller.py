"""EB-GRPO 熵带控制器（Entropy-Band GRPO）——论文自研算法的核心模块。

纯标准库实现，可在无 GPU 环境下开发与单测；训练时作为 verl 策略更新
外挂的反馈控制器（集成方式见 README_EB_GRPO.md）。

设计要点（与论文第 4 章一一对应）：
- 观测量：每步 rollout 的 token 级平均策略熵 H_t（nats）；
- 死区：熵落在目标带 [h_lo, h_hi] 内时误差为 0，不动作，避免抖动；
- 误差（方向约定：正 = 探索不足）：
      带下方（H_t < h_lo）: e = h_lo - H_t   （>0，需要鼓励探索）
      带上方（H_t > h_hi）: e = -(H_t - h_hi)（<0，需要抑制）
- 控制律（比例-积分，含抗饱和与泄漏）：
      I_t = clamp(leak * I_{t-1} + e_t, -i_clamp, +i_clamp)
      u_t = clip(k_p * e_t + k_i * I_t, -1, +1)
- 双执行器（**双向**，2026-10-24 预跑实测后由非对称改为对称）：
      entropy_coeff = u * coeff_max                       # u>0 熵奖励，u<0 熵惩罚
      clip_higher   = eps_base + max(0, u) * (eps_high_max - eps_base)
  两个执行器的分工是不对称的：熵系数双向，裁剪只单向放宽——clip-higher 的
  语义（DAPO，arXiv 2503.14476）本来就是"放宽正优势 token 的更新上界以放开
  探索"，反向收紧等价于回到默认对称裁剪 0.2，因此抑制方向只动用熵系数。
  改双向的实测依据：0.5B/1.5B base 模型在 4 个数 Countdown 上 vanilla GRPO 的
  熵是**单调上升并飘出带外**（1.80→2.36 nats），只抬不压的控制器对整个训练
  过程完全不动作，无法构成"维持熵在带内"的验证。

熵带的两种给法（对应论文 §4.2 取值表）：
- 绝对带（band_relative=False）：直接给 [h_lo, h_hi]（nats）。
- 相对带（band_relative=True）：h_lo/h_hi 解释为**初始熵的比例**，
  带宽 = [r_lo, r_hi] × H_ref。H_ref 优先取显式给定的 h_ref；为 0 时用
  控制器收到的第一个观测熵自标定。跨模型规模（0.5B / 1.5B 初始熵不同）
  因此不需要逐模型重设常数，这正是"少先验调参"主张的实现。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControlSignal:
    """单步控制信号（完整记录，供训练日志与论文绘图使用）。"""

    step: int
    entropy: float      # 观测熵 H_t
    error: float        # 方向化误差 e_t
    integral: float     # 积分项 I_t
    u: float            # 归一化控制量
    entropy_coeff: float
    clip_higher: float


class EntropyBandController:
    def __init__(
        self,
        h_lo: float,
        h_hi: float,
        coeff_max: float = 0.01,
        eps_base: float = 0.2,
        eps_high_max: float = 0.28,
        k_p: float = 4.0,
        k_i: float = 0.5,
        i_clamp: float = 8.0,
        leak: float = 0.95,
        band_relative: bool = False,
        h_ref: float = 0.0,
    ):
        # 绝对带时 h_* 是 nats；相对带时是初始熵的比例，取值约束相同
        if not (0.0 < h_lo < h_hi):
            raise ValueError("要求 0 < h_lo < h_hi（相对带时为 0 < 比例下界 < 上界）")
        if h_ref < 0.0:
            raise ValueError("h_ref 不允许为负；0 表示用首个观测自标定")
        self.band_relative = band_relative
        self.r_lo, self.r_hi = h_lo, h_hi
        if band_relative:
            # h_ref>0：立刻折算；h_ref=0：等第一个观测熵到来时折算
            self.h_ref = h_ref
            self.h_lo = h_lo * h_ref
            self.h_hi = h_hi * h_ref
        else:
            self.h_ref = 0.0
            self.h_lo = h_lo
            self.h_hi = h_hi
        self.coeff_max = coeff_max
        self.eps_base = eps_base
        self.eps_high_max = eps_high_max
        self.k_p = k_p
        self.k_i = k_i
        self.i_clamp = i_clamp
        self.leak = leak
        self.integral = 0.0
        self.history: list[ControlSignal] = []
        self._step = 0

    def error_of(self, h: float) -> float:
        """方向化误差：带下方为正（探索不足），带上方为负，带内为 0。"""
        if h < self.h_lo:
            return self.h_lo - h
        if h > self.h_hi:
            return -(h - self.h_hi)
        return 0.0

    def update(self, entropy: float) -> ControlSignal:
        if self.band_relative and self.h_ref == 0.0:
            # 自标定：以收到的第一个观测熵为参考熵 H0，把比例带宽折算成绝对带宽
            if entropy <= 0.0:
                raise ValueError("相对带的参考熵必须为正")
            self.h_ref = entropy
            self.h_lo = self.r_lo * entropy
            self.h_hi = self.r_hi * entropy
        e = self.error_of(entropy)
        # 泄漏 + 抗饱和：带内误差为 0，积分按 leak 衰减回零，防止 windup
        self.integral = max(-self.i_clamp, min(self.i_clamp, self.leak * self.integral + e))
        u = max(-1.0, min(1.0, self.k_p * e + self.k_i * self.integral))
        boost = max(0.0, u)
        sig = ControlSignal(
            step=self._step,
            entropy=entropy,
            error=e,
            integral=self.integral,
            u=u,
            entropy_coeff=u * self.coeff_max,
            clip_higher=self.eps_base + boost * (self.eps_high_max - self.eps_base),
        )
        self.history.append(sig)
        self._step += 1
        return sig
