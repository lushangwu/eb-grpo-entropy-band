"""EB-GRPO 训练桥（由 install_eb_patch.py 安装到 verl/workers/utils/eb_bridge.py）。

职责：
- eb_step_begin(cfg)：每个 optimizer step 开始时调用。用上一步观测的
  token 平均策略熵更新熵带控制器，把控制量写回 worker 持有的 ActorConfig
  （entropy_coeff / clip_ratio_high）。ppo_loss 每个微批实时读同一个
  config 对象，因此本步内所有损失计算都用新参数。首步无观测：保持配置初值（中性）。
- eb_step_end(cfg, output)：step 结束时调用。从 train_mini_batch 返回的
  non-tensor `metrics`（{name: [Metric|float, ...]}，每个微批一条）用 verl
  自带的 reduce_metrics 聚合成标量，① 取 token 平均熵存为下一步观测；
  ② 把 {step, entropy, u, entropy_coeff, clip_higher, ...} 追加到 JSONL 日志。

说明：
- 反馈存在一步延迟（t 步的参数由 t-1 步的熵决定）——PI 控制器带宽远低
  于训练步频，论文方法章按采样控制论述，不影响机制。
- 熵在 verl 里是正值（losses.py: policy_loss -= entropy_coeff * entropy_loss），
  与控制器观测约定一致。
- baseline/fixed 组（eb_enabled=False）不做任何控制动作，但仍写日志，
  使三组有同构的过程指标序列（论文绘图的数据契约）。
- 多 DP rank 时各 rank 独立更新会造成参数分歧，且每 rank 各写一行；本实验为
  单卡（dp=1），日志只在 rank0 写；多卡扩展留待未来工作（论文局限已声明）。
"""

from __future__ import annotations

import json
import os

_STATE = {
    "controller": None,
    "last_entropy": None,
    "step": 0,
    "warned": False,
}

# 日志字段名 -> verl 聚合后指标名（按候选顺序取第一个命中的）
_METRIC_SLOTS = {
    "entropy": ("actor/entropy_loss", "actor/entropy"),
    "kl_loss": ("kl_loss", "actor/kl_loss"),
    "ppo_kl": ("actor/ppo_kl", "ppo_kl"),
    "clipfrac": ("actor/pg_clipfrac", "pg_clipfrac"),
    "clipfrac_lower": ("actor/pg_clipfrac_lower", "pg_clipfrac_lower"),
    "pg_loss": ("actor/pg_loss", "pg_loss"),
    "grad_norm": ("actor/grad_norm", "grad_norm"),
}


def _is_log_rank() -> bool:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def _ensure_controller(cfg):
    if _STATE["controller"] is None:
        from verl.workers.utils.eb_entropy_controller import EntropyBandController

        _STATE["controller"] = EntropyBandController(
            h_lo=float(cfg.eb_h_lo),
            h_hi=float(cfg.eb_h_hi),
            coeff_max=float(cfg.eb_coeff_max),
            eps_base=0.2,
            eps_high_max=float(cfg.eb_eps_high_max),
            k_p=float(cfg.eb_k_p),
            k_i=float(cfg.eb_k_i),
            leak=float(cfg.eb_leak),
            i_clamp=float(cfg.eb_i_clamp),
            band_relative=bool(cfg.eb_band_relative),
            h_ref=float(cfg.eb_h_ref),
        )


def eb_step_begin(cfg) -> None:
    if cfg is None or not getattr(cfg, "eb_enabled", False):
        return
    _ensure_controller(cfg)
    if _STATE["last_entropy"] is None:
        return  # 首步无观测：保持配置初值
    sig = _STATE["controller"].update(_STATE["last_entropy"])
    cfg.entropy_coeff = sig.entropy_coeff
    cfg.clip_ratio_high = sig.clip_higher


def _reduce(output):
    """把 train_mini_batch 返回的 per-micro-batch 指标聚合成 {name: float}。"""
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric import reduce_metrics

    metrics = tu.get(output, "metrics")
    if not metrics:
        return {}
    flat = reduce_metrics(dict(metrics))
    out = {}
    for key, val in flat.items():
        try:
            out[key] = float(val)
        except (TypeError, ValueError):
            continue  # 跳过非数值项
    return out


def eb_step_end(cfg, output) -> None:
    if cfg is None or output is None:
        return
    try:
        flat = _reduce(output)
    except Exception as exc:  # 诊断用：让第一步就把取不到指标的问题暴露到 stdout
        if not _STATE["warned"]:
            print(f"[eb_bridge] 聚合指标失败：{exc!r}", flush=True)
            _STATE["warned"] = True
        return

    h = None
    for name in _METRIC_SLOTS["entropy"]:
        if name in flat:
            h = flat[name]
            break
    if h is None:
        if not _STATE["warned"]:
            print(f"[eb_bridge] 返回指标里没有熵，可用键={sorted(flat)}", flush=True)
            _STATE["warned"] = True
        return

    if getattr(cfg, "eb_enabled", False):
        _STATE["last_entropy"] = h
    _STATE["step"] += 1

    log_file = getattr(cfg, "eb_log_file", "") or ""
    if not log_file or not _is_log_rank():
        return

    hist = _STATE["controller"].history if _STATE["controller"] is not None else []
    sig = hist[-1] if hist else None  # 首步控制器尚未动作：无控制信号
    rec = {
        "step": _STATE["step"],
        "entropy": h,
        "u": getattr(sig, "u", 0.0),
        "entropy_coeff": float(cfg.entropy_coeff),
        "clip_higher": float(cfg.clip_ratio_high),
    }
    if sig is not None:
        rec["error"] = getattr(sig, "error", None)
        rec["integral"] = getattr(sig, "integral", None)
    ctrl = _STATE["controller"]
    if ctrl is not None:
        # 本步实际生效的绝对熵带（相对带模式下由参考熵折算而来）
        rec["band_lo"] = float(ctrl.h_lo)
        rec["band_hi"] = float(ctrl.h_hi)
        rec["h_ref"] = float(ctrl.h_ref)
    for field, names in _METRIC_SLOTS.items():
        if field == "entropy":
            continue
        for name in names:
            if name in flat:
                rec[field] = flat[name]
                break
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
