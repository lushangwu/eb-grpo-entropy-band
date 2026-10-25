"""在 verl v0.8.0 克隆上安装 EB-GRPO 补丁（幂等，可重复执行）。

改动面（全部有锚点、可回退）：
1. 拷贝 experiment/algo/entropy_controller.py -> verl/workers/utils/eb_entropy_controller.py
2. 拷贝 experiment/verl_integration/eb_bridge.py -> verl/workers/utils/eb_bridge.py
3. verl/workers/config/actor.py：ActorConfig 增加 eb_* 字段（带默认值，向后兼容），
   并把 entropy_coeff / clip_ratio_high 加入 _mutable_fields 供控制器逐步改写
4. verl/workers/engine_workers.py：
   a. init_model 中保存 actor_config 引用（ppo_loss 的 partial 用的就是它）
   b. update_actor 注入 eb_step_begin/eb_step_end

安装后生成 experiment/patches/eb_grpo_verl_v0.8.0.patch 存档（git diff）。
用法（服务器）：python install_eb_patch.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

EBGRPO = "/root/ebgrpo"
VERL = os.path.join(EBGRPO, "verl")
EXP = os.path.join(EBGRPO, "experiment")
PATCH_DIR = os.path.join(EXP, "patches")

EB_FIELDS = """    eb_enabled: bool = False
    eb_h_lo: float = 0.20
    eb_h_hi: float = 0.40
    eb_band_relative: bool = False
    eb_h_ref: float = 0.0
    eb_coeff_max: float = 0.01
    eb_eps_high_max: float = 0.28
    eb_k_p: float = 4.0
    eb_k_i: float = 0.5
    eb_leak: float = 0.95
    eb_i_clamp: float = 8.0
    eb_log_file: str = ""
"""

UPDATE_ACTOR_OLD = """    def update_actor(self, data: TensorDict) -> TensorDict:
        output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None
"""

UPDATE_ACTOR_NEW = """    def update_actor(self, data: TensorDict) -> TensorDict:
        from verl.workers.utils.eb_bridge import eb_step_begin, eb_step_end

        cfg = getattr(self, "_eb_actor_config", None)
        eb_step_begin(cfg)
        output = self.actor.train_mini_batch(data=data)
        out = output.cpu() if output is not None else None
        eb_step_end(cfg, out)
        return out
"""


def patch_file(path: str, edits: list[tuple[str, str, str]]) -> bool:
    """对文件逐个 (anchor, replacement, tag) 做锚点替换；tag 已存在则跳过。幂等。"""
    text = open(path, encoding="utf-8").read()
    changed = False
    for anchor, replacement, tag in edits:
        if tag in text:
            print(f"  [skip] {os.path.basename(path)}: {tag} 已存在")
            continue
        if anchor not in text:
            raise RuntimeError(f"锚点未找到（verl 版本不符？）: {path} :: {anchor[:60]!r}")
        text = text.replace(anchor, replacement, 1)
        changed = True
        print(f"  [ok]   {os.path.basename(path)}: {tag}")
    if changed:
        shutil.copy(path, path + ".pre_eb.bak")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return changed


def main() -> int:
    # 1/2. 拷贝两个新文件
    for src, dst in (
        (os.path.join(EXP, "algo", "entropy_controller.py"), os.path.join(VERL, "verl/workers/utils/eb_entropy_controller.py")),
        (os.path.join(EXP, "verl_integration", "eb_bridge.py"), os.path.join(VERL, "verl/workers/utils/eb_bridge.py")),
    ):
        shutil.copy(src, dst)
        print(f"  [copy] {dst}")

    # 3. ActorConfig 增加 eb_* 字段 + 放开两个执行器
    #    BaseConfig.__setattr__ 对已存在字段抛 FrozenInstanceError，除非名字在
    #    _mutable_fields 里；控制器逐步改写 entropy_coeff / clip_ratio_high 必须
    #    走这条 verl 官方通道（实测：不改写会 FrozenInstanceError）。
    patch_file(
        os.path.join(VERL, "verl/workers/config/actor.py"),
        [
            ("    entropy_coeff: float = 0\n", "    entropy_coeff: float = 0\n" + EB_FIELDS, "eb_enabled"),
            # 增量条目：给已被旧版脚本打过（缺相对带两字段）的克隆补上；全新克隆会因 tag 存在而跳过
            (
                "    eb_coeff_max: float = 0.01\n",
                "    eb_band_relative: bool = False\n    eb_h_ref: float = 0.0\n    eb_coeff_max: float = 0.01\n",
                "eb_band_relative",
            ),
            (
                '        "model_config",\n    }\n',
                '        "model_config",\n'
                '        # EB-GRPO 执行器：控制器逐步改写（见 verl/base_config.py 的 _mutable_fields 约定）\n'
                '        "entropy_coeff",\n'
                '        "clip_ratio_high",\n'
                "    }\n",
                "EB-GRPO 执行器",
            ),
        ],
    )

    # 3b. yaml 同步 eb_* 默认值（hydra struct 校验要求键在 yaml 中存在）
    EB_YAML_TOP = (
        "eb_enabled: false\n"
        "eb_h_lo: 0.20\n"
        "eb_h_hi: 0.40\n"
        "eb_band_relative: false\n"
        "eb_h_ref: 0.0\n"
        "eb_coeff_max: 0.01\n"
        "eb_eps_high_max: 0.28\n"
        "eb_k_p: 4.0\n"
        "eb_k_i: 0.5\n"
        "eb_leak: 0.95\n"
        "eb_i_clamp: 8.0\n"
        'eb_log_file: ""\n'
    )
    EB_YAML_INDENT = "".join("    " + line for line in EB_YAML_TOP.splitlines(keepends=True))
    patch_file(
        os.path.join(VERL, "verl/trainer/config/actor/actor.yaml"),
        [
            ("entropy_coeff: 0\n", "entropy_coeff: 0\n" + EB_YAML_TOP, "eb_enabled"),
            (
                "eb_coeff_max: 0.01\n",
                "eb_band_relative: false\neb_h_ref: 0.0\neb_coeff_max: 0.01\n",
                "eb_band_relative",
            ),
        ],
    )
    patch_file(
        os.path.join(VERL, "verl/trainer/config/_generated_ppo_trainer.yaml"),
        [
            ("    entropy_coeff: 0\n", "    entropy_coeff: 0\n" + EB_YAML_INDENT, "eb_enabled"),
            (
                "    eb_coeff_max: 0.01\n",
                "    eb_band_relative: false\n    eb_h_ref: 0.0\n    eb_coeff_max: 0.01\n",
                "eb_band_relative",
            ),
        ],
    )

    # 4. engine_workers.py 两处
    patch_file(
        os.path.join(VERL, "verl/workers/engine_workers.py"),
        [
            (
                "                self.loss_fn = partial(ppo_loss, config=actor_config)\n",
                "                self.loss_fn = partial(ppo_loss, config=actor_config)\n"
                "                self._eb_actor_config = actor_config\n",
                "_eb_actor_config = actor_config",
            ),
            (UPDATE_ACTOR_OLD, UPDATE_ACTOR_NEW, "eb_step_begin"),
        ],
    )

    # 5. 语法自检 + 生成补丁存档
    for f in (
        os.path.join(VERL, "verl/workers/config/actor.py"),
        os.path.join(VERL, "verl/workers/engine_workers.py"),
        os.path.join(VERL, "verl/workers/utils/eb_bridge.py"),
        os.path.join(VERL, "verl/workers/utils/eb_entropy_controller.py"),
    ):
        subprocess.run([sys.executable, "-m", "py_compile", f], check=True)
    print("  [ok]   py_compile 全部通过")

    os.makedirs(PATCH_DIR, exist_ok=True)
    subprocess.run(["git", "-C", VERL, "add", "-N", "verl/workers/utils/eb_bridge.py",
                    "verl/workers/utils/eb_entropy_controller.py"], check=False)
    diff = subprocess.run(["git", "-C", VERL, "diff"], capture_output=True, text=True).stdout
    patch_path = os.path.join(PATCH_DIR, "eb_grpo_verl_v0.8.0.patch")
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(diff)
    print(f"  [ok]   补丁存档 -> {patch_path}（{len(diff.splitlines())} 行）")

    # 6. 导入自检
    sys.path.insert(0, VERL)
    import verl.workers.utils.eb_bridge as bridge  # noqa: F401

    from verl.workers.config.actor import ActorConfig  # noqa: F401  (触发 dataclass 校验)

    print("EB-PATCH-INSTALL-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
