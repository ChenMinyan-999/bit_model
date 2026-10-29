# LORO 复现实验方案（Reproduction Protocol）

> 目标论文：*Improving the Data-efficiency of Reinforcement Learning by Warm-starting with LLM*（arXiv:2505.10861v1）
> 复现范围：CartPole / CliffWalking / FrozenLake / MountainCar 四个环境
> 文档性质：**可执行方案，本文档中的代码片段未在本机运行验证**（按需求只出方案）。所有接口名以实际安装版本为准，文中已标注需核对之处。
> 配套材料：[2505.10861v1_中文翻译.md](2505.10861v1_中文翻译.md)、[2505.10861v1_论文讲解.md](2505.10861v1_论文讲解.md)

---

## 0 先明确边界：能复现什么，不能复现什么

| 层次 | 内容 | 本方案能否复现 | 说明 |
|---|---|---|---|
| L1 机制 | "LLM 数据 → 离线预训练 → 在线微调" 的因果链 | ✅ 完全可以 | 与 LLM 是什么模型无关，只依赖 D 的分布特性 |
| L1 机制 | 论文两组核心消融（预训练 vs 仅混合；LLM 数据 vs 随机数据） | ✅ 完全可以 | 这是全文最硬的结论，也是最该先验证的 |
| L2 行为 | "LLM 策略非平凡但次优"、"CoT 必需" | ⚠️ 条件性 | 需要真 LLM；小模型很可能"完全不会玩"，无法落到"次优"区间 |
| L3 数值 | 论文图 2/3/4 的具体累积回报数字 | ❌ 不可能 | 论文用 Qwen2.5-7B/32B + 两块 H100；本机 4 GB 显存。LLM 调用是**逐 token 采样**，temperature=0.9，本身就不可逐位复现，换硬件/换版本还会漂移 |

**结论**：本方案把复现目标定义在 **L1（机制）+ L2（条件性行为）**，并给出 L1 的**替身方案**——用"脚本化次优策略"扮演 π_LLM。这不是偷懒：论文唯一的理论前提是**假设 1（覆盖）**，而覆盖是**数据分布的属性**，与产生数据的智能体是不是 LLM 无关。用脚本策略可以把"LLM 质量"变成可控自变量，反而比真 LLM 更适合做因果验证。

> 关键设计思想：**把"是不是 LLM"和"数据好不好"解耦**。真 LLM 实验同时变动了这两个因素，论文本身也没能分离它们（见 §11 风险 R3）。

---

## 1 复现目标与判定标准

### 1.1 五个可判定的命题

复现不是"抄数字"，而是检验下列命题是否成立。每条给出**通过判据**（先验设定，避免事后挑数据）。

| 编号 | 命题 | 通过判据 | 对应论文 |
|---|---|---|---|
| **P1** | LORO 优于纯在线 RL | 4 个环境中 ≥3 个，LORO 的累积回报均值高于在线 RL，且配对检验 p<0.10 | 图 2 |
| **P2** | 预训练是必要的，仅混合数据不够 | LORO > "仅混合" 在 ≥3 个环境成立，效应量 ≥10% | 图 3 / §5.2 |
| **P3** | 增益来自数据**质量**而非"多了点数据" | 用随机策略数据预训练**不优于**在线 RL；LORO > 随机数据预训练 在 ≥3 个环境成立 | 图 4 / §5.3 |
| **P4** | 在线微调能超越初始策略 | 在线阶段结束时 ≥ 初始 τ 回合表现的 1.5 倍（CartPole/FrozenLake 用成功率口径） | §5.1 |
| **P5** | 对 τ 不敏感 | τ ∈ {5,10,20} 三档的最终性能差异 < 20% 相对幅度 | 附录 A.4 |

> P5 是**证伪性最强**的一条：论文声称"数据量无关"，但同时承认 CartPole 上 τ=10 会崩。若本复现发现 τ 强敏感，则论文的"不敏感"结论应被标记为**不成立**。

### 1.2 明确不判定的内容

- 不判定"LLM 的推理能力是否有用"——需要真 7B 以上模型，本机做不到。
- 不判定"4× 提升"——该数字依赖论文那个未调参的 DDQN 基线，本复现若把基线调好，比值必然缩小（这正是我们要**量化**的东西，见 §5.3）。

---

## 2 环境规格

### 2.1 四个环境的确定性参数

| 环境 | Gymnasium ID | 观测 | 动作 | 每步奖励 | 终止 | H（回合上限） | T（总回合） |
|---|---|---|---|---|---|---|---|
| CartPole | `CartPole-v1` | 4 维连续 | 2 离散 | +1 | 位置 \|x\|>2.4 或 \|θ\|>0.2095 | 200 | 150 |
| CliffWalking | `CliffWalking-v0` | 1 维（位置 0..47） | 4 离散 | −1；悬崖 −100 并重置位置 | 到达 (3,11) | 200 | 200 |
| FrozenLake | `FrozenLake-v1`（`is_slippery=True`，默认 4×4 地图） | 1 维（位置 0..15） | 4 离散 | 0；目标 +1 | 到达目标或掉洞 | 100 | 150 |
| MountainCar | `MountainCar-v0` | 2 维连续 | 3 离散 | −1 | 到达右侧山顶 | 200 | 300 |

**H 与 T 的取值来源**（论文只给了 T，H 是从论文数字反推的，需在报告中注明这是推断）：

- CartPole：图 2 纵轴上限 200，且 `On-policy` 累计 24411 < 150×200=30000 → H=200（**注意 CartPole-v1 默认 500，必须用 `TimeLimit(200)` 覆盖**）；
- CliffWalking：提示词明写"200 actions"；随机基线 −412659 只有在"每次掉崖 −100 但回合继续"时才可能达到，与 gymnasium 实现一致；
- MountainCar：提示词明写"episode is 200"；随机基线 −60000 = 300×(−200) ✓ 完全吻合；
- FrozenLake：默认 100；`On-policy` 累计 75（150 回合成功一半）合理。

> ✅ **一致性自检**：MountainCar 的 −60000 和 CartPole 的 30000 上界能同时对上，说明 H/T 推断正确。跑之前请先跑一次随机基线与这两个数字对齐，对不上就说明环境配置错了——这是最便宜的**环境校验器**。

### 2.2 观测 → 文本 的关键实现

论文用 LlamaGym 式的文本包装器。四个环境的文本化规则如下（**必须与论文提示词口径一致，否则 LLM 无法理解**）：

```python
# env/text_wrapper.py（示意，需按 gymnasium 版本核对）
import numpy as np

def describe(env_name, obs, env, history=None, prev=None):
    if env_name == "CartPole":
        x, v, th, thdot = obs
        return (f"The cart is positioned at {x:.3f}, with a velocity of {v:.2f} "
                f"towards the {'right' if v>=0 else 'left'}. "
                f"The pole is tilted at {abs(th):.2f} radians, rotating at {abs(thdot):.2f} "
                f"radians per second towards the {'right' if thdot>=0 else 'left'}.")
    if env_name == "CliffWalking":
        row, col = divmod(env.unwrapped.s, 12)
        return f"You are at location ({row}, {col}) in the grid world."
    if env_name == "FrozenLake":
        row, col = divmod(env.unwrapped.s, 4)
        return f"You are at row {row}, column {col}."
    if env_name == "MountainCar":
        p, v = obs
        return (f"The car is positioned at {p:.3f}, with a velocity of {abs(v):.3f} "
                f"towards the {'right' if v>=0 else 'left'}.")
```

**注意两个"信息不对称"陷阱**（论文原文如此，复现时必须显式记录，见 §11 R2）：

1. **FrozenLake / CliffWalking 需要"环境历史"**：把已发现的信息摘要进提示词（如"Reward 0.0 at locations: (0,0),(0,1)…"）。**这份历史只给 LLM，RL 智能体的在线阶段拿不到。** 复现时务必做成开关 `use_history_prior: true/false`，并跑两组——这是**分离"LLM 常识"与"人工先验"的唯一办法**。
2. **CartPole 的"朝向"措辞**：上面代码把符号转成了"left/right"，论文提示词正是这么写的（原文明写 `velocity of 0.04 towards the right`）。这是无害的表示层转换，不属于信息泄漏，但要在报告中说明。

---

## 3 π_LLM 的两种实现（择一，或都做）

### 3.1 方案 A：脚本化"次优策略"（CPU 可跑，推荐先做）

目的：**人为构造一个"覆盖够、质量不够"的 π_LLM**，直接检验假设 1。四个策略都设计成"不犯错但不高明"——对应论文中 LLM 的典型表现。

```python
# policies/scripted.py（示意）
import numpy as np

class ScriptedPolicy:
    """参数化的'有用但次优'策略。quality 越低越差，用于扫覆盖-性能曲线。"""
    def __init__(self, name, quality=0.8, rng=None):
        self.name, self.q, self.rng = name, quality, rng or np.random.default_rng(0)

    def __call__(self, obs, env=None, info=None):
        # 以 quality 概率走启发式，否则随机 —— 模拟 LLM 的偶发失误
        if self.rng.random() > self.q:
            return self.rng.integers(0, self.n_actions)
        return self._heuristic(obs, env)

    def _heuristic(self, obs, env):
        if self.name == "CartPole":
            # PD 控制：角度+角速度的线性反馈（比最优 LQR 差，但没有饱和保护）
            x, v, th, thdot = obs
            return 1 if (th + 0.5 * thdot) > 0 else 0      # 1=右推
        if self.name == "CliffWalking":
            row, col = divmod(env.unwrapped.s, 12)
            if row > 0 and col < 11: return 0              # 上
            if col < 11:             return 1              # 右
            return 2                                       # 下（到终点）
        if self.name == "FrozenLake":
            return self._frozenlake_partial_map(env)
        if self.name == "MountainCar":
            p, v = obs
            return 2 if v > 0 else 0                       # 顺着速度方向加速（能量法，慢）
```

四个策略的"次优"设计要点（**这是本方案最关键的设计**）：

| 环境 | 次优在哪里 | 为什么这样设计 |
|---|---|---|
| CartPole | 无饱和保护、增益固定 → 平均 ~120 步倒 | 落在"能撑一会儿但远不到 200"的区间，正好是 LLM 的典型水位 |
| CliffWalking | 走"上→右→下"的安全路径，但偶尔因 quality<1 走错一格 | 覆盖了最优路径（满足假设 1），但多走步数（次优）。**绝不允许掉崖**，否则不再是"高质量数据" |
| FrozenLake | **只知道第 0 行与第 0 列**的地图（部分可观测先验） | 精确模拟"LLM 看得到部分提示词但推理不全"，覆盖度中等——这是四个里最能体现覆盖假设的 |
| MountainCar | 能量法但没有"先退后冲"的蓄势 | 能到终点但 ~180 步（最优 ~130），次优但轨迹在同一条流形上 |

**quality 参数化很关键**：把 `quality` 从 0.3 扫到 1.0，就能画出**"π_LLM 质量 → LORO 最终性能"**的曲线。论文只给了 2 个点（7B、32B），本复现能给一条完整曲线，这是**超出原文的增量贡献**。

### 3.2 方案 B：真 LLM（有 GPU/API 时）

保留方案 A 的接口，把 `__call__` 换成一次模型调用。**提示词必须逐字使用原文**（下面是 4 个环境的英文原文，可直接用；中文译文见翻译稿附录 C）：

<details>
<summary>展开：四个环境的原始英文提示词</summary>

**CliffWalking**
```
You are an expert-level game player. Cliff walking is a task in which you control a player
navigating a '4x12' grid world. The ('x', 'y') coordinate indicates the position at row 'x' and
column 'y'. The player needs to find a goal location while avoiding cliffs . The player can
choose from 4 actions: move up, move right, move down, or move left. If the player takes an
action at ('x', 'y'), he tries to move to ('a', 'b'). Rules:
• 1. If ('a', 'b') is a cliff, the player incurs a large penalty of -100, and is reset to the
starting position.
• 2. If ('a', 'b') is safe or towards the grid boundary, results in a small penalty of -1. If
('a', 'b') is outside the grid's boundaries, it does not change position but still receive
the -1 penalty.
• 3. The game ends when the ('a', 'b') is the goal or 200 actions are performed. The
goal is to navigate from the starting point to an target, while avoiding the cliff, in as
few steps as possible.
Type a number to indicate the action. For current position ('x', 'y'), the action means the
player try to step into the next position. Type '1' to move up, which means trying to step into
('x-1', 'y'), '2' to move right, which means ('x', 'y+1'), '3' to move down, which means
('x+1', 'y'), or '4' to move left, which means ('x', 'y-1'). Ensure you only provide the action
number from the valid action list, i.e., [1, 2, 3, 4]. Environment history: Cliff: Reward -100
at locations: (3, 0). Reward -1 at locations: (2, 0), (3, 0), (1, 0). Previous location: (1, 0),
previous action: 1, previous reward: -1. Return the action at the end of your answer without
the target's location.
```
User: `You are at location (2, 0) in the grid world.` + `Think step by step.`

**CartPole**
```
You are an expert-level game player. In the CartPole game, you control a cart that moves
along a horizontal track. There is a pole standing upright on the cart. The goal of the game is
to keep the pole balanced upright by moving the cart left or right. The game ends if the pole
tilts too far from the vertical position or if the cart moves too far from the center of the track.
The longer you can keep the pole balanced, the higher your score. Note that when the Cart
Position is out of the (-2.4, 2.4) zone or the Pole Angle is out of the zone (-.2095, .2095), the
round ends and the game is lost. The goal is to keep the pole balanced upright for as long as
possible.
Type '1' to push the cart to the left or '2' to push the cart to the right. Ensure you only provide
the action number from the valid action list, i.e., [1, 2].
```
User: `The cart is positioned at 0.006, with a velocity of 0.04 towards the right. The pole is tilted at 0.02 radians, rotating at 0.02 radians per second towards the right.` + `Think step by step.`

**MountainCar**
```
You are an expert-level game player. In the Mountain Car game, you control a car placed
stochastically at the bottom of a sinusoidal valley. The only possible actions are the accelera-
tions that can be applied to the car in either direction. The goal of the game is to strategically
accelerate the car to reach the goal state on top of the right hill as quickly as possible. The
episode ends if either the car reaches the goal position on top of the right hill or the length of
the episode is 200. The goal is to reach the flag placed on top of the right hill as quickly as
possible.
Type '1' to accelerate to the left, '2' to not accelerate, or '3' to accelerate to the right. Ensure
you only provide the action number from the valid action list, i.e., [1, 2, 3].
```
User: `The car is positioned at -0.541, with a velocity of 0.000 towards the left.` + `Think step by step.`

**FrozenLake**
```
You are an expert-level game player. In the FrozenLake game, the player starts at the start
position of the grid and tries to reach the goal position. There are holes which the player
must avoid. The frozen lake is slippery, meaning that the player might not always move in
the intended direction. The game ends when the player reaches the goal or falls into a hole.
The goal is to navigate across the frozen lake and reach the goal position without falling into
any holes. For current position ('x', 'y'), the action means the player try to step into the next
position. The possible actions are:
• 1: Move left, which means ('x', 'y-1'),
• 2: Move down, which means ('x+1', 'y'),
• 3: Move right, which means ('x', 'y+1'),
• 4: Move up, which means trying to step into ('x-1', 'y').
Ensure you only provide the action number from the valid action list, i.e., [1, 2, 3, 4]. Do
not return the target's coordination. Environment history: Reward 0.0 at locations: (0, 0).
Previous location: (0, 0), previous action: 1, previous reward: 0.0. Return the action at the
end of your answer without the target's location.
```
User: `You are at row 0, column 0.` + `Think step by step.`

> ⚠️ 注意 FrozenLake 的动作编号映射与其他三个环境**不同**（1=左，2=下，3=右，4=上），而 CliffWalking 是 1=上、2=右、3=下、4=左。写错这个映射，实验必然失败，且很难从曲线上看出来。

</details>

**解码参数**（论文附录 B）：`max_new_tokens=2000, top_p=0.6, top_k=0, temperature=0.9`。
**动作提取**：取回答中**最后一个数字**；动作**从 1 开始编号**（原文说明：LLM 对 0 有偏置）。
**失败处理**（论文没写，必须自己定）：解析不到合法动作时记 `parse_fail`，该步**回退为随机动作**并计数。报告中须给出 `parse_fail_rate`——若 >5%，说明提取逻辑需要修（论文自己也承认"有相当多提取失败"）。

---

## 4 LORO 的实现规格

### 4.1 主流程

```python
# loro/train.py（示意）
def loro(env_fn, pi_llm, tau, T, alg_factory, keep_offline, n_pretrain_steps):
    D = []                                   # 离线数据集
    ep_rewards = []
    for t in range(tau):                     # 阶段 1：LLM 采集
        obs, _ = env.reset()
        for h in range(H):
            a = pi_llm(obs, env)             # 关键：动作来自 π_LLM
            nobs, r, term, trunc, _ = env.step(a)
            D.append((obs, a, r, nobs, float(term)))   # 需存 done 标志
            obs = nobs
            if term or trunc: break
        ep_rewards.append(ep_return)

    algo = alg_factory()
    algo.fit(make_mdp_dataset(D), n_steps=n_pretrain_steps)   # 阶段 2：离线预训练

    if not keep_offline:
        D = []                               # ← 论文附录 A.1.2 的说法：在线只用在线数据
    for t in range(tau, T):                  # 阶段 3：在线微调
        obs, _ = env.reset()
        for h in range(H):
            a = algo.predict(obs)[0]         # 或直接 algo.fit_online(env)
            ...
            D.append(...)
        algo.update(D)                       # 见 §4.2：两种更新方式都要试
    return ep_rewards
```

### 4.2 必须先解决的一处论文矛盾（本方案的核心决策）

论文对"在线阶段是否保留离线数据 D"**自相矛盾**：

- **算法 1 第 14 行**：`D ← D ∪ {(s,a,r)}`，且第 12 行 `Alg(D, π_prev)` → **保留**；
- **附录 A.1.2**：*"we collected data to pre-train a policy and then only used online data in the online learning process"* → **丢弃**；
- **§5.2 的解释**："预训练让 LORO 对高质量数据**过拟合**，这种保守反而有益" → 这个解释只有在**保留**时才成立（丢弃后模型会被在线数据迅速覆盖，无所谓过拟合）。

**方案：把它变成一个显式实验因子，而不是去猜。**

| 变体 | 在线阶段数据 | 预期 | 判定意义 |
|---|---|---|---|
| `LORO-KEEP` | 保留 D（符合算法 1） | 若 §5.2 的"过拟合有益"解释成立，此变体应最好 | 若最好 → 论文的解释正确，附录 A.1.2 是笔误 |
| `LORO-DROP` | 丢弃 D（符合附录 A.1.2） | 若与 `LORO-KEEP` 无差异，则 §5.2 的解释不成立 | 若两者无差异 → 论文的机制解释需要重写 |

这一条是本复现**最有价值的产出**：论文自己没说清，我们能给出答案。

### 4.3 RL 算法与超参

论文用 d3rlpy 默认值 + 少量覆盖。**下表"来源"列区分哪些是论文给的、哪些是我补的默认值**（论文只列了 6 个，其余必须自己定，务必记录）。

| 参数 | 值 | 来源 |
|---|---|---|
| 算法（离散动作） | DDQN（d3rlpy `DoubleDQN`） | 论文 A.1.2 |
| 算法（连续动作） | SAC | 论文 A.1.2（本复现 4 个环境全离散，可用不到） |
| batch_size | 256 | 论文 |
| buffer_size | 100,000 | 论文 |
| learning_rate | 5e-5 | 论文 |
| gamma | 0.99 | 论文 |
| target_update_interval | 1,000 | 论文（**单位是"梯度步"，不是环境步，容易踩坑**） |
| epsilon（探索） | 0.1 | 论文（对应 d3rlpy 的 `explorer`） |
| n_pretrain_steps | 1000 | 论文 |
| n_steps（N-step return） | 1 | *默认* |
| update_interval | 1 | *默认* |
| q_func hidden | [64,64]（或 256） | *默认*，需固定 |
| 每环境 τ | 10 | 论文 |
| seeds | 5（建议 10） | 论文 5 |

> **d3rlpy 版本坑**：v1 与 v2 的 API 差异很大（`fit_online` 的签名、`algos.DoubleDQN` vs `algos.DoubleDQNConfig` 的构建方式、`update_start_step` 参数）。请**固定版本**并在 `requirements.txt` 里锁死，例如 `d3rlpy==2.6.0`（以实际可安装为准）。
> **Windows 坑**：d3rlpy 在 Windows 上可能需要编译依赖（Cython/C++）。若安装失败，**退回方案 4.4**，不要在这上面耗一天。

### 4.4 Plan B：150 行手写 DDQN

复现的目标是"离线预训练 + 在线"这条管线，换掉 RL 库不改变结论。手写 DDQN 反而更可控（可精确控制"离线预训练步数"与"在线更新频率"）。

```python
# rl/ddqn.py（示意，约 150 行）
class Replay:
    def __init__(self, cap): self.buf, self.cap, self.i = [], cap, 0
    def add(self, tr): ...
    def sample(self, n, rng): ...

class DDQN:
    def __init__(self, obs_dim, n_act, lr=5e-5, gamma=0.99,
                 target_update=1000, batch=256, hidden=(64, 64)):
        self.q, self.qt = MLP(obs_dim, n_act, hidden), MLP(obs_dim, n_act, hidden)
        self.qt.load_state_dict(self.q.state_dict())
    def act(self, s, eps=0.1, rng=None):     # 在线阶段的探索率
        if rng.random() < eps: return rng.integers(self.n_act)
        return int(self.q(torch.as_tensor(s, dtype=torch.float32)).argmax())
    def pretrain_offline(self, D, n_steps):
        """在固定小数据集 D 上反复采样 —— 这正是'过拟合小数据'的实现"""
        for k in range(n_steps):
            self._grad_step(D, bootstrap_from_offline=True)
    def update_online(self, buffer, step):
        ...
```

两条实现纪律：
1. **离线阶段用固定集 D 反复采样**（`pretrain_offline`），这正是论文说的"过拟合"；不要用"过一遍就完"。
2. **终止态 bootstrap 要置零**（`target = r + gamma*(1-done)*qt(s')`），否则 CliffWalking 学会"故意跳崖"——这是最常见的错误，且只会让曲线难看、不会报错。

---

## 5 实验矩阵与基线

### 5.1 主实验（对齐论文图 2）

| ID | 方法 | 数据来源 | 说明 |
|---|---|---|---|
| E0 | Random | 均匀随机 | **环境校验器**，见 §2.1 |
| E1 | Online-RL | 在线自采 | 对应论文"On-policy"（其实是 DDQN，见 §11 R1） |
| E2 | Online-RL-tuned | 在线自采 | **论文没有**：把 lr 提到 1e-3、batch 64、加 reward scaling，作为"公平基线" |
| E3 | π_LLM only | 脚本策略 | 对应"纯 LLM 策略"，只跑 τ 回合 |
| E4 | **LORO-KEEP** | 脚本策略 → 在线 | 主方法 |
| E5 | **LORO-DROP** | 脚本策略 → 在线 | 主方法的另一实现 |

### 5.2 消融（对齐论文图 3 / 图 4）

| ID | 消融 | 数据来源 | 检验 |
|---|---|---|---|
| A1 | 仅混合、不预训练 | 脚本策略 | P2 |
| A2 | 预训练 + 随机策略数据 | Random | P3 |
| A3 | 预训练 + 在线 RL 数据 | 在线 DDQN 采 10 回合 | P3（区分"高质量"与"任意数据"） |
| A4 | 预训练 + **专家级**脚本数据 | quality=1.0 | **论文没有**：LORO 的增益会不会只是"任何像样的先验数据都行"？ |
| A5 | τ ∈ {5,10,20} | 脚本策略 | P5 |
| A6 | 预训练步数 ∈ {1000, 3000} | 脚本策略 | 附录 A.3 |
| A7 | π_LLM quality ∈ {0.3,0.5,0.7,0.85,1.0} | 脚本策略 | **论文没有**：画"覆盖—性能"曲线 |
| A8 | 无环境历史（`use_history_prior=false`） | 脚本策略 | **论文没有**：量化人工先验的贡献（§11 R2） |

- `A1` 与 `E4` 的差值 = 预训练的贡献（论文的核心主张）；
- `A2`/`A3` 与 `E4` 的差值 = 数据质量的贡献（论文的核心主张）；
- **`A4` 是审稿人一定会问的那个对照**：如果专家数据也差不多，那 LORO 的故事就从"LLM 有常识"降级为"给 RL 喂点像样数据就行"——两者对领域的意义完全不同。

### 5.3 样本效率的度量（不要只报累积回报）

论文只报"累积回报"，这个指标**对探索期的坏动作极其敏感**，容易被基线的一次崩盘放大（CliffWalking 的"4×"就是这么来的）。建议同时报：

1. **AUC**：学习曲线的归一化面积（越接近最优策略上界越好）；
2. **Episodes-to-threshold**：达到"最优性能的 90%"所需回合数（最优性能用 E2 收敛后的值）；
3. **Regret**：`Σ(最优每回合回报 − 实际每回合回报)`；
4. **最终 20 回合均值**（排除前期噪声）；
5. **成本**：CPU 秒 / GPU 秒 / LLM 调用次数 / token 数 ← 论文完全没报，**必须补**。

> 第 5 项是本复现的"立场"：如果 LORO 省了 4 倍环境交互，却花了 1000 倍 LLM 算力，那它在工程上是亏的。请把两个数放在同一张表里。

---

## 6 配置与目录结构

### 6.1 目录

```
loro-repro/
├── configs/
│   ├── base.yaml
│   └── exp/{cartpole,cliffwalking,frozenlake,mountaincar}.yaml
├── envs/            # 文本包装、历史摘要、TimeLimit 修正
├── policies/        # scripted.py, llm.py（共用 __call__ 接口）
├── rl/              # ddqn.py（或 d3rlpy 适配层）
├── loro/            # train.py, offline.py, online.py
├── metrics/         # auc, threshold, regret, coverage
├── scripts/         # run_all.ps1, sweep.py
├── results/         # runs/<exp>/<env>/<seed>/{metrics.json,curve.csv}
└── report/
```

### 6.2 单一配置入口（示例）

```yaml
# configs/base.yaml
env:
  name: CartPole          # CartPole | CliffWalking | FrozenLake | MountainCar
  horizon: 200
  total_episodes: 150
  use_history_prior: true  # 只影响 π_LLM 的提示词
llm:
  backend: scripted        # scripted | local | api
  quality: 0.8             # 仅 scripted 生效
  model: Qwen/Qwen2.5-7B-Instruct
  temperature: 0.9
  top_p: 0.6
  top_k: 0
  max_new_tokens: 2000
loro:
  tau: 10
  n_pretrain_steps: 1000
  keep_offline: true       # ← §4.2 的开关
  pretrain_source: llm     # llm | random | online | expert
rl:
  algo: ddqn
  lr: 5.0e-5
  gamma: 0.99
  batch: 256
  buffer: 100000
  target_update: 1000
  eps: 0.1
  hidden: [64, 64]
run:
  seeds: [0, 1, 2, 3, 4]
  out: results/runs
```

### 6.3 运行

```powershell
# 0) 环境（建议独立 venv，别装进系统 Python）
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install "gymnasium==0.29.1" "numpy" "pandas" "matplotlib" "pyyaml"
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU 够用
# 可选：pip install d3rlpy==2.6.0   ← 装不上就跳过，用 rl/ddqn.py

# 1) 环境校验（必做，30 秒）
python -m scripts.sanity            # 断言：CartPole≈3079、MountainCar=-60000、CliffWalking≈-412659 量级

# 2) 主实验
python -m loro.train --config configs/exp/cartpole.yaml --exp E4

# 3) 全量（4 环境 × 8 配置 × 5 seed）
pwsh scripts/run_all.ps1 -Parallel 4
```

> ⚠️ `scripts.sanity` 的数值来自论文的随机基线，**只做量级校验**（±20%），不要当成精确目标。

---

## 7 评测协议与统计

| 项目 | 规定 |
|---|---|
| 随机种子 | 5 个（0–4）；**建议 10 个**，否则配对检验几乎没有功效 |
| 报什么 | 均值 ± 标准误（SE），论文口径；同时给中位数与 IQR（对小样本更稳健） |
| 比较方法 | 同种子**配对**比较；单侧 Wilcoxon signed-rank（n=5 时最小 p=0.0625，只能"接近显著"，必须在报告中说明） |
| 多重比较 | 4 环境 × 8 配置 → 用 Holm–Bonferroni 校正，否则必然出现假阳性 |
| 曲线平滑 | **不平滑**做主图；平滑仅用于附录（论文用"Smoothed"，掩掉了噪声） |
| 报告纪律 | 预先登记 P1–P5 的判据（§1.1），**不得在看完结果后修改** |

**功率预警（诚实提示）**：n=5、环境方差又大时，你很可能看到"方向一致但不显著"。此时正确做法是**增加种子到 20–30**（CPU 跑得动）；而不是挑环境报。若只能 n=5，请把结论写成"趋势一致，未达统计显著"。

---

## 8 算力与时间预算

| 配置 | 单次运行 | 说明 |
|---|---|---|
| E1 在线 DDQN，200 回合 × 200 步 | ~1–2 min（CPU） | 4 万步，小 MLP |
| E4 LORO，含 1000 预训练步 | ~2–3 min（CPU） | 多一次离线拟合 |
| 方案 B 真 LLM（7B，4-bit，CPU 卸载） | **4–10 h/环境** | 2000 次调用 × ~100 token，i5 上 3–6 tok/s → 不可行 |
| 方案 B 真 LLM（API） | 20–60 min/环境 | 受速率限制 |

**全量估算（方案 A）**：4 环境 × 8 配置 × 5 seed × 2.5 min ≈ **7 小时**（4 并行 → ~2 h）。
**方案 B 在本机不可行**，这是物理限制，不是实现问题。

---

## 9 结果记录与验收表

跑完后请填这张表（括号内为论文参考值，**不作为达标线**）：

| 环境 | 指标 | E1 在线 | E4 LORO | 论文 E1 | 论文 E4 | 通过 P1? |
|---|---|---|---|---|---|---|
| CartPole | 累积回报 | | | 24411 | 27078 | ☐ |
| CliffWalking | 累积回报 | | | −41858 | −10436 | ☐ |
| FrozenLake | 累积回报 | | | 75 | 94 | ☐ |
| MountainCar | 累积回报 | | | −57616 | −45453 | ☐ |

消融表：

| 环境 | E4 LORO | A1 仅混合 | A2 随机数据 | A3 在线数据 | A4 专家数据 | 论文 A1 | 论文 A2 |
|---|---|---|---|---|---|---|---|
| CartPole | | | | | | 23144 | 25351 |
| CliffWalking | | | | | | −34769 | −581684 |
| FrozenLake | | | | | | 62 | 65 |
| MountainCar | | | | | | −59752 | −516516 |

**验收逻辑**（按顺序判）：

1. **环境校验通过？**（E0 量级对吗）→ 不对，先修环境，别往下走。
2. **E4 > E1 吗？**（P1）→ 不成立，说明替换 π_LLM 后机制失效，回到 §10 归因。
3. **E4 > A1 吗？**（P2）→ 这是论文最核心的主张。若 `A1 ≈ E4`，则"预训练"这一步是多余的，LORO 退化为普通 offline-to-online。
4. **E4 > A2、A3 吗？**（P3）→ 若 `A2 ≈ E4`，则"质量"不重要，只需"有数据"——论文的解释就错了。
5. **A4 与 E4 差多少？** → 差距小 = 增益来自"任何像样先验"；差距大 = 增益确实来自己 π_LLM 的特定分布。

---

## 10 偏差归因决策树（结果对不上时用）

```
LORO 没有优于在线 RL？
├─ E0 随机基线对不上论文 → 环境/奖励/终止条件错了（最常见：CartPole 没设 TimeLimit(200)）
├─ E1 收敛得比论文好很多（如 CliffWalking 到 -1300 附近）
│   └─ 说明论文基线欠调参 → 你要复现的是"差距缩小"，这是**有效发现**，不是失败
├─ π_LLM 的 D 覆盖不到最优轨迹
│   └─ 检查 A7：quality 曲线在低 quality 处崩塌 → 假设 1 的边界，直接报出来
└─ 预训练步数过多把网络训崩（DQN + 小数据 = 发散）
    └─ 降 lr、加 target 更新间隔、检查终止态 bootstrap 是否为 0

消融结果反了（A1 ≈ E4）？
├─ keep_offline=False 且在线更新太快 → 离线权重被冲掉，等价于没预训练（这就是 §4.2 的矛盾！）
├─ 你的 A1 实现里其实也做了预训练？→ 核对 A1 的定义：只混合、不 fit
└─ 小数据集 + 高 lr → 预训练"学到了"但立刻被覆盖，属于实现问题

FrozenLake 全 0？
└─ is_slippery=True 时 4×4 掉洞概率高，150 回合可能一次都没成功 → 这是正常的，论文也是 0/75
```

---

## 11 已知风险与踩坑清单

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| **R1** | 论文的"On-policy"基线实为 DDQN/SAC（离策略算法），且超参未调 | "4×"被夸大 | 加 E2 调参基线；同时报 E1/E2，把差距拆开 |
| **R2** | **信息不对称**：FrozenLake/CliffWalking 的"环境历史"只给 LLM | 增益被误归因于"LLM 常识" | `A8` 无历史对照；报告中明确写出这部分先验是人工注入的 |
| **R3** | "是不是 LLM"与"数据好不好"被绑在一起 | 无法分离两个因素 | 用 `A4`（专家脚本）做第四组对照 |
| **R4** | 算法 1 与附录 A.1.2 矛盾（D 是否保留） | 结果不可解释 | §4.2 双变体 `KEEP`/`DROP` |
| **R5** | 假设 1 从未被测量 | 结论靠循环论证 | 加覆盖率指标（§12.1），画 quality–性能曲线 |
| **R6** | CartPole 的 H 是推断值（论文只给 T） | 累积回报无法与论文对齐 | 用 30000 上界反推校验；报告注明"推断" |
| **R7** | FrozenLake 动作映射与其他环境不同 | 静默失败 | 单元测试：每个环境断言 `action→(dx,dy)` 映射 |
| **R8** | 终止态 bootstrap 未置零 | CliffWalking 学会跳崖 | 单元测试：手写一个"必死"轨迹检查 Q 值 |
| **R9** | 5 个种子功效不足 | 假阴性 | 提到 10–20；或改用配对 + 效应量 |
| **R10** | 论文未报 LLM 调用成本 | 效率结论不完整 | 强制记录 token/调用/秒，与样本节省并列 |
| **R11** | d3rlpy 在 Windows 可能装不上 | 阻塞 | Plan B 手写 DDQN（§4.4） |
| **R12** | 曲线平滑掩盖噪声 | 过度乐观 | 主图不平滑 |

---

## 12 两个"超出原文"的增量实验（建议做，成本很低）

### 12.1 覆盖率度量（补上论文的最大缺口）

论文的核心假设从未被量化。做法：

1. 用 E2 训练一个**收敛的** DDQN，取 greedily 跑 100 回合，得到最优策略访问集 `S*`；
2. 对连续观测做离散化（CartPole：4 维 × 6 桶；MountainCar：位置 20 桶 × 速度 20 桶；网格世界：精确位置）；
3. 计算 `coverage = |D ∩ S*| / |S*|`（D 为 10 回合 LLM 数据）；
4. 画 **coverage vs 最终回报** 散点（用 A7 的 quality 扫点）。

若两者强相关 → **假设 1 得到实证支持**，这是对论文的实质补强；若无相关 → 假设 1 不是真正的机制，LORO 的有效性另有原因（值得单独写一段讨论）。

### 12.2 成本-效率 Pareto 曲线

把横轴从"环境步数"换成"等效总成本"：

```
cost = 环境交互步数 × c_env + LLM调用次数 × c_llm + 预训练梯度步 × c_grad
```

`c_llm` 取一次 7B 调用（约 2000 token）的实测墙钟时间或 API 价格。结论会很尖锐：**当 c_llm / c_env > 约 100 时，LORO 在总成本上并不划算**——这个阈值本身就很有报告价值。

---

## 13 交付物清单

| 文件 | 内容 |
|---|---|
| `README.md` | 一键复现步骤、版本锁、与本文档的对应关系 |
| `results/summary.csv` | 所有 (exp, env, seed, 指标) 长表 |
| `report/fig2_main.png` | 复现图 2（主结果） |
| `report/fig3_pretrain.png` | 复现图 3（预训练消融）+ `KEEP/DROP` 双变体 |
| `report/fig4_data.png` | 复现图 4（数据质量）+ 新增 `A4` 专家对照 |
| `report/coverage.png` | 覆盖率–性能曲线（增量贡献） |
| `report/cost_table.md` | 样本节省 vs 算力开销（增量贡献） |
| `report/deviations.md` | **所有与论文不一致之处的清单**，含判据 P1–P5 的结论 |

> `deviations.md` 是复现工作里最容易被忽略、也最该写的一份：**"没复现出来"往往比"复现出来了"更有信息量。**

---

## 14 一句话执行建议

**先花 30 分钟把 `E0` 和 `E1` 跑通并与论文的量级对齐，再花 2 小时做 `E4` 与 `A1`（P1、P2）。** 这两组如果成立，论文的机制就站住了；`A2/A4` 决定它的故事能不能讲得更大。别一上来就搭 LLM——那是最贵、也最不可控的一环。
