"""与原文（LORO-legacy）逐字一致的 CartPole 提示词与动作解析。

来源：env/classic_control/cartpole_translator.py 与 env/translation_agent.py
（arXiv:2505.10861v1 官方实现）。改动会直接改变 π_LLM 的行为，请勿随意"润色"。
"""

# 原文把三段描述用 f-string 拼成 system prompt：
#   f"You are an expert-level game player. {describe_game()} {describe_goal()} {describe_action()}"
GAME_DESC = (
    "In the CartPole game, you control a cart that moves along a horizontal track. There is a pole "
    "standing upright on the cart. The goal of the game is to keep the pole balanced upright by moving the "
    "cart left or right. The game ends if the pole tilts too far from the vertical position or if the cart "
    "moves too far from the center of the track. The longer you can keep the pole balanced, the higher "
    "your score.Note that when the Cart Position is out of the (-2.4, 2.4) zone or the Pole Angle is out "
    "of the zone (-.2095, .2095), the round ends and the game is lost. "
)
GOAL_DESC = "The goal is to keep the pole balanced upright for as long as possible."
ACTION_DESC = (
    "Type '1' to push the cart to the left or '2' to push the cart to the right. Ensure you only provide "
    "the action number from the valid action list, i.e., [1, 2]."
)

SYSTEM_PROMPT = (
    f"You are an expert-level game player. {GAME_DESC} {GOAL_DESC} {ACTION_DESC}"
)

THINK_SUFFIX = "Think step by step."


def format_observation(obs) -> str:
    """4 维观测 -> 原文的自然语言描述（数值格式化必须一致：.3f / .2f）。"""
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = obs
    cart_direction = "right" if cart_velocity > 0 else "left"
    pole_direction = "right" if pole_angular_velocity > 0 else "left"
    return (
        f"The cart is positioned at {cart_position:.3f}, with a velocity of {abs(cart_velocity):.2f} "
        f"towards the {cart_direction}. "
        f"The pole is tilted at {abs(pole_angle):.2f} radians, rotating at {abs(pole_angular_velocity):.2f} "
        f"radians per second towards the {pole_direction}."
    )


def user_message(obs) -> str:
    """单步 user 消息。原文在 COT 模式下把 'Think step by step.' 追加到消息末尾。"""
    return f"{format_observation(obs)}\n{THINK_SUFFIX}"


def parse_action(response: str) -> int:
    """原文 CartPoleAgent.extract_action 的可执行版本，返回 0/1。

    规则（照抄）：
      1. 取整段文本里最后一个数字字符；
      2. 若该字符是 '1'/'2' -> 用它；
      3. 否则看文本里有没有 left/right 关键词；
      4. 都不行 -> 返回 1（即"向左"，不是随机）。
    """
    digits = [c for c in response if c.isdigit()]
    out = -1
    if len(digits) == 0 or digits[-1] not in ("1", "2"):
        low = response.lower()
        if "left" in low:
            out = 1
        elif "right" in low:
            out = 2
    else:
        out = int(digits[-1])
    if out not in (1, 2):
        out = 1
    return out - 1  # LLM 对 0 有偏置，提示词用 1/2 编号，再转回 0/1
