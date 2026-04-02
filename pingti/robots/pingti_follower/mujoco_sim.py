"""简化的 MuJoCo 仿真环境 - 仅需最小依赖."""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    raise ImportError("请安装 MuJoCo: pip install mujoco>=3.1.0")


class PingtiSimulator:
    """简化的 PingTi 机械臂仿真器.
    
    这是一个最小化的仿真实现，只依赖 MuJoCo。
    
    使用示例:
        >>> sim = PingtiSimulator(render=True)
        >>> obs = sim.reset()
        >>> for _ in range(1000):
        ...     action = np.random.uniform(-1, 1, 5)  # 5个关节
        ...     obs, reward, done = sim.step(action)
        >>> sim.close()
    """
    
    def __init__(
        self,
        model_path: str = None,
        render: bool = False,
        dt: float = 0.002,
        max_episode_steps: int = 1000,
    ):
        """初始化仿真器.
        
        Args:
            model_path: MuJoCo XML 模型路径
            render: 是否实时渲染
            dt: 仿真时间步
            max_episode_steps: 最大步数
        """
        self.render_enabled = render
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.viewer = None
        
        # 加载模型
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "pingti_simple.xml"
        
        print(f"加载模型: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        
        # 获取关节信息
        self.n_joints = self.model.njnt
        self.n_qpos = self.model.nq
        self.n_qvel = self.model.nv
        
        print(f"✓ 模型加载成功")
        print(f"  关节数: {self.n_joints}")
        print(f"  广义坐标: {self.n_qpos}")
        print(f"  广义速度: {self.n_qvel}")
    
    def reset(self) -> np.ndarray:
        """重置仿真环境.
        
        Returns:
            初始观测 (关节位置)
        """
        # 重置状态
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始位置 (5个关节)
        # [肩膀转动, 肩膀抬起, 肘部弯曲, 腕部弯曲, 腕部旋转]
        self.data.qpos[:5] = [0.0, 0.5, 0.0, -1.0, 0.0]
        
        # 前向仿真
        mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """执行一个仿真步骤.
        
        Args:
            action: 关节控制命令 [-1, 1] 范围的数组，长度为关节数
            
        Returns:
            obs: 观测 (关节位置)
            reward: 奖励值
            done: 是否完成
        """
        # 动作缩放和限制
        action = np.clip(action, -1.0, 1.0)
        control_signal = action * 10.0  # 缩放到力矩范围
        
        # 设置控制信号
        self.data.ctrl[:len(action)] = control_signal
        
        # 执行仿真步骤
        mujoco.mj_step(self.model, self.data)
        
        # 获取观测
        obs = self._get_obs()
        
        # 计算奖励 (简单的位置误差)
        reward = self._compute_reward()
        
        # 检查是否完成
        self.step_count += 1
        done = self.step_count >= self.max_episode_steps
        
        # 渲染
        if self.render_enabled:
            self.render()
        
        return obs, reward, done
    
    def _get_obs(self) -> np.ndarray:
        """获取观测.
        
        Returns:
            观测数组 [关节位置 + 关节速度]
        """
        pos = self.data.qpos[:self.n_qpos].copy().astype(np.float32)
        vel = self.data.qvel[:self.n_qvel].copy().astype(np.float32)
        return np.concatenate([pos, vel])
    
    def _compute_reward(self) -> float:
        """计算奖励.
        
        Returns:
            奖励值
        """
        # 简单奖励: 鼓励靠近初始位置
        target_pos = np.array([0.0, 0.5, 0.0, -1.0, 0.0])
        current_pos = self.data.qpos[:5]
        
        # 位置误差
        pos_error = np.sum((current_pos - target_pos) ** 2)
        reward = -0.1 * pos_error
        
        # 速度惩罚
        speed = np.sum(self.data.qvel[:5] ** 2)
        reward -= 0.01 * speed
        
        return float(reward)
    
    def get_joint_positions(self) -> np.ndarray:
        """获取关节位置."""
        return self.data.qpos[:5].copy()
    
    def set_joint_positions(self, positions: np.ndarray) -> None:
        """设置关节位置."""
        self.data.qpos[:5] = positions
        mujoco.mj_forward(self.model, self.data)
    
    def render(self) -> None:
        """渲染环境."""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
        
        self.viewer.sync()
    
    def close(self) -> None:
        """关闭仿真环境."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def __del__(self):
        """析构函数."""
        try:
            self.close()
        except:
            pass


class PingtiSimulatorEnv:
    """包装器类，提供 Gym 风格的接口."""
    
    def __init__(self, render: bool = False):
        self.sim = PingtiSimulator(render=render)
        self.action_space_shape = (5,)  # 5个关节
        self.observation_space_shape = (10,)  # 5个位置 + 5个速度
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """重置环境."""
        obs = self.sim.reset()
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行步骤."""
        obs, reward, done = self.sim.step(action)
        return obs, reward, done, {}
    
    def render(self) -> None:
        """渲染环境."""
        self.sim.render()
    
    def close(self) -> None:
        """关闭环境."""
        self.sim.close()