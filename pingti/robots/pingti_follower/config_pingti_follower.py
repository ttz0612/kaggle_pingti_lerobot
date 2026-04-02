from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("pingti_follower")
@dataclass
class PingtiFollowerConfig(RobotConfig):
    """PingTi 机械臂配置.
    
    仿真模式示例:
        config = PingtiFollowerConfig(
            id="pingti_sim",
            is_simulation=True,
        )
    """
    type: str = "pingti_follower"
    port: str = "/dev/ttyUSB0"
    use_degrees: bool = False
    action_filter_type: str = "exponential"
    action_filter_alpha: float = 0.95
    action_filter_window_size: int = 10
    action_filter_adaptation_threshold: float = 0.2
    max_relative_target: float | None = None
    disable_torque_on_disconnect: bool = True
    
    # 新增: 仿真配置
    is_simulation: bool = False
    simulation_render: bool = False
    simulation_dt: float = 0.002
    simulation_max_steps: int = 1000
    
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "hand_camera": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )