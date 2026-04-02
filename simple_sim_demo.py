"""
最简单的仿真演示脚本 - 无需机械臂硬件.

快速开始:
    python simple_sim_demo.py
"""

import numpy as np
import time
from pingti.robots.pingti_follower.mujoco_sim import PingtiSimulator, PingtiSimulatorEnv


def demo_1_basic_simulation():
    """演示 1: 基础仿真 - 随机动作."""
    print("\n" + "="*60)
    print("演示 1: 基础仿真 - 随机动作")
    print("="*60)
    
    # 创建仿真器 (不渲染)
    sim = PingtiSimulator(render=False)
    
    # 重置
    obs = sim.reset()
    print(f"初始观测形状: {obs.shape}")
    print(f"初始关节位置: {sim.get_joint_positions()}")
    
    # 运行 100 步
    total_reward = 0
    for step in range(100):
        # 随机动作
        action = np.random.uniform(-1, 1, 5)
        obs, reward, done = sim.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"步数 {step:3d}: 奖励 {reward:6.3f}, 总奖励 {total_reward:7.3f}")
    
    sim.close()
    print(f"✓ 演示完成")


def demo_2_visualization():
    """演示 2: 可视化仿真 - 正弦波控制."""
    print("\n" + "="*60)
    print("演示 2: 可视化仿真 - 正弦波控制")
    print("="*60)
    print("提示: 将显示 MuJoCo 可视化窗口")
    
    # 创建仿真器 (启用渲染)
    sim = PingtiSimulator(render=True)
    obs = sim.reset()
    
    # 运行 500 步，使用正弦波控制
    t = 0
    for step in range(500):
        # 正弦波控制
        action = np.array([
            0.5 * np.sin(2 * np.pi * t / 100),
            0.3 * np.sin(2 * np.pi * t / 150),
            0.4 * np.sin(2 * np.pi * t / 200),
            0.2 * np.sin(2 * np.pi * t / 250),
            0.5 * np.sin(2 * np.pi * t / 300),
        ])
        
        obs, reward, done = sim.step(action)
        t += 1
        
        # 每 50 步打印一次
        if step % 50 == 0:
            print(f"步数 {step:3d}: 关节位置 {sim.get_joint_positions()}")
    
    sim.close()
    print(f"✓ 演示完成")


def demo_3_gym_interface():
    """演示 3: Gym 风格接口."""
    print("\n" + "="*60)
    print("演示 3: Gym 风格接口")
    print("="*60)
    
    env = PingtiSimulatorEnv(render=False)
    
    # 重置
    obs, info = env.reset()
    print(f"观测形状: {obs.shape}")
    print(f"观测样本: {obs}")
    
    # 运行 200 步
    total_reward = 0
    for step in range(200):
        action = np.random.uniform(-1, 1, 5)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 50 == 0:
            print(f"步数 {step:3d}: 奖励 {reward:6.3f}, 总奖励 {total_reward:7.3f}")
    
    env.close()
    print(f"✓ 演示完成")


def demo_4_trajectory_tracking():
    """演示 4: 轨迹跟踪控制."""
    print("\n" + "="*60)
    print("演示 4: 轨迹跟踪控制")
    print("="*60)
    
    sim = PingtiSimulator(render=False)
    obs = sim.reset()
    
    # 目标轨迹
    target_trajectory = [
        [0.0, 0.5, 0.0, -1.0, 0.0],
        [0.5, 0.3, 0.5, -1.2, 0.2],
        [0.3, 0.6, -0.3, -0.8, -0.2],
        [0.0, 0.5, 0.0, -1.0, 0.0],
    ]
    
    # 简单控制器 (P 控制器)
    Kp = 2.0  # 比例增益
    
    total_error = 0
    step = 0
    
    for _ in range(4):  # 重复轨迹
        for target_pos in target_trajectory:
            for _ in range(50):  # 每个目标位置 50 步
                current_pos = sim.get_joint_positions()
                
                # P 控制
                action = Kp * (np.array(target_pos) - current_pos)
                action = np.clip(action, -1, 1)
                
                obs, reward, done = sim.step(action)
                
                error = np.sum((current_pos - np.array(target_pos)) ** 2)
                total_error += error
                step += 1
                
                if step % 200 == 0:
                    print(f"步数 {step:4d}: 跟踪误差 {error:.4f}, 平均误差 {total_error/step:.4f}")
    
    sim.close()
    print(f"✓ 演示完成 - 总步数: {step}, 平均误差: {total_error/step:.4f}")


def demo_5_data_collection():
    """演示 5: 数据收集 - 生成仿真轨迹."""
    print("\n" + "="*60)
    print("演示 5: 数据收集 - 生成仿真轨迹")
    print("="*60)
    
    sim = PingtiSimulator(render=False)
    
    # 收集 5 个轨迹
    trajectories = []
    
    for episode in range(5):
        print(f"收集轨迹 {episode + 1}/5...")
        obs = sim.reset()
        
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
        }
        
        for step in range(100):
            # 随机动作
            action = np.random.uniform(-0.5, 0.5, 5)
            obs, reward, done = sim.step(action)
            
            trajectory['observations'].append(obs.copy())
            trajectory['actions'].append(action.copy())
            trajectory['rewards'].append(reward)
        
        trajectories.append(trajectory)
        print(f"  收集了 100 步数据")
    
    sim.close()
    
    # 统计信息
    print(f"\n✓ 数据收集完成")
    print(f"  轨迹数: {len(trajectories)}")
    print(f"  每个轨迹长度: {len(trajectories[0]['observations'])} 步")
    print(f"  观测维度: {trajectories[0]['observations'][0].shape}")
    print(f"  动作维度: {trajectories[0]['actions'][0].shape}")
    print(f"  平均奖励: {np.mean([r for t in trajectories for r in t['rewards']]):.4f}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PingTi 机械臂仿真演示 - 无需硬件")
    print("="*60)
    
    try:
        # 运行所有演示
        demo_1_basic_simulation()
        demo_3_gym_interface()
        demo_4_trajectory_tracking()
        demo_5_data_collection()
        
        # 演示 2 最后运行 (因为需要显示窗口)
        print("\n按 Enter 运行可视化演示 (需要 X11 或类似显示服务)...")
        # input()
        # demo_2_visualization()
        
        print("\n" + "="*60)
        print("所有演示完成！🎉")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()