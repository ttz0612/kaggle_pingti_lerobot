"""快速测试 - 验证仿真环境是否工作."""

import sys
import numpy as np

def test_mujoco_import():
    """测试 MuJoCo 导入."""
    try:
        import mujoco
        print("✓ MuJoCo 导入成功")
        return True
    except ImportError as e:
        print(f"✗ MuJoCo 导入失败: {e}")
        return False

def test_simulator_creation():
    """测试仿真器创建."""
    try:
        from pingti.robots.pingti_follower.mujoco_sim import PingtiSimulator
        sim = PingtiSimulator(render=False)
        print("✓ 仿真器创建成功")
        return True
    except Exception as e:
        print(f"✗ 仿真器创建失败: {e}")
        return False

def test_simulator_reset():
    """测试仿真器重置."""
    try:
        from pingti.robots.pingti_follower.mujoco_sim import PingtiSimulator
        sim = PingtiSimulator(render=False)
        obs = sim.reset()
        print(f"✓ 仿真器重置成功, 观测形状: {obs.shape}")
        sim.close()
        return True
    except Exception as e:
        print(f"✗ 仿真器重置失败: {e}")
        return False

def test_simulator_step():
    """测试仿真器步骤."""
    try:
        from pingti.robots.pingti_follower.mujoco_sim import PingtiSimulator
        sim = PingtiSimulator(render=False)
        obs = sim.reset()
        
        action = np.zeros(5)
        obs, reward, done = sim.step(action)
        
        print(f"✓ 仿真器步骤成功")
        print(f"  观测形状: {obs.shape}")
        print(f"  奖励: {reward:.4f}")
        print(f"  完成: {done}")
        
        sim.close()
        return True
    except Exception as e:
        print(f"✗ 仿真器步骤失败: {e}")
        return False

def main():
    """运行所有测试."""
    print("\n" + "="*60)
    print("仿真环境快速测试")
    print("="*60 + "\n")
    
    tests = [
        ("MuJoCo 导入", test_mujoco_import),
        ("仿真器创建", test_simulator_creation),
        ("仿真器重置", test_simulator_reset),
        ("仿真器步骤", test_simulator_step),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"测试: {test_name}")
        result = test_func()
        results.append(result)
        print()
    
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"测试结果: {passed}/{total} 通过")
    print("="*60)
    
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())