import sys
import os
from streamlit.testing.v1 import AppTest
import numpy as np

def test_app_run():
    # 初始化 AppTest
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()

    # 1. 验证标题和初始状态
    assert at.title[0].value == "L2 RRM 单小区仿真控制台"
    
    # 2. 点击 "单小区 TDD 基线" 预设
    # 查找所有按钮并找到匹配标签的
    preset_btn = at.button[0] # 对应第一个按钮
    preset_btn.click().run()
    
    # 3. 修改 Slots 为 100 以加快测试速度
    at.number_input(key="num_slots").set_value(100).run()
    
    # 4. 点击 "运行仿真"
    run_btn = at.button[3] # 对应 "运行仿真" 按钮 (之前有3个预设按钮)
    run_btn.click().run()
    
    # 5. 检查是否产出了结果 (指标)
    # 预期至少有 "Delivered Throughput" 相关的 metric
    metrics = [m.label for m in at.metric]
    print(f"Found metrics: {metrics}")
    assert "Delivered Throughput" in metrics
    
    # 6. 检查是否有数据表格渲染
    assert len(at.dataframe) >= 1
    print("UI Test Passed: Simulation triggered and results rendered.")

if __name__ == "__main__":
    try:
        test_app_run()
        print("\n[SUCCESS] UI baseline check passed.")
    except Exception as e:
        print(f"\n[FAILURE] UI test failed: {e}")
        import traceback
        traceback.print_exc()
