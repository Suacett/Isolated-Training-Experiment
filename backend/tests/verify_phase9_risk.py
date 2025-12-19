
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from services.risk_management import calculate_atr

def test_calculate_atr_validation():
    print("Testing calculate_atr validation...")
    
    # Test empty DF
    try:
        calculate_atr(pd.DataFrame())
    except ValueError as e:
        print(f"✅ Caught expected empty DF error: {e}")
        
    # Test missing columns
    df = pd.DataFrame({'close': [1, 2, 3]})
    try:
        calculate_atr(df)
    except ValueError as e:
        print(f"✅ Caught expected missing columns error: {e}")
        
    # Test invalid period
    df = pd.DataFrame({'high': [10, 11], 'low': [9, 10], 'close': [10, 10.5]})
    try:
        calculate_atr(df, period=-1)
    except ValueError as e:
        print(f"✅ Caught expected invalid period error: {e}")

    # Test insufficient data
    try:
        calculate_atr(df, period=14)
    except ValueError as e:
        print(f"✅ Caught expected insufficient data error: {e}")

    # Test valid data
    df_valid = pd.DataFrame({
        'high': np.random.uniform(105, 110, 20),
        'low': np.random.uniform(100, 105, 20),
        'close': np.random.uniform(100, 110, 20)
    })
    try:
        atr = calculate_atr(df_valid, period=14)
        print(f"✅ Valid ATR calculation successful. Last value: {atr.iloc[-1]:.4f}")
    except Exception as e:
        print(f"❌ Valid ATR calculation failed: {e}")

if __name__ == "__main__":
    test_calculate_atr_validation()
