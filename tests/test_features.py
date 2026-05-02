import pytest
import pandas as pd
import numpy as np

def test_distance_and_angle_logic():
    # Simulate the logic from build_features.py
    
    # Test cases:
    # 1. Shot from center ice (x=0, y=0) attacking net_x=89
    # 2. Shot from goal line (x=89, y=42) attacking net_x=89
    # 3. Shot from top of circle (x=60, y=20) attacking net_x=89
    
    df = pd.DataFrame([
        {'x_coord': 0, 'y_coord': 0, 'net_x': 89},
        {'x_coord': 89, 'y_coord': 42, 'net_x': 89},
        {'x_coord': 60, 'y_coord': 20, 'net_x': 89},
        {'x_coord': -60, 'y_coord': -20, 'net_x': -89} # Attacking negative net
    ])
    
    df['shot_distance'] = np.sqrt((df['net_x'] - df['x_coord'])**2 + df['y_coord']**2)
    
    df['shot_angle'] = np.where(df['shot_distance'] > 0, 
                                np.abs(np.arcsin(df['y_coord'] / df['shot_distance'])) * 180 / np.pi, 
                                0)
                                
    # Center ice shot: distance = 89, angle = 0
    assert df.loc[0, 'shot_distance'] == 89
    assert df.loc[0, 'shot_angle'] == 0
    
    # Goal line shot: distance = 42, angle = 90
    assert df.loc[1, 'shot_distance'] == 42
    assert df.loc[1, 'shot_angle'] == 90
    
    # Top of circle: distance = sqrt(29^2 + 20^2) = sqrt(841 + 400) = sqrt(1241) ≈ 35.2
    assert np.isclose(df.loc[2, 'shot_distance'], np.sqrt(1241))
    
    # Symmetric shot on other side
    assert np.isclose(df.loc[3, 'shot_distance'], np.sqrt(1241))
    assert np.isclose(df.loc[2, 'shot_angle'], df.loc[3, 'shot_angle'])

def test_royal_road_cross_logic():
    # Previous event < 3s ago, y_coord flipped sign, prev event in offensive zone
    df = pd.DataFrame([
        # Pass across ice (y flipped, fast, offensive zone) -> 1
        {'time_since_last_event': 2, 'y_coord': -15, 'prev_y': 15, 'prev_x': 70, 'net_x': 89},
        # Pass down boards (y didn't flip) -> 0
        {'time_since_last_event': 2, 'y_coord': 15, 'prev_y': 25, 'prev_x': 70, 'net_x': 89},
        # Slow play (too long ago) -> 0
        {'time_since_last_event': 5, 'y_coord': -15, 'prev_y': 15, 'prev_x': 70, 'net_x': 89},
        # Breakout pass (prev event was in defensive zone) -> 0
        {'time_since_last_event': 2, 'y_coord': -15, 'prev_y': 15, 'prev_x': -70, 'net_x': 89},
    ])
    
    df['royal_road_cross'] = (
        (df['time_since_last_event'] <= 3) & 
        (np.sign(df['y_coord']) != np.sign(df['prev_y'])) &
        (np.sign(df['prev_x']) == np.sign(df['net_x']))
    ).astype(int)
    
    assert df.loc[0, 'royal_road_cross'] == 1
    assert df.loc[1, 'royal_road_cross'] == 0
    assert df.loc[2, 'royal_road_cross'] == 0
    assert df.loc[3, 'royal_road_cross'] == 0
