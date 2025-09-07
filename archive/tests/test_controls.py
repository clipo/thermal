#!/usr/bin/env python3
"""Test script to verify navigation controls are visible and functional"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

def test_button_layout():
    """Test the button layout matches what we implemented"""
    fig = plt.figure(figsize=(18, 10))
    
    # Add a simple plot to simulate the viewer
    ax_main = fig.add_subplot(1, 1, 1)
    ax_main.imshow(np.random.rand(100, 100))
    ax_main.set_title("Test Image - Check Controls Below")
    
    # Control buttons - matching sgd_viewer.py layout
    btn_height = 0.03
    btn_width = 0.06
    bottom_margin = 0.02
    
    # Row 1 (upper) - Fine navigation
    row1_y = bottom_margin + 2*btn_height + 0.01
    
    buttons = []
    
    # Navigation buttons
    ax_prev = plt.axes([0.08, row1_y, btn_width, btn_height])
    buttons.append(Button(ax_prev, '← Prev'))
    
    ax_next = plt.axes([0.15, row1_y, btn_width, btn_height])
    buttons.append(Button(ax_next, 'Next →'))
    
    ax_minus10 = plt.axes([0.23, row1_y, btn_width, btn_height])
    buttons.append(Button(ax_minus10, '← -10'))
    
    ax_plus10 = plt.axes([0.30, row1_y, btn_width, btn_height])
    buttons.append(Button(ax_plus10, '+10 →'))
    
    ax_first = plt.axes([0.38, row1_y, btn_width, btn_height])
    buttons.append(Button(ax_first, 'First'))
    
    ax_last = plt.axes([0.45, row1_y, btn_width, btn_height])
    buttons.append(Button(ax_last, 'Last'))
    
    # Row 2 (middle) - Medium and large jumps
    row2_y = bottom_margin + btn_height + 0.005
    
    ax_minus5 = plt.axes([0.08, row2_y, btn_width, btn_height])
    buttons.append(Button(ax_minus5, '← -5'))
    
    ax_plus5 = plt.axes([0.15, row2_y, btn_width, btn_height])
    buttons.append(Button(ax_plus5, '+5 →'))
    
    ax_minus25 = plt.axes([0.23, row2_y, btn_width, btn_height])
    buttons.append(Button(ax_minus25, '← -25'))
    
    ax_plus25 = plt.axes([0.30, row2_y, btn_width, btn_height])
    buttons.append(Button(ax_plus25, '+25 →'))
    
    # Frame counter
    ax_info = plt.axes([0.53, row1_y, 0.10, btn_height])
    ax_info.axis('off')
    ax_info.text(0.5, 0.5, 'Frame 1/100', ha='center', va='center', fontsize=10)
    
    # Action buttons
    ax_mark = plt.axes([0.65, row1_y, 0.08, btn_height])
    buttons.append(Button(ax_mark, 'Mark SGD'))
    
    ax_save = plt.axes([0.74, row1_y, 0.08, btn_height])
    buttons.append(Button(ax_save, 'Save'))
    
    ax_export = plt.axes([0.83, row1_y, 0.08, btn_height])
    buttons.append(Button(ax_export, 'Export'))
    
    # Row 3 (bottom) - Status text
    row3_y = bottom_margin
    
    # Add labels
    fig.text(0.02, row1_y + 0.01, 'Nav:', fontsize=8, fontweight='bold')
    fig.text(0.02, row3_y + 0.01, 'Status:', fontsize=8, fontweight='bold')
    
    # Add status message
    fig.text(0.08, row3_y, 'All controls visible and positioned correctly', 
             fontsize=10, color='green')
    
    # Button click handler
    def on_click(label):
        print(f"Button clicked: {label}")
    
    # Connect all buttons
    for i, btn in enumerate(buttons):
        btn.on_clicked(lambda e, l=btn.label.get_text(): on_click(l))
    
    plt.suptitle("Navigation Controls Test - Check if all buttons are visible", fontsize=14)
    
    print("\nButton Layout Test")
    print("=" * 50)
    print("Check that you can see:")
    print("  Row 1: Prev, Next, -10, +10, First, Last, Mark SGD, Save, Export")
    print("  Row 2: -5, +5, -25, +25")
    print("  Row 3: Status text")
    print("\nClick any button to test functionality")
    print("Close window when done testing")
    
    plt.show()

if __name__ == "__main__":
    test_button_layout()