import sys
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import requests
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QLabel, QTextEdit, QFileDialog, QDialog,
    QFormLayout, QLineEdit, QDialogButtonBox, QMenu, QAction, QMessageBox,
    QListWidgetItem, QTabWidget, QFrame, QCheckBox, QComboBox, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import datetime
import seaborn as sns
from scipy import stats
import torch
import ctypes
import geopandas as gpd
from sqlalchemy import create_engine

# Add PostgreSQL bin directory to PATH
os.environ['PATH'] += r"PATH" # Add this path incase Postgres dll file is not being recognized

# Load the libpq.dll manually using the full path
ctypes.CDLL(r"DLL FILE PATH")  # Add this path incase Postgres dll file is not being recognized

import psycopg2
print("psycopg2 installed successfully")

import google.generativeai as genai

import logging

# Configure logging
logging.basicConfig(filename='chatbot_debug.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

from flask import Flask, request, jsonify

class MapCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        super().__init__(self.fig)
        self.setStyleSheet("background-color: white;")
        self.legend = None
        self.raster_layers = []
        self.show_legend = True  # Add this line
        
        # Add zoom functionality
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.pressed = None
        
        # Connect events
        self.mpl_connect('scroll_event', self.zoom)
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)

    def plot_rasters(self, raster_layers):
        self.raster_layers = raster_layers
        self.ax.clear()
        self.ax.axis('off')
        
        # Always clean up any existing legend first
        if self.legend:
            try:
                self.legend.remove()
            except:
                pass
            self.legend = None
        
        # Find the first visible layer to set display limits
        visible_layers = [layer for layer in raster_layers if layer['visible'] and layer['data'] is not None]
        if not visible_layers:
            return
            
        # Use the first visible layer for initial display limits
        first_layer = visible_layers[0]
        self.ax.set_xlim(0, first_layer['data'].shape[1])
        self.ax.set_ylim(first_layer['data'].shape[0], 0)
        
        # Apply zoom and pan relative to data extent
        if self.zoom_scale != 1.0:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            
            # Calculate zoomed width and height
            width = (xlim[1] - xlim[0]) / self.zoom_scale
            height = (ylim[1] - ylim[0]) / self.zoom_scale
            
            # Apply pan after zoom
            x_center += self.pan_x
            y_center += self.pan_y
            
            # Set new limits
            self.ax.set_xlim(x_center - width/2, x_center + width/2)
            self.ax.set_ylim(y_center - height/2, y_center + height/2)

        # Plot all layers
        final_map_im = None
        for layer in raster_layers:
            if layer['visible'] and layer['data'] is not None:
                im = self.ax.imshow(layer['data'], 
                                  cmap=layer['cmap'], 
                                  alpha=0.7,
                                  extent=[0, layer['data'].shape[1], 
                                        layer['data'].shape[0], 0])
                if layer.get('is_final_map', False):
                    final_map_im = im

        # Add legend only if needed
        if final_map_im and self.show_legend:
            ticks = [1, 2, 3, 4, 5]
            labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
            self.legend = self.fig.colorbar(
                final_map_im, ax=self.ax,
                label='Flood Susceptibility',
                ticks=ticks,
                shrink=0.3,
                aspect=5,
                pad=0.02
            )
            self.legend.ax.set_yticklabels(labels, fontsize=8)
            self.legend.ax.set_ylabel('Flood Susceptibility', fontsize=8)

        self.fig.tight_layout()
        self.draw()

    def zoom(self, event):
        if event.inaxes != self.ax:
            return
        
        # Zoom with mouse wheel
        if event.button == 'up':
            self.zoom_scale *= 0.9
        elif event.button == 'down':
            self.zoom_scale *= 1.1
        
        self.zoom_scale = max(0.1, min(self.zoom_scale, 3.0))  # Limit zoom range
        self.plot_rasters(self.raster_layers)
    
    def on_press(self, event):
        # Start panning
        if event.button == 1:  # Left click
            self.pressed = event.xdata, event.ydata
    
    def on_motion(self, event):
        # Pan while dragging
        if self.pressed and event.xdata and event.ydata:
            dx = event.xdata - self.pressed[0]
            dy = event.ydata - self.pressed[1]
            self.pan_x -= dx * self.zoom_scale
            self.pan_y -= dy * self.zoom_scale
            self.plot_rasters(self.raster_layers)
    
    def on_release(self, event):
        # Stop panning
        self.pressed = None
    
    def reset_view(self):
        """Reset zoom and pan to initial state"""
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.ax.clear()  # Clear the axes completely
        self.ax.axis('off')
        
        # Replot all layers at initial extent
        visible_layers = [layer for layer in self.raster_layers if layer['visible'] and layer['data'] is not None]
        if not visible_layers:
            return
            
        # Reset to full extent of first visible layer
        first_layer = visible_layers[0]
        self.ax.set_xlim(0, first_layer['data'].shape[1])
        self.ax.set_ylim(first_layer['data'].shape[0], 0)
        
        # Replot layers
        self.plot_rasters(self.raster_layers)
    
    def toggle_legend(self, state):
        """Toggle legend visibility"""
        self.show_legend = state
        self.plot_rasters(self.raster_layers)  # Redraw with or without legend


class WeightDialog(QDialog):
    def __init__(self, layers):
        super().__init__()
        self.setWindowTitle("Set Weights for Layers")
        self.weights = []
        layout = QFormLayout()
        self.inputs = []
        for layer in layers:
            line_edit = QLineEdit("1")
            self.inputs.append(line_edit)
            layout.addRow(layer, line_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)
    
    def get_weights(self):
        return [float(inp.text()) for inp in self.inputs]


class SymbologyDialog(QDialog):
    def __init__(self, current_cmap):
        super().__init__()
        self.setWindowTitle("Change Symbology")
        layout = QVBoxLayout()
        
        self.cmaps = {
            'Sequential': ['viridis', 'plasma', 'inferno', 'magma', 'YlOrRd', 'YlOrBr'],
            'Diverging': ['RdYlBu_r', 'coolwarm', 'RdBu_r', 'BrBG', 'RdYlGn'],
            'Qualitative': ['Set1', 'Set2', 'Set3', 'Paired', 'tab10']
        }
        
        self.cmap_list = QListWidget()
        for category, maps in self.cmaps.items():
            self.cmap_list.addItem(f"=== {category} ===")
            for cmap in maps:
                item = QListWidgetItem(cmap)
                self.cmap_list.addItem(item)
                # Preselect the current colormap
                if cmap == current_cmap:
                    self.cmap_list.setCurrentItem(item)
        
        # Add a preview
        self.preview_label = QLabel("Color Ramp Preview:")
        layout.addWidget(self.preview_label)
        
        # Create preview canvas
        self.preview_canvas = FigureCanvas(Figure(figsize=(4, 1)))
        self.preview_ax = self.preview_canvas.figure.add_subplot(111)
        layout.addWidget(self.preview_canvas)
        
        # Update preview when selection changes
        self.cmap_list.itemClicked.connect(self.update_preview)
        layout.addWidget(QLabel("Select Color Scheme:"))
        layout.addWidget(self.cmap_list)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
        # Show initial preview
        self.update_preview(self.cmap_list.currentItem())
    
    def update_preview(self, item):
        if not item or item.text().startswith('==='):
            return
        # Create gradient data for preview
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        
        # Update preview with selected colormap
        self.preview_ax.clear()
        self.preview_ax.imshow(gradient, aspect='auto', cmap=item.text())
        self.preview_ax.set_yticks([])
        self.preview_ax.set_xticks([])
        self.preview_canvas.draw()


class StatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create tabs for different visualizations
        self.tabs = QTabWidget()
        
        # Histogram tab
        self.hist_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        self.hist_ax = self.hist_canvas.figure.add_subplot(111)
        self.tabs.addTab(self.hist_canvas, "Histogram")
        
        # Scatter plot tab
        self.scatter_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        self.scatter_ax = self.scatter_canvas.figure.add_subplot(111)
        self.tabs.addTab(self.scatter_canvas, "Scatter Plot")
        
        # Box plot tab
        self.box_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        self.box_ax = self.box_canvas.figure.add_subplot(111)
        self.tabs.addTab(self.box_canvas, "Box Plot")
        
        layout.addWidget(self.tabs)
        
        # Statistics text display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(100)
        layout.addWidget(self.stats_text)
        
        self.setLayout(layout)
    
    def plot_histogram(self, data, title="Histogram"):
        self.hist_ax.clear()
        self.hist_ax.hist(data[~np.isnan(data)].flatten(), bins=50)
        self.hist_ax.set_title(title)
        self.hist_canvas.draw()
    
    def plot_scatter(self, data1, data2, label1="Layer 1", label2="Layer 2"):
        self.scatter_ax.clear()
        valid_mask = ~(np.isnan(data1) | np.isnan(data2))
        self.scatter_ax.scatter(data1[valid_mask].flatten(), 
                              data2[valid_mask].flatten(), 
                              alpha=0.5)
        self.scatter_ax.set_xlabel(label1)
        self.scatter_ax.set_ylabel(label2)
        self.scatter_ax.set_title("Scatter Plot")
        self.scatter_canvas.draw()
    
    def plot_boxplot(self, data_list, labels):
        self.box_ax.clear()
        plot_data = [d[~np.isnan(d)].flatten() for d in data_list]
        self.box_ax.boxplot(plot_data, labels=labels)
        self.box_ax.set_title("Box Plot")
        self.box_canvas.draw()
    
    def update_stats(self, data):
        if data is None:
            return
        
        valid_data = data[~np.isnan(data)]
        stats_text = f"""
        Statistical Summary:
        Mean: {np.mean(valid_data):.2f}
        Median: {np.median(valid_data):.2f}
        Std Dev: {np.std(valid_data):.2f}
        Min: {np.min(valid_data):.2f}
        Max: {np.max(valid_data):.2f}
        Skewness: {stats.skew(valid_data):.2f}
        """
        self.stats_text.setText(stats_text)


class ChatPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.setup_ai()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(6)

        # Header with Assistant label
        header = QLabel("Chatbot")
        header.setStyleSheet("font-size: 13px; font-weight: bold; margin-bottom: 2px;")
        layout.addWidget(header)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMaximumHeight(120)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 4px;
            }
        """)
        layout.addWidget(self.chat_display)

        # Input area
        input_layout = QHBoxLayout()
        input_layout.setSpacing(4)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask a question about flood analysis...")
        self.input_field.returnPressed.connect(self.send_message)
        self.input_field.setStyleSheet("padding: 4px;")

        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        send_button.setMaximumWidth(50)
        send_button.setStyleSheet("padding: 4px;")

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(send_button)
        layout.addLayout(input_layout)

        frame = QFrame()
        frame.setLayout(layout)
        frame.setStyleSheet("QFrame { background-color: #f5f5f5; border-radius: 5px; padding: 6px; }")

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(frame)
        self.setLayout(main_layout)

    def setup_ai(self):
        try:
            # Configure Gemini AI
            genai.configure(api_key="INSERT YOUR API KEY")
            self.model = genai.GenerativeModel('INSERT YOUR AI MODEL E.G LIKE GEIMINI')
            self.chat = self.model.start_chat(history=[])

            # Add initial greeting
            self.chat_display.append("<span style='color: #666;'>💡 Assistant: Greetings. I'm here as your guide—ready to assist, explore, and unravel whatever challenges or questions you bring. Just ask.!</span>")

        except Exception as e:
            self.chat_display.append(f"<span style='color: red;'>Error initializing AI: {str(e)}</span>")
            self.input_field.setEnabled(False)

    def send_message(self):
        question = self.input_field.text().strip()
        if not question:
            return

        # Display user message
        self.chat_display.append(f"<span style='font-weight: bold;'>👤 You:</span> {question}")
        self.input_field.clear()

        try:
            # Get AI response
            response = self.chat.send_message(question)
            answer = response.text.strip()

            # Display AI response
            self.chat_display.append(f"<span style='font-weight: bold;'>💡 Assistant:</span> {answer}\n")

        except Exception as e:
            self.chat_display.append(f"<span style='color: red;'>Error: {str(e)}</span>\n")

        # Scroll to bottom
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )


class StatsWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Statistical Analysis Tools")
        self.setGeometry(200, 200, 600, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        # Add stats panel
        self.stats_panel = StatsPanel()
        layout.addWidget(self.stats_panel)
        
        # Add analysis buttons
        btn_layout = QHBoxLayout()
        analyze_btn = QPushButton("Analyze Selected Layer")
        analyze_btn.clicked.connect(self.analyze_selected_layer)
        compare_btn = QPushButton("Compare Selected Layers")
        compare_btn.clicked.connect(self.compare_selected_layers)
        btn_layout.addWidget(analyze_btn)
        btn_layout.addWidget(compare_btn)
        layout.addLayout(btn_layout)
        
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        self.raster_layers = []
        self.layer_panel = None
    
    def set_data(self, raster_layers, layer_panel):
        self.raster_layers = raster_layers
        self.layer_panel = layer_panel
    
    def analyze_selected_layer(self):
        items = self.layer_panel.selectedItems()
        if not items:
            QMessageBox.warning(self, "Warning", "Please select a layer to analyze!")
            return
        
        layer_name = items[0].text()
        layer = next((layer for layer in self.raster_layers if layer['name'] == layer_name), None)
        
        if layer and layer['data'] is not None:
            self.stats_panel.plot_histogram(layer['data'], title=f"Histogram of {layer_name}")
            self.stats_panel.update_stats(layer['data'])
            self.stats_panel.tabs.setCurrentIndex(0)
    
    def compare_selected_layers(self):
        items = self.layer_panel.selectedItems()
        if len(items) != 2:
            QMessageBox.warning(self, "Warning", "Please select exactly two layers to compare!")
            return
            
        layer1 = next((layer for layer in self.raster_layers if layer['name'] == items[0].text()), None)
        layer2 = next((layer for layer in self.raster_layers if layer['name'] == items[1].text()), None)
        
        if layer1 and layer2 and layer1['data'] is not None and layer2['data'] is not None:
            self.stats_panel.plot_scatter(layer1['data'], layer2['data'], 
                                        label1=layer1['name'], 
                                        label2=layer2['name'])
            self.stats_panel.plot_boxplot([layer1['data'], layer2['data']], 
                                        [layer1['name'], layer2['name']])
            self.stats_panel.tabs.setCurrentIndex(1)


class FloodMappingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flood Susceptibility Mapping App")
        self.setGeometry(100, 100, 1200, 800)
        self.raster_layers = []
        self.stats_window = None
        self.fuzzy_membership_dialog = None
        self.init_ui()
    
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left panel
        left_panel_layout = QVBoxLayout()
        
        # Layer panel with Add Data button
        layer_label = QLabel("🗂 Layer Panel")
        layer_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        left_panel_layout.addWidget(layer_label)

        add_data_btn = QPushButton("➕ Add Data")
        add_data_btn.clicked.connect(self.add_data)
        left_panel_layout.addWidget(add_data_btn)

        self.layer_panel = QListWidget()
        self.layer_panel.setSelectionMode(QListWidget.ExtendedSelection)
        self.layer_panel.setContextMenuPolicy(Qt.CustomContextMenu)
        self.layer_panel.customContextMenuRequested.connect(self.show_layer_context_menu)
        self.layer_panel.itemClicked.connect(self.bring_layer_to_front)
        left_panel_layout.addWidget(self.layer_panel)

        main_layout.addLayout(left_panel_layout)

        # Center panel (map)
        self.map_display = MapCanvas()
        main_layout.addWidget(self.map_display, 2)

        # Right panel
        right_panel_layout = QVBoxLayout()
        
        # Tools section at the top
        tools_frame = QFrame()
        tools_frame.setStyleSheet("QFrame { background-color: #f0f0f0; border-radius: 5px; padding: 10px; }")
        tools_layout = QVBoxLayout()
        
        tools_label = QLabel("🛠️ Tools")
        tools_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        tools_layout.addWidget(tools_label)
        
        # Add remaining tool buttons
        stats_btn = QPushButton("📊 Statistical Analysis")
        stats_btn.clicked.connect(self.show_stats_window)
        tools_layout.addWidget(stats_btn)

        generate_btn = QPushButton("🗺️ Generate Flood Map")
        generate_btn.clicked.connect(self.generate_flood_map)
        tools_layout.addWidget(generate_btn)

        report_btn = QPushButton("📄 Generate Report")
        report_btn.clicked.connect(self.generate_report)
        tools_layout.addWidget(report_btn)
        
        # Add reset view button to tools section
        reset_view_btn = QPushButton("🔄 Reset View")
        reset_view_btn.clicked.connect(lambda: self.map_display.reset_view())
        tools_layout.addWidget(reset_view_btn)

        # Add legend control checkbox
        legend_control = QCheckBox("🏷️ Show Legend")
        legend_control.setChecked(True)
        legend_control.stateChanged.connect(lambda state: self.map_display.toggle_legend(bool(state)))
        tools_layout.addWidget(legend_control)

        # Add Fuzzy Logic Info button
        fuzzy_info_btn = QPushButton("ℹ️ Fuzzy Rules Info")
        fuzzy_info_btn.clicked.connect(self.show_fuzzy_info)
        tools_layout.addWidget(fuzzy_info_btn)

        # Add "Display Web App" button to tools section
        display_web_app_btn = QPushButton("🌐 Display Web App")
        display_web_app_btn.clicked.connect(self.open_web_app)
        tools_layout.addWidget(display_web_app_btn)

        # Add "PostGIS" button to tools section
        postgis_btn = QPushButton("🌍 PostGIS")
        postgis_btn.clicked.connect(self.connect_postgis)
        tools_layout.addWidget(postgis_btn)

        # Add "Open Raster App" button to tools section
        open_raster_app_btn = QPushButton("🌍 Open Raster App")
        open_raster_app_btn.clicked.connect(self.open_raster_app)
        tools_layout.addWidget(open_raster_app_btn)

        tools_frame.setLayout(tools_layout)
        right_panel_layout.addWidget(tools_frame)

        # Add stretcher to push chat panel to bottom
        right_panel_layout.addStretch()
        
        # Chat panel at the bottom with AI symbol
        chat_frame = QFrame()
        chat_frame.setStyleSheet("QFrame { background-color: #f5f5f5; border-radius: 5px; padding: 10px; }")
        chat_layout = QVBoxLayout()
        
        # AI Assistant header
        header_text = QLabel("Assistant")
        header_text.setStyleSheet("font-size: 16px;")
        chat_layout.addWidget(header_text)
        
        # Add chat panel
        self.chat_panel = ChatPanel()
        chat_layout.addWidget(self.chat_panel)
        chat_frame.setLayout(chat_layout)
        right_panel_layout.addWidget(chat_frame)
        
        right_panel_widget = QWidget()
        right_panel_widget.setLayout(right_panel_layout)
        right_panel_widget.setFixedWidth(400)
        main_layout.addWidget(right_panel_widget)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def show_stats_window(self):
        if self.stats_window is None:
            self.stats_window = StatsWindow()
        self.stats_window.set_data(self.raster_layers, self.layer_panel)
        self.stats_window.show()
    
    def add_data(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open Raster(s)", "", "TIFF Files (*.tif *.tiff)")
        for path in paths:
            with rasterio.open(path) as src:
                raster = src.read(1).astype(float)
                raster[raster == src.nodata] = np.nan
                self.raster_layers.append({
                    "name": os.path.basename(path),
                    "path": path,
                    "data": raster,
                    "cmap": 'viridis',
                    "visible": True
                })
                self.layer_panel.addItem(os.path.basename(path))
        self.update_map()
    
    def update_map(self):
        self.map_display.plot_rasters(self.raster_layers)
    
    def show_layer_context_menu(self, position):
        selected_items = self.layer_panel.selectedItems()
        if not selected_items:
            return
        menu = QMenu()
        
        toggle_visibility = QAction("🔁 Toggle Visibility", self)
        toggle_visibility.triggered.connect(lambda: self.toggle_layer_visibility(selected_items))
        menu.addAction(toggle_visibility)
        
        change_symbology = QAction("🌈 Change Symbology", self)
        change_symbology.triggered.connect(lambda: self.change_symbology(selected_items))
        menu.addAction(change_symbology)

        remove_layer = QAction("❌ Remove Selected Layer(s)", self)
        remove_layer.triggered.connect(lambda: self.remove_layers(selected_items))
        menu.addAction(remove_layer)
        
        menu.exec_(self.layer_panel.mapToGlobal(position))
    
    def toggle_layer_visibility(self, items):
        for item in items:
            name = item.text()
            for layer in self.raster_layers:
                if layer['name'] == name:
                    layer['visible'] = not layer['visible']
                    break
        self.update_map()
    
    def change_symbology(self, items):
        if not items:
            return
        
        # Get the first selected layer
        selected_item = items[0]
        layer_name = selected_item.text()
        
        # Find the layer by name
        selected_layer = None
        for layer in self.raster_layers:
            if layer['name'] == layer_name:
                selected_layer = layer
                break
            
        if not selected_layer:
            return
        
        # Open symbology dialog with current colormap
        dialog = SymbologyDialog(selected_layer['cmap'])
        if dialog.exec_():
            # Get the selected colormap from the dialog
            selected_item = dialog.cmap_list.currentItem()
            if selected_item and selected_item.text() and not selected_item.text().startswith('==='):
                # Update the layer's colormap
                selected_layer['cmap'] = selected_item.text()
                self.update_map()
                QMessageBox.information(self, "Success", f"Symbology updated for {layer_name}")

    def remove_layers(self, items):
        for item in items:
            name = item.text()
            # Find and remove layer by name
            for i, layer in enumerate(self.raster_layers):
                if layer['name'] == name:
                    del self.raster_layers[i]
                    self.layer_panel.takeItem(self.layer_panel.row(item))
                    break
        
        # Clear map display if no layers remain
        if len(self.raster_layers) == 0:
            self.map_display.ax.clear()
            self.map_display.ax.axis('off')
            self.map_display.draw()
        else:
            self.update_map()
    
    def bring_layer_to_front(self, item):
        name = item.text()
        # Find layer by name
        for i, layer in enumerate(self.raster_layers):  # Fixed the unpacking issue by using enumerate
            if layer['name'] == name:
                moved_layer = self.raster_layers.pop(i)
                self.raster_layers.append(moved_layer)
                break
        self.update_map()
    
    def show_fuzzy_info(self):
        try:
            loaded_files = [layer['name'].lower() for layer in self.raster_layers]
            # Check if all required layers are loaded
            required_factors = ['rainfall', 'elevation', 'lulc', 'soil', 'river_distance']
            
            missing = []
            for factor in required_factors:
                if not any(factor in file for file in loaded_files):
                    missing.append(factor.title())
            
            if missing:
                QMessageBox.warning(self, "Missing Factors", 
                                  f"Please load all required factors first.\nMissing: {', '.join(missing)}")
                return
                
            self.fuzzy_membership_dialog = FuzzyMembershipDialog(self, self.raster_layers)
            self.fuzzy_membership_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def generate_flood_map(self):
        try:
            if len(self.raster_layers) < 5:
                QMessageBox.warning(self, "Warning", "Please add all required layers: Rainfall, Elevation, Land Use, Soil, River Distance")
                return
                
            # Get weights
            weight_dialog = WeightDialog(['Rainfall', 'Elevation', 'Land Use', 'Soil', 'River Distance'])
            if weight_dialog.exec_():
                weights = np.array(weight_dialog.get_weights())
                weights = weights / np.sum(weights)  # Normalize weights
            else:
                return
            
            # Normalize layers for fuzzy processing
            normalized_layers = []
            for layer in self.raster_layers[:5]:
                data = layer['data']
                min_val = np.nanmin(data)
                max_val = np.nanmax(data)
                norm = (data - min_val) / (max_val - min_val)
                norm = norm * 5  # Scale to 0-5 range
                normalized_layers.append(norm)
            
            # Apply fuzzy logic
            fuzzy = FuzzyMembershipDialog(self)
            final_map = fuzzy.apply_fuzzy_logic(normalized_layers, weights)
            
            # Save result
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Flood Map", "", "TIFF Files (*.tif)")
            if save_path:
                with rasterio.open(save_path, 'w',
                                 driver='GTiff',
                                 height=final_map.shape[0],
                                 width=final_map.shape[1],
                                 count=1,
                                 dtype='float32',
                                 crs='+proj=latlong',
                                 transform=rasterio.transform.from_origin(0, 0, 1, 1)) as dst:
                    dst.write(final_map.astype('float32'), 1)
                
                self.raster_layers.append({
                    "name": "Flood Susceptibility Map",
                    "path": save_path,
                    "data": final_map,
                    "cmap": 'RdYlBu_r',
                    "visible": True,
                    "is_final_map": True
                })
                self.layer_panel.addItem("Flood Susceptibility Map")
                self.update_map()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating flood map: {str(e)}")

    def generate_report(self):
        if not any(layer.get('is_final_map', False) for layer in self.raster_layers):
            QMessageBox.warning(self, "Warning", "Please generate flood susceptibility map first!")
            return
        
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Report As", "", "PDF Files (*.pdf)")
        if not save_path:
            return

        doc = SimpleDocTemplate(save_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        
        # Add title and date
        story.append(Paragraph("Flood Susceptibility Analysis Report", title_style))
        story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
        story.append(Spacer(1, 20))
        
        # Input data table
        story.append(Paragraph("Input Data", styles['Heading2']))
        input_data = []
        for layer in self.raster_layers:
            if not layer.get('is_final_map', False):
                data = layer['data']
                input_data.append([
                    layer['name'],
                    f"{data.shape[0]}x{data.shape[1]}",
                    f"{np.nanmin(data):.2f}",
                    f"{np.nanmax(data):.2f}"  # Removed extra colon
                ])
        
        if input_data:
            table = Table([['Layer Name', 'Dimensions', 'Min Value', 'Max Value']] + input_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
        
        # Methodology section
        story.append(Paragraph("Methodology", styles['Heading2']))
        methodology_text = """
        The flood susceptibility mapping process involves the following steps:
        1. Data Preprocessing: Input raster layers are normalized and classified into 5 classes.
        2. Weight Assignment: Each factor is assigned a weight based on its importance.
        3. Weighted Overlay: The classified layers are combined using weighted sum method.
        4. Susceptibility Classification: The final map is classified into 5 categories:
           Very Low, Low, Moderate, High, and Very High susceptibility zones.

        Fuzzy Methodology Implementation:
        The analysis incorporates fuzzy logic principles to handle uncertainty and imprecision:
        • Fuzzification: Input criteria are transformed into fuzzy membership values (0-1 range)
        • Fuzzy Overlay: Weighted combination of fuzzy membership values
        • Defuzzification: Final fuzzy values are converted to crisp susceptibility classes

        The fuzzy approach provides:
        • Better handling of uncertainty in input data
        • Smooth transitions between susceptibility classes
        • More realistic representation of natural phenomena
        • Integration of expert knowledge through weight assignment
        """
        story.append(Paragraph(methodology_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Add map image
        try:
            buf = BytesIO()
            self.map_display.fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            img = Image(buf)
            img.drawHeight = 300
            img.drawWidth = 400
            story.append(Paragraph("Final Flood Susceptibility Map", styles['Heading2']))
            story.append(img)
        except Exception as e:
            print(f"Error saving map: {e}")

        # Build and save the report
        try:
            doc.build(story)
            QMessageBox.information(self, "Success", "Report generated successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to generate report: {str(e)}")

    def open_web_app(self):
        """Open the OpenLayers web app in the default browser."""
        QDesktopServices.openUrl(QUrl("LOCAL HOST PATH")) # ADD YOUR LOCALHOST PATH FOR WEB APPLICATION HERE

    def connect_postgis(self):
        """Connect to the PostGIS database and display available tables."""
        try:
            # Connect to the database
            conn = psycopg2.connect(
                dbname="DATABASE NAME",
                user="USERNAME",
                password="PASSWORD",
                host="localhost",
                port="5432"
            )
            cursor = conn.cursor()

            # Fetch all table names
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            tables = cursor.fetchall()

            if not tables:
                QMessageBox.warning(self, "No Tables Found", "No tables found in the database.")
                return

            # Create a dialog to select a table
            dialog = QDialog(self)
            dialog.setWindowTitle("Select PostGIS Table")
            layout = QVBoxLayout()

            table_label = QLabel("Select a table:")
            layout.addWidget(table_label)

            table_combo = QComboBox()
            for table in tables:
                table_combo.addItem(table[0])
            layout.addWidget(table_combo)

            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            dialog.setLayout(layout)

            if dialog.exec_() == QDialog.Accepted:
                selected_table = table_combo.currentText()
                self.display_table_data(conn, selected_table)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to connect to PostGIS database: {str(e)}")

    def display_table_data(self, conn, table_name):
        """Display data from the selected PostGIS table and visualize it."""
        try:
            cursor = conn.cursor()

            # Fetch data from the selected table
            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()
            colnames = [desc[0] for desc in cursor.description]

            # Create a table widget to display the data
            table_widget = QTableWidget()
            table_widget.setRowCount(len(rows))
            table_widget.setColumnCount(len(colnames))
            table_widget.setHorizontalHeaderLabels(colnames)

            for row_idx, row in enumerate(rows):
                for col_idx, value in enumerate(row):
                    table_widget.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

            # Show the table widget in a new dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Data from {table_name}")
            layout = QVBoxLayout()
            layout.addWidget(table_widget)

            # Add a button to visualize the data
            visualize_btn = QPushButton("Visualize Data")
            visualize_btn.clicked.connect(lambda: self.visualize_table_data(conn, table_name))
            layout.addWidget(visualize_btn)

            # Add a button to add the selected shapefile to the application
            add_shapefile_btn = QPushButton("Add Selected Shapefile")
            add_shapefile_btn.clicked.connect(lambda: self.add_shapefile_to_application(conn, table_name))
            layout.addWidget(add_shapefile_btn)

            # Add a button to add the selected raster to the application
            add_raster_btn = QPushButton("Add Selected Raster")
            add_raster_btn.clicked.connect(lambda: self.add_raster_to_application(conn, table_name))
            layout.addWidget(add_raster_btn)

            dialog.setLayout(layout)
            dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch data from table {table_name}: {str(e)}")

    def add_shapefile_to_application(self, conn, table_name):
        """Add the selected shapefile to the application and display it as a raster layer."""
        try:
            # Use SQLAlchemy engine for GeoPandas
            engine = create_engine(f"postgresql+psycopg2://postgres:Postgre@localhost:5432/DATABASE NAME") # INSERT DATABASE NAME

            # Use GeoPandas to fetch spatial data
            query = f"SELECT * FROM {table_name};"
            gdf = gpd.read_postgis(query, engine, geom_col='geom')

            # Set a default CRS if not defined
            if gdf.crs is None:
                print("CRS is not defined. Setting default CRS to EPSG:4326.") # INSERT YOUR CRS ACCORDING TO COUNTRY
                gdf.set_crs("EPSG:4326", inplace=True)

            # Reproject to WGS84 (EPSG:4326) if not already
            gdf = gdf.to_crs("EPSG:4326")

            # Log CRS and bounds for debugging
            if gdf.crs:
                print(f"CRS of the shapefile: {gdf.crs}")
            else:
                print("CRS is not defined for the shapefile.")

            print(f"Bounds of the shapefile: {gdf.total_bounds}")

            # Validate and fix geometries
            if 'geometry' in gdf.columns:
                gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
            elif 'geom' in gdf.columns:
                gdf['geom'] = gdf['geom'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
            else:
                raise ValueError("No valid geometry column found in the shapefile data. Ensure the shapefile has a 'geometry' or 'geom' column.")

            # Remove rows with invalid geometries after attempting repair
            gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid]
            if gdf.empty:
                raise ValueError("All geometries are invalid or could not be repaired.")

            # Ensure the geometry column is correctly set
            if 'geometry' in gdf.columns:
                gdf = gdf.set_geometry('geometry')
            elif 'geom' in gdf.columns:
                gdf = gdf.set_geometry('geom')
            else:
                raise ValueError("Failed to set geometry column. Ensure the shapefile has a valid geometry column.")

            # Log the geometry column being used
            print(f"Using geometry column: {gdf.geometry.name}")

            # Rasterize the shapefile data
            bounds = gdf.total_bounds  # Get the bounding box of the shapefile
            resolution = 0.01  # Define the resolution of the raster
            transform = rasterio.transform.from_bounds(*bounds, int((bounds[2] - bounds[0]) / resolution), int((bounds[3] - bounds[1]) / resolution))

            out_shape = (int((bounds[3] - bounds[1]) / resolution), int((bounds[2] - bounds[0]) / resolution))
            raster = features.rasterize(
                ((geom, 1) for geom in gdf.geometry),
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype='uint8'
            )

            # Save the rasterized data to a temporary file
            raster_path = f"{table_name}_raster.tif"
            with rasterio.open(
                raster_path, 'w',
                driver='GTiff',
                height=raster.shape[0],
                width=raster.shape[1],
                count=1,
                dtype=raster.dtype,
                crs="EPSG:4326",
                transform=transform
            ) as dst:
                dst.write(raster, 1)

            # Add the rasterized shapefile to the layer panel
            layer_name = table_name
            self.raster_layers.append({
                "name": layer_name,
                "data": raster_path,
                "cmap": 'viridis',
                "visible": True
            })
            self.layer_panel.addItem(layer_name)

            # Update the map display
            self.update_map_with_raster(raster_path)

            QMessageBox.information(self, "Success", f"Shapefile '{layer_name}' added to the application as a raster layer.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add shapefile '{table_name}' to the application: {str(e)}")

    def visualize_table_data(self, conn, table_name):
        """Visualize spatial data from the selected PostGIS table."""
        try:
            # Use GeoPandas to fetch and visualize spatial data
            query = f"SELECT * FROM {table_name};"
            gdf = gpd.read_postgis(query, conn, geom_col='geom')

            # Plot the data
            gdf.plot(color='green', edgecolor='black')
            plt.title(f"Visualization of {table_name}")
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to visualize data from table {table_name}: {str(e)}")

    def open_raster_app(self):
        """Open the Raster App in the default browser."""
        QDesktopServices.openUrl(QUrl("LOCAL HOST PATH")) # INSERT YOUR LOCALHOST PATH FOR RASTER WEB APPLICATION

    def add_raster_to_application(self, conn, table_name):
        """Add the selected raster to the application and display it in the viewer."""
        try:
            # Fetch raster data from the database
            query = f"SELECT rid, ST_AsGDALRaster(rast, 'GTiff') AS raster_data FROM {table_name};"
            cursor = conn.cursor()
            cursor.execute(query)
            raster_data = cursor.fetchone()

            if raster_data:
                rid, raster_blob = raster_data
                raster_path = os.path.join(UPLOAD_FOLDER, f"{table_name}_raster_{rid}.tif")

                # Save the raster blob to a file
                with open(raster_path, 'wb') as f:
                    f.write(raster_blob)

                # Add the raster to the layer panel and map viewer
                layer_name = f"{table_name}_raster_{rid}"
                
                # Read raster data and include it in the layer
                with rasterio.open(raster_path) as src:
                    raster_data = src.read(1)

                self.raster_layers.append({
                    "name": layer_name,
                    "path": raster_path,
                    "data": raster_data,
                    "visible": True,
                    "cmap": 'viridis'
                })
                self.layer_panel.addItem(layer_name)

                # Update the map display
                self.update_map_with_raster(raster_path)

                # Notify the user
                QMessageBox.information(self, "Success", f"Raster '{table_name}' added to the application.")

                # Update the map display
                self.update_map()
            else:
                QMessageBox.warning(self, "No Data", f"No raster data found in table '{table_name}'.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add raster '{table_name}' to the application: {str(e)}")

    def update_map_with_raster(self, raster_path):
        """Update the map display with the given raster file."""
        try:
            with rasterio.open(raster_path) as src:
                raster_data = src.read(1)
                extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

            # Display the raster on the map canvas
            self.map_display.ax.clear()
            self.map_display.ax.imshow(raster_data, extent=extent, cmap='viridis', origin='upper')
            self.map_display.ax.set_title("Raster Layer")
            self.map_display.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update map with raster: {str(e)}")


class FuzzyMembershipDialog(QDialog):
    def __init__(self, parent=None, raster_layers=None):
        super().__init__(parent)
        self.setWindowTitle("Fuzzy Sets")
        self.setFixedSize(900, 700)
        self.raster_layers = raster_layers
        
        # Simplified fuzzy sets definitions
        self.fuzzy_sets = {
            'rainfall': {
                'sets': ['VL', 'L', 'M', 'H', 'VH'],
                'centers': [0.5, 1.5, 2.5, 3.5, 4.5],
                'spreads': [0.4] * 5,
                'type': 'direct'  # Higher is worse
            },
            'elevation': {
                'sets': ['VL', 'L', 'M', 'H', 'VH'],
                'centers': [4.5, 3.5, 2.5, 1.5, 0.5],
                'spreads': [0.4] * 5,
                'type': 'inverse'  # Lower is worse
            },
            'lulc': {
                'sets': ['VL', 'L', 'M', 'H', 'VH'],
                'centers': [0.5, 1.5, 2.5, 3.5, 4.5],
                'spreads': [0.4] * 5,
                'type': 'direct'
            },
            'soil': {
                'sets': ['VL', 'L', 'M', 'H', 'VH'],
                'centers': [0.5, 1.5, 2.5, 3.5, 4.5],
                'spreads': [0.4] * 5,
                'type': 'direct'
            },
            'river_distance': {
                'sets': ['VL', 'L', 'M', 'H', 'VH'],
                'centers': [4.5, 3.5, 2.5, 1.5, 0.5],
                'spreads': [0.4] * 5,
                'type': 'inverse'
            }
        }
        
        self.fuzzy_rules = {
            'rainfall': {
                'description': 'Higher rainfall increases flood risk. Direct relationship:',
                'rules': [
                    'VL (0-1): Minimal rainfall, low flood risk',
                    'L  (1-2): Light rainfall',
                    'M  (2-3): Moderate rainfall', 
                    'H  (3-4): Heavy rainfall',
                    'VH (4-5): Intense rainfall, high flood risk'
                ]
            },
            'elevation': {
                'description': 'Lower elevation increases flood risk. Inverse relationship:',
                'rules': [
                    'VL (4-5): High elevation, low risk',
                    'L  (3-4): Moderately high elevation',
                    'M  (2-3): Medium elevation',
                    'H  (1-2): Low elevation',
                    'VH (0-1): Very low elevation, high risk'
                ]
            },
            'lulc': {
                'description': 'Built-up areas increase flood risk. Direct relationship:',
                'rules': [
                    'VL (0-1): Forest/dense vegetation',
                    'L  (1-2): Mixed vegetation',
                    'M  (2-3): Agricultural land',
                    'H  (3-4): Sparse vegetation',
                    'VH (4-5): Built-up/impervious areas'
                ]
            },
            'soil': {
                'description': 'Less permeable soils increase flood risk. Direct relationship:',
                'rules': [
                    'VL (0-1): Sandy/high permeability',
                    'L  (1-2): Sandy loam',
                    'M  (2-3): Loam',
                    'H  (3-4): Clay loam',
                    'VH (4-5): Clay/low permeability'
                ]
            },
            'river_distance': {
                'description': 'Closer to rivers means higher flood risk. Inverse relationship:',
                'rules': [
                    'VL (4-5): Far from river',
                    'L  (3-4): Moderately far',
                    'M  (2-3): Medium distance',
                    'H  (1-2): Close to river',
                    'VH (0-1): Very close to river'
                ]
            }
        }
        
        # Setup UI with tabs
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.setup_membership_plot()
        self.setup_info_tab()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def setup_info_tab(self):
        info_tab = QWidget()
        layout = QVBoxLayout()
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        
        html = """<h2>Fuzzy Factor Rules</h2>"""
        for factor, info in self.fuzzy_rules.items():
            html += f"""
            <h3>{factor.upper()}</h3>
            <p>{info['description']}</p>
            <ul>
            """
            for rule in info['rules']:
                html += f"<li>{rule}</li>"
            html += "</ul>"
            
        info_text.setHtml(html)
        layout.addWidget(info_text)
        info_tab.setLayout(layout)
        self.tabs.addTab(info_tab, "Factor Rules")

    def setup_membership_plot(self):
        plot_tab = QWidget()
        layout = QVBoxLayout()
        
        # Create plots
        self.canvas = FigureCanvas(Figure(figsize=(10, 12)))
        self.canvas.figure.subplots_adjust(hspace=0.4)
        
        colors = ['#4575b4', '#74add1', '#fee090', '#f46d43', '#d73027']
        
        for i, (factor, props) in enumerate(self.fuzzy_sets.items()):
            ax = self.canvas.figure.add_subplot(5, 1, i+1)
            x = np.linspace(0, 5, 200)
            
            for j, (set_name, center, spread) in enumerate(zip(props['sets'], props['centers'], props['spreads'])):
                y = np.exp(-((x - center) ** 2) / (2 * spread ** 2))
                if props['type'] == 'inverse':
                    y = 1 - y
                ax.plot(x, y, label=set_name, color=colors[j], linewidth=2)
            
            ax.set_title(factor.upper(), fontsize=10, pad=5)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, 5)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=5, loc='upper center')
        
        layout.addWidget(self.canvas)
        plot_tab.setLayout(layout)
        self.tabs.addTab(plot_tab, "Fuzzy Sets")

    def apply_fuzzy_logic(self, normalized_layers, weights):
        """Enhanced fuzzy overlay with improved aggregation"""
        fuzzy_results = []
        
        # Step 1: Calculate fuzzy memberships
        for layer, (factor, props) in zip(normalized_layers, self.fuzzy_sets.items()):
            factor_memberships = np.zeros((len(props['sets']), *layer.shape))
            
            # Calculate membership for each fuzzy set
            for i, (center, spread) in enumerate(zip(props['centers'], props['spreads'])):
                membership = np.exp(-((layer - center) ** 2) / (2 * spread ** 2))
                if props['type'] == 'inverse':
                    membership = 1 - membership
                factor_memberships[i] = membership
            
            # Get max membership value for each pixel
            fuzzy_results.append(np.max(factor_memberships, axis=0))
        
        # Step 2: Weight and combine using improved operator
        gamma = 0.7
        product = np.ones_like(normalized_layers[0])
        algebraic_sum = np.zeros_like(normalized_layers[0])
        
        for result, weight in zip(fuzzy_results, weights):
            # Apply weight to membership values
            weighted_result = result ** weight
            product *= weighted_result
            algebraic_sum += weighted_result
        
        # Apply gamma operator for final combination
        combined = (product ** (1 - gamma)) * (algebraic_sum ** gamma)
        
        # Step 3: Defuzzify to risk levels with improved scaling
        min_val = np.nanmin(combined)
        max_val = np.nanmax(combined)
        flood_risk = 1 + 4 * ((combined - min_val) / (max_val - min_val))
        
        # Apply final smoothing
        flood_risk = np.clip(flood_risk, 1, 5)
        
        return flood_risk


# Import necessary libraries
from flask import Flask, render_template, request, jsonify, send_from_directory
import folium
import os
from werkzeug.utils import secure_filename

# Configure Flask app
app = Flask(__name__, static_url_path='/uploads', static_folder='uploads')
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'tif', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import rasterio
from rasterio import features

import logging

# Configure logging for Flask app
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@app.before_request
def log_startup():
    if not hasattr(app, 'startup_logged'):
        logging.info("Flask app is starting...")
        app.startup_logged = True

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"An error occurred: {e}")
    return jsonify({'error': 'An internal error occurred'}), 500

@app.route('/')
def index():
    # Create a Folium map centered at an example location
    m = folium.Map(location=[29.7604, -95.3698], zoom_start=10)  # Example coordinates for Houston, TX

    # Add raster overlays from the uploads folder
    raster_files = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
    for raster in raster_files:
        raster_path = os.path.join(UPLOAD_FOLDER, raster)
        try:
            with rasterio.open(raster_path) as src:
                bounds = src.bounds
                image_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

                # Add the raster as an overlay on the map
                folium.raster_layers.ImageOverlay(
                    image=raster_path,
                    bounds=image_bounds,
                    opacity=0.6
                ).add_to(m)
        except Exception as e:
            print(f"Error adding raster {raster}: {e}")

    # Save the map to an HTML file and render it
    map_path = os.path.join('templates', 'map.html')
    m.save(map_path)
    return render_template('map.html')

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    """Serve files from the uploads directory."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'filePath': file_path}), 200

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/raster_map')
def raster_map():
    """Serve the raster_map.html file."""
    return render_template('raster_map.html')

def start_flask_server():
    """Start the Flask server in a separate thread."""
    global app
    app.run(port=5001, debug=True, use_reloader=False)

if __name__ == '__main__':
    import threading

    # Start the Flask server in a separate thread
    flask_thread = threading.Thread(target=start_flask_server)
    flask_thread.daemon = True
    flask_thread.start()

    # Start the PyQt application
    app = QApplication(sys.argv)
    window = FloodMappingApp()
    window.show()
    sys.exit(app.exec_())


