# Canon MRI Interface Color Scheme
CANON_COLORS = {
    # Main backgrounds
    'main_bg': '#0a0e1a',  # Very dark blue
    'panel_bg': '#1a2332',  # Dark blue panels
    'section_bg': '#2c3e50',  # Slightly lighter for sections
    
    # Accent colors
    'highlight_blue': '#4a90e2',  # Bright blue for selections
    'active_blue': '#3498db',  # Active elements
    'header_gradient_start': '#2c5aa0',
    'header_gradient_end': '#1e3a5f',
    
    # Text colors
    'text_white': '#ffffff',
    'text_light': '#e0e0e0',
    'text_dim': '#a0a0a0',
    'text_value': '#ffffff',
    
    # Input fields
    'input_bg': '#0a1422',
    'input_border': '#2c4158',
    'input_focus': '#4a90e2',
    
    # Buttons
    'button_bg': '#2c4158',
    'button_hover': '#3a5169',
    'button_pressed': '#1e2a3a',
    'button_text': '#ffffff',
    
    # Special elements
    'edit_button': '#4a90e2',
    'dropdown_arrow': '#8a8a8a',
    'separator': '#2c4158',
}

# Parameter definitions matching Canon interface
PARAMETER_DEFINITIONS = {
    'TR': {'name': 'TR', 'category': 'Basic', 'type': 'float', 'unit': 'ms'},
    'TE': {'name': 'TE', 'category': 'Basic', 'type': 'float', 'unit': 'ms'},
    'FLEXTE': {'name': 'TE', 'category': 'Basic', 'type': 'float', 'unit': 'ms'},
    'FLIP': {'name': 'Flip/Flop', 'category': 'Basic', 'type': 'str'},
    'FOV': {'name': 'FOV(cm)', 'category': 'Geometry', 'type': 'str'},
    'MTX': {'name': 'Matrix', 'category': 'Geometry', 'type': 'str'},
    'THK': {'name': 'Thick.(mm)', 'category': 'Geometry', 'type': 'float'},
    'GAP': {'name': 'Gap(mm)', 'category': 'Geometry', 'type': 'float'},
    'TS': {'name': 'Num.', 'category': 'Geometry', 'type': 'int'},
    'NAQ': {'name': 'NAQ', 'category': 'Basic', 'type': 'int'},
    'TACQT': {'name': 'Time', 'category': 'Info', 'type': 'str'},
    'PLN': {'name': 'Plane', 'category': 'Geometry', 'type': 'str'},
    'SEQID': {'name': 'Seq.', 'category': 'Info', 'type': 'str'},
}

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from canon_colors import CANON_COLORS

# Global stylesheet for Canon look
CANON_GLOBAL_STYLE = """
QWidget {
    background-color: #0a0e1a;
    color: #ffffff;
    font-family: Arial, sans-serif;
    font-size: 11px;
}

QGroupBox {
    background-color: #1a2332;
    border: 1px solid #2c4158;
    border-radius: 5px;
    margin-top: 15px;
    padding-top: 15px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
    color: #e0e0e0;
}

QLabel {
    color: #e0e0e0;
    background-color: transparent;
}

QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #0a1422;
    border: 1px solid #2c4158;
    border-radius: 3px;
    padding: 4px;
    color: #ffffff;
    selection-background-color: #4a90e2;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #4a90e2;
}

QPushButton {
    background-color: #2c4158;
    border: 1px solid #3a5169;
    border-radius: 3px;
    padding: 5px 15px;
    color: #ffffff;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #3a5169;
}

QPushButton:pressed {
    background-color: #1e2a3a;
}

QPushButton#edit_button {
    background-color: #4a90e2;
    border: 1px solid #5ba0f2;
    padding: 3px 10px;
    font-size: 10px;
}

QComboBox {
    background-color: #0a1422;
    border: 1px solid #2c4158;
    border-radius: 3px;
    padding: 4px;
    color: #ffffff;
    min-width: 80px;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #8a8a8a;
    margin-right: 5px;
}

QTabWidget::pane {
    background-color: #1a2332;
    border: 1px solid #2c4158;
}

QTabBar::tab {
    background-color: #1a2332;
    color: #a0a0a0;
    padding: 8px 20px;
    border: 1px solid #2c4158;
    border-bottom: none;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #2c4158;
    color: #ffffff;
    border: 2px solid #4a90e2;
}

QScrollArea {
    background-color: #0a0e1a;
    border: none;
}

QScrollBar:vertical {
    background-color: #1a2332;
    width: 12px;
    border: none;
}

QScrollBar::handle:vertical {
    background-color: #2c4158;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #3a5169;
}
"""

class CanonHeaderWidget(QWidget):
    """Canon-style header with gradient background"""
    def __init__(self, title="", show_time=True):
        super().__init__()
        self.title = title
        self.show_time = show_time
        self.setup_ui()
        
    def setup_ui(self):
        self.setFixedHeight(40)
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        self.setLayout(layout)
        
        # Title
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("""
            color: #ffffff;
            font-size: 14px;
            font-weight: bold;
        """)
        layout.addWidget(self.title_label)
        
        layout.addStretch()
        
        # Status indicators
        if self.show_time:
            self.time_label = QLabel("Time: 4:20")
            self.time_label.setStyleSheet("color: #ffffff; font-size: 12px;")
            layout.addWidget(self.time_label)
            
            self.cover_label = QLabel("Cover: 1")
            self.cover_label.setStyleSheet("color: #ffffff; font-size: 12px; margin-left: 20px;")
            layout.addWidget(self.cover_label)
            
            self.rf_label = QLabel("RF: ----%")
            self.rf_label.setStyleSheet("color: #ffffff; font-size: 12px; margin-left: 20px;")
            layout.addWidget(self.rf_label)
            
    def paintEvent(self, event):
        """Paint gradient background"""
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(CANON_COLORS['header_gradient_start']))
        gradient.setColorAt(1, QColor(CANON_COLORS['header_gradient_end']))
        painter.fillRect(self.rect(), gradient)

class CanonParameterPanel(QWidget):
    """Panel for displaying parameters in Canon style"""
    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        
        # Content area
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet(f"""
            background-color: {CANON_COLORS['panel_bg']};
            border: 1px solid {CANON_COLORS['input_border']};
            border-radius: 3px;
            padding: 10px;
        """)
        self.content_layout = QGridLayout()
        self.content_layout.setSpacing(8)
        self.content_widget.setLayout(self.content_layout)
        
        layout.addWidget(self.content_widget)
        
    def add_parameter(self, label, value, row, editable=True):
        """Add a parameter to the panel"""
        # Label
        param_label = QLabel(label)
        param_label.setStyleSheet("""
            color: #e0e0e0;
            font-size: 11px;
            padding-right: 5px;
        """)
        param_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # Value widget
        if editable:
            value_widget = QLineEdit(str(value))
            value_widget.setMinimumWidth(80)
        else:
            value_widget = QLabel(str(value))
            value_widget.setStyleSheet("""
                background-color: #0a1422;
                border: 1px solid #2c4158;
                border-radius: 3px;
                padding: 4px;
                color: #ffffff;
                min-width: 80px;
            """)
            
        self.content_layout.addWidget(param_label, row, 0)
        self.content_layout.addWidget(value_widget, row, 1)
        
        return value_widget

def apply_canon_theme(app):
    """Apply Canon theme to entire application"""
    app.setStyleSheet(CANON_GLOBAL_STYLE)
    
    # Set application palette for native widgets
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(CANON_COLORS['main_bg']))
    palette.setColor(QPalette.WindowText, QColor(CANON_COLORS['text_white']))
    palette.setColor(QPalette.Base, QColor(CANON_COLORS['input_bg']))
    palette.setColor(QPalette.AlternateBase, QColor(CANON_COLORS['panel_bg']))
    palette.setColor(QPalette.Text, QColor(CANON_COLORS['text_white']))
    palette.setColor(QPalette.Button, QColor(CANON_COLORS['button_bg']))
    palette.setColor(QPalette.ButtonText, QColor(CANON_COLORS['text_white']))
    palette.setColor(QPalette.Highlight, QColor(CANON_COLORS['highlight_blue']))
    palette.setColor(QPalette.HighlightedText, QColor(CANON_COLORS['text_white']))
    
    app.setPalette(palette)

    import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import os

@dataclass
class ProtocolParameter:
    """Represents a single parameter in a Canon protocol"""
    name: str
    value: Any
    unit: str = ""
    category: str = "Basic"
    editable: bool = True
    description: str = ""
    
    def __str__(self):
        if self.unit:
            return f"{self.value} {self.unit}"
        return str(self.value)

@dataclass
class CanonProtocol:
    """Represents a complete Canon MRI protocol"""
    name: str
    sequence_id: str
    parameters: Dict[str, ProtocolParameter]
    metadata: Dict[str, str]
    
    def get_parameter(self, key: str) -> Optional[ProtocolParameter]:
        """Get parameter by key"""
        return self.parameters.get(key)
    
    def get_parameters_by_category(self, category: str) -> List[ProtocolParameter]:
        """Get all parameters in a specific category"""
        return [p for p in self.parameters.values() if p.category == category]
    
    def get_categories(self) -> List[str]:
        """Get all unique categories"""
        categories = set(p.category for p in self.parameters.values())
        # Return in preferred order
        preferred_order = ['Basic', 'Geometry', 'Contrast', 'Motion', 'Advanced', 'Info']
        ordered = [c for c in preferred_order if c in categories]
        # Add any remaining categories
        remaining = sorted(categories - set(ordered))
        return ordered + remaining

class CanonXMLParser:
    """Parser for Canon PAS XML protocol files"""
    
    def __init__(self):
        from canon_colors import PARAMETER_DEFINITIONS
        self.param_definitions = PARAMETER_DEFINITIONS
        
    def parse_file(self, file_path: str) -> CanonProtocol:
        """Parse a Canon PAS XML file"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract protocol name
            protocol_name = os.path.basename(file_path).replace('.xml', '')
            
            # Extract sequence ID
            sequence_id = self._extract_text(root, './/SEQID', 'Unknown')
            
            # Parse all parameters
            parameters = self._parse_parameters(root)
            
            # Extract metadata
            metadata = self._parse_metadata(root)
            
            return CanonProtocol(
                name=protocol_name,
                sequence_id=sequence_id,
                parameters=parameters,
                metadata=metadata
            )
            
        except Exception as e:
            raise Exception(f"Error parsing Canon XML file: {str(e)}")
    
    def _parse_parameters(self, root: ET.Element) -> Dict[str, ProtocolParameter]:
        """Parse all parameters from XML"""
        parameters = {}
        
        # Common parameter paths in Canon XML
        param_paths = {
            # Basic parameters
            'TR': './/TR',
            'TE': './/TE',
            'FLEXTE': './/FLEXTE',
            'FLIP': './/FLIP',
            'NAQ': './/NAQ',
            
            # Geometry
            'FOV': './/FOV',
            'MTX': './/MTX',
            'THK': './/THK',
            'GAP': './/GAP',
            'TS': './/TS',
            'PLN': './/PLN',
            
            # Sequence info
            'SEQID': './/SEQID',
            'TACQT': './/TACQT',
            
            # Additional parameters you might find
            'ETL': './/ETL',
            'BW': './/BW',
            'FA': './/FA',
            'TI': './/TI',
            'NEX': './/NEX',
        }
        
        for param_key, xpath in param_paths.items():
            value = self._extract_text(root, xpath)
            if value is not None:
                # Get parameter definition
                param_def = self.param_definitions.get(param_key, {})
                
                # Create parameter
                param = ProtocolParameter(
                    name=param_def.get('name', param_key),
                    value=self._convert_value(value, param_def.get('type', 'str')),
                    unit=param_def.get('unit', ''),
                    category=param_def.get('category', 'Basic'),
                    editable=param_def.get('editable', True)
                )
                
                parameters[param_key] = param
        
        # Parse FOV as special case (might be formatted as "20x20")
        if 'FOV' in parameters and 'x' in str(parameters['FOV'].value):
            parameters['FOV'].value = parameters['FOV'].value.replace('x', ' x ')
        
        # Parse Matrix as special case (might be formatted as "256x256")
        if 'MTX' in parameters and 'x' in str(parameters['MTX'].value):
            parameters['MTX'].value = parameters['MTX'].value.replace('x', ' x ')
            
        return parameters
    
    def _parse_metadata(self, root: ET.Element) -> Dict[str, str]:
        """Parse metadata from XML"""
        metadata = {}
        
        # Common metadata fields
        metadata_paths = {
            'StudyDate': './/StudyDate',
            'StudyTime': './/StudyTime',
            'PatientName': './/PatientName',
            'PatientID': './/PatientID',
            'Modality': './/Modality',
            'Manufacturer': './/Manufacturer',
            'InstitutionName': './/InstitutionName',
            'StationName': './/StationName',
        }
        
        for key, xpath in metadata_paths.items():
            value = self._extract_text(root, xpath)
            if value:
                metadata[key] = value
                
        return metadata
    
    def _extract_text(self, root: ET.Element, xpath: str, default: Optional[str] = None) -> Optional[str]:
        """Extract text from XML element"""
        element = root.find(xpath)
        if element is not None and element.text:
            return element.text.strip()
        return default
    
    def _convert_value(self, value: str, value_type: str) -> Any:
        """Convert string value to appropriate type"""
        try:
            if value_type == 'int':
                return int(value)
            elif value_type == 'float':
                return float(value)
            else:
                return value
        except:
            return value

def create_sample_xml(file_path: str):
    """Create a sample Canon PAS XML file for testing"""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<Protocol>
    <StudyDate>20240115</StudyDate>
    <StudyTime>143052</StudyTime>
    <Modality>MR</Modality>
    <Manufacturer>Canon</Manufacturer>
    <SEQID>FSE-T2</SEQID>
    <TR>4000</TR>
    <TE>100</TE>
    <FLIP>90/180</FLIP>
    <FOV>24x24</FOV>
    <MTX>256x256</MTX>
    <THK>5.0</THK>
    <GAP>1.0</GAP>
    <TS>20</TS>
    <NAQ>1</NAQ>
    <TACQT>2:40</TACQT>
    <PLN>AX</PLN>
    <ETL>16</ETL>
    <BW>130</BW>
</Protocol>"""
    
    with open(file_path, 'w') as f:
        f.write(xml_content)
    print(f"Sample XML file created: {file_path}")
    import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from canon_colors import CANON_COLORS, PARAMETER_DEFINITIONS
from canon_style import (
    apply_canon_theme, 
    CanonHeaderWidget, 
    CanonParameterPanel
)
from canon_xml_parser import (
    CanonXMLParser, 
    CanonProtocol, 
    create_sample_xml
)

class CanonProtocolViewer(QMainWindow):
    """Main application window for Canon Protocol Viewer"""
    
    def __init__(self):
        super().__init__()
        self.current_protocol = None
        self.parser = CanonXMLParser()
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Canon MRI Protocol Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)
        
        # Add header
        self.header = CanonHeaderWidget("Protocol Viewer", show_time=True)
        main_layout.addWidget(self.header)
        
        # Toolbar
        self.create_toolbar()
        
        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget {
                background-color: #0a0e1a;
            }
        """)
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_basic_tab()
        self.create_advanced_tab()
        self.create_table_view_tab()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"""
            background-color: {CANON_COLORS['panel_bg']};
            color: {CANON_COLORS['text_light']};
            border-top: 1px solid {CANON_COLORS['separator']};
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def create_toolbar(self):
        """Create application toolbar"""
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setStyleSheet(f"""
            QToolBar {{
                background-color: {CANON_COLORS['panel_bg']};
                border: none;
                spacing: 10px;
                padding: 5px;
            }}
            QToolButton {{
                background-color: {CANON_COLORS['button_bg']};
                border: 1px solid {CANON_COLORS['input_border']};
                border-radius: 3px;
                padding: 5px 10px;
                color: white;
                font-weight: bold;
            }}
            QToolButton:hover {{
                background-color: {CANON_COLORS['button_hover']};
            }}
        """)
        
        # Actions
        open_action = QAction("Open Protocol", self)
        open_action.triggered.connect(self.open_protocol)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        save_action = QAction("Save Protocol", self)
        save_action.triggered.connect(self.save_protocol)
        toolbar.addAction(save_action)
        
        export_action = QAction("Export", self)
        export_action.triggered.connect(self.export_protocol)
        toolbar.addAction(export_action)
        
        toolbar.addSeparator()
        
        sample_action = QAction("Load Sample", self)
        sample_action.triggered.connect(self.load_sample)
        toolbar.addAction(sample_action)
        
    def create_basic_tab(self):
        """Create the Basic parameters tab"""
        basic_widget = QWidget()
        basic_widget.setStyleSheet(f"background-color: {CANON_COLORS['main_bg']};")
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(basic_widget)
        
        # Layout
        layout = QVBoxLayout()
        layout.setSpacing(15)
        basic_widget.setLayout(layout)
        
        # Basic parameters panel
        self.basic_panel = CanonParameterPanel("Basic Parameters")
        layout.addWidget(self.basic_panel)
        
        # Geometry panel
        self.geometry_panel = CanonParameterPanel("Geometry")
        layout.addWidget(self.geometry_panel)
        
        # Contrast panel
        self.contrast_panel = CanonParameterPanel("Contrast")
        layout.addWidget(self.contrast_panel)
        
        layout.addStretch()
        
        self.tab_widget.addTab(scroll, "Basic")
        
    def create_advanced_tab(self):
        """Create the Advanced parameters tab"""
        advanced_widget = QWidget()
        advanced_widget.setStyleSheet(f"background-color: {CANON_COLORS['main_bg']};")
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(advanced_widget)
        
        # Layout
        layout = QVBoxLayout()
        layout.setSpacing(15)
        advanced_widget.setLayout(layout)
        
        # Advanced panels
        self.motion_panel = CanonParameterPanel("Motion")
        layout.addWidget(self.motion_panel)
        
        self.advanced_panel = CanonParameterPanel("Advanced")
        layout.addWidget(self.advanced_panel)
        
        self.sequence_panel = CanonParameterPanel("Sequence Info")
        layout.addWidget(self.sequence_panel)
        
        layout.addStretch()
        
        self.tab_widget.addTab(scroll, "Advanced")
        
    def create_table_view_tab(self):
        """Create the table view tab"""
        # Table widget
        self.table_widget = QTableWidget()
        self.table_widget.setStyleSheet(f"""
            QTableWidget {{
                background-color: {CANON_COLORS['main_bg']};
                gridline-color: {CANON_COLORS['separator']};
                color: {CANON_COLORS['text_white']};
            }}
            QTableWidget::item {{
                padding: 5px;
                border-bottom: 1px solid {CANON_COLORS['separator']};
            }}
            QTableWidget::item:selected {{
                background-color: {CANON_COLORS['highlight_blue']};
            }}
            QHeaderView::section {{
                background-color: {CANON_COLORS['panel_bg']};
                color: {CANON_COLORS['text_white']};
                padding: 5px;
                border: 1px solid {CANON_COLORS['separator']};
                font-weight: bold;
            }}
        """)
        
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["Parameter", "Value", "Unit", "Category"])
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.setSortingEnabled(True)
        
        self.tab_widget.addTab(self.table_widget, "Table View")
        
    def open_protocol(self):
        """Open a Canon protocol XML file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Canon Protocol", 
            "", 
            "XML Files (*.xml);;All Files (*)"
        )
        
        if file_path:
            try:
                self.current_protocol = self.parser.parse_file(file_path)
                self.update_display()
                self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load protocol:\n{str(e)}")
                
    def load_sample(self):
        """Load a sample protocol for testing"""
        sample_path = "sample_protocol.xml"
        create_sample_xml(sample_path)
        
        try:
            self.current_protocol = self.parser.parse_file(sample_path)
            self.update_display()
            self.status_bar.showMessage("Loaded sample protocol")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load sample:\n{str(e)}")
            
    def update_display(self):
        """Update all displays with current protocol data"""
        if not self.current_protocol:
            return
            
        # Clear existing parameters
        self.clear_panels()
        
        # Update header
        seq_id = self.current_protocol.sequence_id
        self.header.title_label.setText(f"Protocol: {self.current_protocol.name} - {seq_id}")
        
        # Update panels by category
        self.populate_panels()
        
        # Update table view
        self.populate_table()
        
    def clear_panels(self):
        """Clear all parameter panels"""
        for panel in [self.basic_panel, self.geometry_panel, self.contrast_panel,
                     self.motion_panel, self.advanced_panel, self.sequence_panel]:
            # Clear the grid layout
            while panel.content_layout.count():
                item = panel.content_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                    
    def populate_panels(self):
        """Populate panels with protocol parameters"""
        if not self.current_protocol:
            return
            
        # Map panels to categories
        panel_map = {
            'Basic': self.basic_panel,
            'Geometry': self.geometry_panel,
            'Contrast': self.contrast_panel,
            'Motion': self.motion_panel,
            'Advanced': self.advanced_panel,
            'Info': self.sequence_panel
        }
        
        # Add parameters to appropriate panels
        row_counts = {panel: 0 for panel in panel_map.values()}
        
        for param_key, param in self.current_protocol.parameters.items():
            panel = panel_map.get(param.category)
            if panel:
                row = row_counts[panel]
                panel.add_parameter(param.name, param.value, row)
                row_counts[panel] += 1
                
    def populate_table(self):
        """Populate table view with protocol parameters"""
        if not self.current_protocol:
            return
            
        params = list(self.current_protocol.parameters.values())
        self.table_widget.setRowCount(len(params))
        
        for row, param in enumerate(params):
            # Parameter name
            self.table_widget.setItem(row, 0, QTableWidgetItem(param.name))
            
            # Value
            self.table_widget.setItem(row, 1, QTableWidgetItem(str(param.value)))
            
            # Unit
            self.table_widget.setItem(row, 2, QTableWidgetItem(param.unit))
            
            # Category
            self.table_widget.setItem(row, 3, QTableWidgetItem(param.category))
            
        self.table_widget.resizeColumnsToContents()
        
    def save_protocol(self):
        """Save current protocol (placeholder)"""
        QMessageBox.information(self, "Save", "Save functionality will be implemented")
        
    def export_protocol(self):
        """Export protocol to different format (placeholder)"""
        QMessageBox.information(self, "Export", "Export functionality will be implemented")

def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Canon MRI Protocol Viewer")
    
    # Apply Canon theme
    apply_canon_theme(app)
    
    # Create and show main window
    viewer = CanonProtocolViewer()
    viewer.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
