import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Import the enhanced parser and widgets
# (In your single file, these would be included above)
from canon_enhanced_parser import *
from canon_compact_widgets import *

# Canon color definitions
CANON_COLORS = {
    'main_bg': '#0a0e1a',
    'panel_bg': '#1a2332',
    'section_bg': '#2c3e50',
    'highlight_blue': '#4a90e2',
    'active_blue': '#3498db',
    'header_gradient_start': '#2c5aa0',
    'header_gradient_end': '#1e3a5f',
    'text_white': '#ffffff',
    'text_light': '#e0e0e0',
    'text_dim': '#a0a0a0',
    'input_bg': '#0a1422',
    'input_border': '#2c4158',
    'button_bg': '#2c4158',
    'button_hover': '#3a5169',
    'edit_button': '#4a90e2',
    'separator': '#2c4158',
}

# Global stylesheet
CANON_GLOBAL_STYLE = """
QWidget {
    background-color: #0a0e1a;
    color: #ffffff;
    font-family: Arial, sans-serif;
    font-size: 11px;
}

QLabel {
    color: #e0e0e0;
    background-color: transparent;
}

QLineEdit {
    background-color: #0a1422;
    border: 1px solid #2c4158;
    border-radius: 3px;
    padding: 4px;
    color: #ffffff;
    selection-background-color: #4a90e2;
}

QLineEdit:focus {
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

QTableWidget {
    background-color: #0a0e1a;
    gridline-color: #2c4158;
    color: #ffffff;
}

QTableWidget::item {
    padding: 5px;
    border-bottom: 1px solid #2c4158;
}

QTableWidget::item:selected {
    background-color: #4a90e2;
}

QHeaderView::section {
    background-color: #1a2332;
    color: #ffffff;
    padding: 5px;
    border: 1px solid #2c4158;
    font-weight: bold;
}

QCheckBox {
    color: #e0e0e0;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #2c4158;
    background-color: #0a1422;
    border-radius: 3px;
}

QCheckBox::indicator:checked {
    background-color: #4a90e2;
}
"""

class CanonHeaderWidget(QWidget):
    """Canon-style header with gradient background"""
    def __init__(self, title=""):
        super().__init__()
        self.title = title
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
        self.time_label = QLabel("")
        self.time_label.setStyleSheet("color: #ffffff; font-size: 12px;")
        layout.addWidget(self.time_label)
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #ffffff; font-size: 12px; margin-left: 20px;")
        layout.addWidget(self.status_label)
            
    def paintEvent(self, event):
        """Paint gradient background"""
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(CANON_COLORS['header_gradient_start']))
        gradient.setColorAt(1, QColor(CANON_COLORS['header_gradient_end']))
        painter.fillRect(self.rect(), gradient)
        
    def update_info(self, protocol_name="", seq_count=0, total_time=""):
        """Update header information"""
        self.title_label.setText(f"Protocol: {protocol_name}")
        self.status_label.setText(f"Sequences: {seq_count}")
        if total_time:
            self.time_label.setText(f"Total Time: {total_time}")

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
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)
        
        # Add header
        self.header = CanonHeaderWidget("Protocol Viewer")
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
        self.create_sequence_view_tab()
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
        
        compare_action = QAction("Compare", self)
        compare_action.triggered.connect(self.compare_protocols)
        toolbar.addAction(compare_action)
        
    def create_sequence_view_tab(self):
        """Create the sequence view tab with compact display"""
        self.sequence_display = CanonProtocolDisplay()
        self.tab_widget.addTab(self.sequence_display, "Sequence View")
        
    def create_table_view_tab(self):
        """Create the table view tab"""
        # Table widget
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(6)
        self.table_widget.setHorizontalHeaderLabels([
            "Seq #", "Name", "Type", "Parameter", "Value", "Unit"
        ])
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
        sample_path = "sample_multi_sequence.xml"
        create_multi_sequence_sample_xml(sample_path)
        
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
            
        # Update header
        self.header.update_info(
            self.current_protocol.name,
            self.current_protocol.get_sequence_count()
        )
        
        # Update sequence display
        self.sequence_display.display_protocol(self.current_protocol)
        
        # Update table view
        self.populate_table()
        
    def populate_table(self):
        """Populate table view with protocol parameters"""
        if not self.current_protocol:
            return
            
        # Clear existing rows
        self.table_widget.setRowCount(0)
        
        # Add parameters from each sequence
        row = 0
        for seq in self.current_protocol.sequences:
            for param_key, param in seq.parameters.items():
                self.table_widget.insertRow(row)
                
                # Sequence number
                self.table_widget.setItem(row, 0, QTableWidgetItem(str(seq.number)))
                
                # Sequence name
                self.table_widget.setItem(row, 1, QTableWidgetItem(seq.name))
                
                # Sequence type
                self.table_widget.setItem(row, 2, QTableWidgetItem(seq.sequence_type))
                
                # Parameter name
                self.table_widget.setItem(row, 3, QTableWidgetItem(param.display_name))
                
                # Value
                self.table_widget.setItem(row, 4, QTableWidgetItem(str(param.value)))
                
                # Unit
                self.table_widget.setItem(row, 5, QTableWidgetItem(param.unit))
                
                row += 1
                
        self.table_widget.resizeColumnsToContents()
        
    def save_protocol(self):
        """Save current protocol (placeholder)"""
        QMessageBox.information(self, "Save", "Save functionality will be implemented")
        
    def export_protocol(self):
        """Export protocol to different format (placeholder)"""
        QMessageBox.information(self, "Export", "Export functionality will be implemented")
        
    def compare_protocols(self):
        """Compare multiple protocols (placeholder)"""
        QMessageBox.information(self, "Compare", "Protocol comparison will be implemented")

class SequenceInfoDialog(QDialog):
    """Dialog to show detailed information about detected sequences"""
    def __init__(self, protocol, parser_debug_info=None, parent=None):
        super().__init__(parent)
        self.protocol = protocol
        self.parser_debug_info = parser_debug_info
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Sequence Detection Information")
        self.setMinimumSize(800, 600)
        
        # Apply Canon styling to dialog
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {CANON_COLORS['main_bg']};
                color: {CANON_COLORS['text_white']};
            }}
        """)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Header
        header_label = QLabel(f"Protocol: {self.protocol.name}")
        header_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #ffffff;
            padding: 10px;
            background-color: #2c4158;
            border-radius: 3px;
        """)
        layout.addWidget(header_label)
        
        # Summary
        summary_label = QLabel(f"Total Sequences Found: {len(self.protocol.sequences)}")
        summary_label.setStyleSheet("""
            font-size: 12px;
            color: #4a90e2;
            padding: 5px;
        """)
        layout.addWidget(summary_label)
        
        # Table of sequences
        self.sequence_table = QTableWidget()
        self.sequence_table.setColumnCount(6)
        self.sequence_table.setHorizontalHeaderLabels([
            "#", "Name", "Type/ID", "Parameters", "Time", "Status"
        ])
        self.sequence_table.horizontalHeader().setStretchLastSection(True)
        
        # Populate table
        self.sequence_table.setRowCount(len(self.protocol.sequences))
        for idx, seq in enumerate(self.protocol.sequences):
            # Number
            self.sequence_table.setItem(idx, 0, QTableWidgetItem(str(seq.number)))
            
            # Name
            self.sequence_table.setItem(idx, 1, QTableWidgetItem(seq.name))
            
            # Type
            self.sequence_table.setItem(idx, 2, QTableWidgetItem(seq.sequence_type))
            
            # Parameter count
            param_count = len(seq.parameters)
            self.sequence_table.setItem(idx, 3, QTableWidgetItem(f"{param_count} params"))
            
            # Time
            time_str = seq.parameters.get('TACQT', SequenceParameter('', 'N/A')).value
            self.sequence_table.setItem(idx, 4, QTableWidgetItem(str(time_str)))
            
            # Status
            status = "Enabled" if seq.enabled else "Disabled"
            status_item = QTableWidgetItem(status)
            if seq.enabled:
                status_item.setForeground(QColor("#4a90e2"))
            else:
                status_item.setForeground(QColor("#a0a0a0"))
            self.sequence_table.setItem(idx, 5, status_item)
            
        self.sequence_table.resizeColumnsToContents()
        layout.addWidget(self.sequence_table)
        
        # Debug info text area
        if self.parser_debug_info:
            debug_label = QLabel("Parser Debug Information:")
            debug_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
            layout.addWidget(debug_label)
            
            debug_text = QTextEdit()
            debug_text.setReadOnly(True)
            debug_text.setMaximumHeight(150)
            debug_text.setPlainText(self.parser_debug_info)
            debug_text.setStyleSheet("""
                background-color: #0a1422;
                border: 1px solid #2c4158;
                font-family: monospace;
                font-size: 10px;
            """)
            layout.addWidget(debug_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #2c4158;
                border: 1px solid #3a5169;
                border-radius: 3px;
                padding: 8px 20px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a5169;
            }
        """)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

# Add this method to the main window class
def show_sequence_info(self):
    """Show detailed information about detected sequences"""
    if not self.current_protocol:
        QMessageBox.information(self, "No Protocol", "Please load a protocol first.")
        return
        
    dialog = SequenceInfoDialog(self.current_protocol, parent=self)
    dialog.exec_()

# Also update the parser to provide debug information
class DebugCapableParser(EnhancedCanonXMLParser):
    """Parser that captures debug information about sequence detection"""
    def __init__(self):
        super().__init__()
        self.debug_log = []
        
    def log_debug(self, message):
        """Add a debug message"""
        self.debug_log.append(message)
        if self.debug_mode:
            print(message)
            
    def get_debug_log(self):
        """Get the debug log as a string"""
        return "\n".join(self.debug_log)
        
    def clear_debug_log(self):
        """Clear the debug log"""
        self.debug_log = []
        
    def parse_file(self, file_path: str) -> CanonProtocol:
        """Parse with debug logging"""
        self.clear_debug_log()
        self.log_debug(f"Parsing file: {os.path.basename(file_path)}")
        
        result = super().parse_file(file_path)
        
        self.log_debug(f"Total sequences found: {len(result.sequences)}")
        for seq in result.sequences:
            self.log_debug(f"  - Sequence {seq.number}: {seq.name} [{seq.sequence_type}]")
            self.log_debug(f"    Parameters: {len(seq.parameters)}")
            
        return result
    
def create_toolbar(self):
    """Create application toolbar with sequence info button"""
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
    
    # File operations
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
    
    # Protocol operations
    sample_action = QAction("Load Sample", self)
    sample_action.triggered.connect(self.load_sample)
    toolbar.addAction(sample_action)
    
    compare_action = QAction("Compare", self)
    compare_action.triggered.connect(self.compare_protocols)
    toolbar.addAction(compare_action)
    
    toolbar.addSeparator()
    
    # Info and debug
    info_action = QAction("Sequence Info", self)
    info_action.setToolTip("Show detailed information about detected sequences")
    info_action.triggered.connect(self.show_sequence_info)
    toolbar.addAction(info_action)

# Enhanced sample XML with better sequence identification
def create_enhanced_sample_xml(file_path: str):
    """Create a sample Canon PAS XML file with clearly named sequences"""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<Protocol>
    <StudyDate>20240115</StudyDate>
    <StudyTime>143052</StudyTime>
    <Modality>MR</Modality>
    <Manufacturer>Canon</Manufacturer>
    <ProtocolName>Brain_Routine_with_Contrast</ProtocolName>
    
    <Sequence>
        <SEQNAME>3-Plane Localizer</SEQNAME>
        <Comment>Scout images for planning</Comment>
        <SEQID>FFE</SEQID>
        <TR>3.5</TR>
        <TE>1.8</TE>
        <FLIP>15</FLIP>
        <FOV>40x40</FOV>
        <MTX>128x128</MTX>
        <THK>10.0</THK>
        <GAP>0.0</GAP>
        <TS>3</TS>
        <NAQ>1</NAQ>
        <TACQT>0:15</TACQT>
        <PLN>3-plane</PLN>
        <BW>1447</BW>
    </Sequence>
    
    <Sequence>
        <SEQNAME>T1W_Sagittal</SEQNAME>
        <Comment>T1 weighted sagittal brain</Comment>
        <SEQID>SE</SEQID>
        <TR>450</TR>
        <TE>15</TE>
        <FLIP>90</FLIP>
        <FOV>24x24</FOV>
        <MTX>256x192</MTX>
        <THK>5.0</THK>
        <GAP>1.0</GAP>
        <TS>19</TS>
        <NAQ>1</NAQ>
        <TACQT>2:30</TACQT>
        <PLN>SAG</PLN>
        <BW>130</BW>
    </Sequence>
    
    <Sequence>
        <SEQNAME>T2W_Axial_FSE</SEQNAME>
        <Comment>T2 weighted axial with fat suppression</Comment>
        <SEQID>FSE</SEQID>
        <TR>4000</TR>
        <TE>100</TE>
        <FLIP>90/180</FLIP>
        <FOV>22x22</FOV>
        <MTX>320x256</MTX>
        <THK>5.0</THK>
        <GAP>1.0</GAP>
        <TS>25</TS>
        <NAQ>2</NAQ>
        <ETL>16</ETL>
        <TACQT>3:20</TACQT>
        <PLN>AX</PLN>
        <BW>200</BW>
    </Sequence>
    
    <Sequence>
        <SEQNAME>FLAIR_Axial</SEQNAME>
        <Comment>Fluid attenuated inversion recovery</Comment>
        <SEQID>IR-FSE</SEQID>
        <TR>9000</TR>
        <TE>120</TE>
        <TI>2500</TI>
        <FLIP>90/180</FLIP>
        <FOV>22x22</FOV>
        <MTX>256x192</MTX>
        <THK>5.0</THK>
        <GAP>1.0</GAP>
        <TS>25</TS>
        <NAQ>1</NAQ>
        <ETL>20</ETL>
        <TACQT>3:45</TACQT>
        <PLN>AX</PLN>
        <BW>180</BW>
    </Sequence>
    
    <Sequence>
        <SEQNAME>DWI_Axial</SEQNAME>
        <Comment>Diffusion weighted imaging b=1000</Comment>
        <SEQID>EPI-DWI</SEQID>
        <TR>5000</TR>
        <TE>80</TE>
        <FLIP>90</FLIP>
        <FOV>24x24</FOV>
        <MTX>128x128</MTX>
        <THK>5.0</THK>
        <GAP>1.0</GAP>
        <TS>25</TS>
        <NAQ>2</NAQ>
        <TACQT>1:30</TACQT>
        <PLN>AX</PLN>
        <BW>1500</BW>
    </Sequence>
    
    <Sequence>
        <SEQNAME>T1W_Post_Gad_Axial</SEQNAME>
        <Comment>Post contrast T1 weighted</Comment>
        <SEQID>SE</SEQID>
        <TR>500</TR>
        <TE>15</TE>
        <FLIP>90</FLIP>
        <FOV>22x22</FOV>
        <MTX>256x192</MTX>
        <THK>5.0</THK>
        <GAP>1.0</GAP>
        <TS>25</TS>
        <NAQ>1</NAQ>
        <TACQT>3:00</TACQT>
        <PLN>AX</PLN>
        <BW>130</BW>
        <ENABLED>FALSE</ENABLED>
    </Sequence>
</Protocol>"""
    
    with open(file_path, 'w') as f:
        f.write(xml_content)
    print(f"Enhanced sample XML created: {file_path}")

def apply_canon_theme(app):
    """Apply Canon theme to entire application"""
    app.setStyleSheet(CANON_GLOBAL_STYLE)
    
    # Set application palette
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
