
import sys
import os
import cartopy.mpl
import cartopy.mpl.contour
import pandas as pd
from datetime import datetime,timedelta
from typing import Tuple,List

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# from estimators import PF as particlefilter
from estimators import TrackerV2 as ParticleFilterTracker
from sklearn.neighbors import KernelDensity
from scipy.integrate import simpson
import cartopy
import matplotlib.colors as mcolors

COLORS =list(mcolors.TABLEAU_COLORS.values())

cartopy.mpl.contour.GeoContourSet

# patch for Qt 5.15 on macos >= 12
os.environ["USE_MAC_SLIDER_PATCH"] = "1"
from superqt import QRangeSlider


# from sys import platform
# if platform == "linux" or platform == "linux2" or platform == "win32":
#     from cuml.neighbors import KernelDensity as CUDAKernelDensity
#     import cupy as cp
# from qtpy import QtCore
# from qtpy import QtWidgets as QtW

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

import cv2

import numpy as np

from PyQt5.QtCore import Qt

from PyQt5.QtGui import (
    QDoubleValidator,
    QIntValidator,
)
from PyQt5.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QAction, 
    QFileDialog, 
    QMessageBox, 
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QSlider,
    QCheckBox
    
)
import pickle

class Settings():


    KM_PER_DEGREE = 110.0

    def __init__(self,filename = 'settings.pk'):
        
        self.filename = filename
        
        self.nswe = {'w': 31.0, 'e': 37.0, 's': 29.0, 'n': 35.0 }
        self.expand = 0.0
        self.date_range = (datetime(2010,1,1),datetime(2050,1,1))
        self.history_window = timedelta(minutes=60)
        self.dt = timedelta(minutes=30)
        self.contour_resolution_x = 100
        self.contour_resolution_y = 100
        self.contour_resolution_km = 10
        self.min_samples = 3
        self.max_dist = 0.2

        self = self._load()

    def _load(self):
        try:
            with open(self.filename, 'rb') as f:
                loaded_settings =  pickle.load(f)
                for key, value in loaded_settings.__dict__.items():
                    setattr(self, key, value)

                # print('Settings loaded!.')
                # print(self)
        except Exception as e:
            print(e)
            print('Couldn\'t load settings!.')
            return self
    
    def _save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self , f)
            print(f'Settings saved {self.filename}!.')

    def __str__(self):
        
        string = f'nswe :{self.nswe}\n\
        expand 	:{self.expand}\n\
        date_range :{self.date_range}\n\
        history_window :{self.history_window}\n\
        dt :{self.dt}\n\
        contour_resolution_x :{self.contour_resolution_x}\n\
        contour_resolution_y :{self.contour_resolution_y}\n\
        contour_resolution_km :{self.contour_resolution_km}\n\
        min_samples	:{self.min_samples}\n\
        max_dist :{self.max_dist}'	
        
        return string


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.settings = Settings()
        self.setWindowTitle("Lighting Density Estimation and Prediction")
        self.setGeometry(100, 100, 1700, 900)



        # Variables
        # self.nswe = {'w': 32.95, 'e': 37.05, 's': 29.4, 'n': 33.5 }
        # self.date_range = (datetime(2010,1,1),datetime(2050,1,1))
        # self.history_window = timedelta(minutes=60)
        # self.dt = timedelta(minutes=30)
        # self.contour_resolution = 100

        # Central widget with layout
        central_widget = QWidget()
        vbox_layout = QVBoxLayout()
        top_pannel_layout = QVBoxLayout()
        middle_pannel_layout = QHBoxLayout()
        bottom_pannel_layout = QVBoxLayout()

        top_pannel_layout.addWidget(self._create_data_widget())
        top_pannel_layout.addWidget(self._create_tracker_widget())
        top_pannel_layout.addWidget(self._create_nswe_widget())
        top_pannel_layout.addWidget(self._create_checkboxes_widgets())
        # vbox_layout.addLayout(top_pannel_layout,1)

        middle_pannel_layout.addWidget(self._create_figures_widget(),12)
        # middle_pannel_layout.addWidget(NavigationToolbar(self.canvas, self))
        middle_pannel_layout.addLayout(top_pannel_layout,3)
        vbox_layout.addWidget(NavigationToolbar(self.canvas, self))
        vbox_layout.addLayout(middle_pannel_layout,10)
        bottom_pannel_layout.addWidget(self._create_simulation_widget())
        vbox_layout.addLayout(bottom_pannel_layout,1)
        
        central_widget.setLayout(vbox_layout)
        self.setCentralWidget(central_widget)
        self._create_menu_bar()

        self.open_file()


    def _init_estimator(self):
        
        # print(self.nswe.values())
        # xminmax = tuple(list(self.settings.nswe.values())[:2])
        # yminmax = tuple(list(self.settings.nswe.values())[2:])
        expand  = self.settings.expand

        xminmax = (self.settings.nswe['w']-expand, self.settings.nswe['e']+expand)
        yminmax = (self.settings.nswe['s']-expand, self.settings.nswe['n']+expand)
        # print(xminmax,yminmax)
        max_dist = self.settings.max_dist
        min_samples = self.settings.min_samples
        self.pf = ParticleFilterTracker(n_particles=200)
        self.pf.init_tracker(max_dist,min_samples)

        # xlin = np.linspace(xminmax[0]-expand, xminmax[1]+expand, self.settings.contour_resolution)
        # ylin = np.linspace(yminmax[0]-expand, yminmax[1]+expand, self.settings.contour_resolution)
        # print(self.settings.contour_resolution_x,self.settings.contour_resolution_y)
        xlin = np.linspace(xminmax[0], xminmax[1], self.settings.contour_resolution_x)
        ylin = np.linspace(yminmax[0], yminmax[1], self.settings.contour_resolution_y)
        self.X, self.Y = np.meshgrid(xlin, ylin)

    # Plots
    def _create_figures_widget(self):

        self.figure, self.axs = plt.subplots(ncols=2,
                                             figsize=(8, 12),
                                             subplot_kw={'projection': ccrs.PlateCarree()})
        self.canvas = FigureCanvas(self.figure)

        extent = self.settings.nswe.values()
        extent = [e + sign * self.settings.expand for e,sign in zip(extent,[-1,1,-1,1])]
        for ax in self.axs:
            ax.set_extent(extent)
            ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            # ax.set_xlabel('Longitude')
            # ax.set_ylabel('Latitude')
                # Manually add labels
            ax.text(0.5, -0.1, 'Longitude', va='bottom', ha='center',
                    rotation='horizontal', rotation_mode='anchor',
                    transform=ax.transAxes)
            ax.text(-0.1, 0.5, 'Latitude', va='bottom', ha='center',
                    rotation='vertical', rotation_mode='anchor',
                    transform=ax.transAxes)
            
            

        self.measurements_scatter = None
        self.future_measurements_scatter = None
        self.measurements_scatter_prev = None
        self.particles_scatter = None
        self.contour_plot = None
        self.contour_lightning_plot = None
        self.arrows_plot_list = []
        self.tracker_artists_list: List[Tuple]  = []

        self.figure.tight_layout()
        return self.canvas

    def _create_checkboxes_widgets(self):
        layout = QVBoxLayout()
        widget = QGroupBox("Plotting Options")
        widget.setStyleSheet("QGroupBox { border: 2px solid gray; margin-top: 10px; } " \
                            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px;}")

        checkbox = QCheckBox()
        checkbox.setText('Show Contours')
        checkbox.setChecked(False) 
        checkbox.stateChanged.connect(self.__on_checkbox_checked)
        self.checkbox_contour_plot = checkbox
        layout.addWidget(self.checkbox_contour_plot)

                # Thr Step slider

        slider = QSlider(Qt.Horizontal)
        # slider.sliderPressed.connect(self.__on_slider_pressed)
        slider.setMaximum(10);slider.setMinimum(0);slider.setValue(10)
        slider.setInvertedControls(True)
        slider.setInvertedAppearance(True)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)

        slider_lbl = QLabel(str(slider.value()/10.)+" Thr(%)")
        slider_lbl.setAlignment(Qt.AlignCenter)
        self.lbl_thr = slider_lbl
        slider.valueChanged.connect(self.__on_threshold_change)


        self.slider_thr = slider

        slider_layout = QVBoxLayout()
        slider_layout.addWidget(slider)
        slider_layout.addWidget(slider_lbl)

        layout.addLayout(slider_layout)


        checkbox = QCheckBox()
        checkbox.setText('Lightning Contours')
        checkbox.setChecked(False) 
        checkbox.stateChanged.connect(self.__on_checkbox_checked)
        self.checkbox_light_contour_plot = checkbox
        layout.addWidget(self.checkbox_light_contour_plot)

                # Thr Lighting Step slider

        slider = QSlider(Qt.Horizontal)
        # slider.sliderPressed.connect(self.__on_slider_pressed)
        slider.setMaximum(10);slider.setMinimum(0);slider.setValue(10)
        slider.setInvertedControls(True)
        slider.setInvertedAppearance(True)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)

        slider_lbl = QLabel(str(slider.value()/10.)+" Lightning thr(%)")
        slider_lbl.setAlignment(Qt.AlignCenter)
        self.lbl_lightning_thr = slider_lbl
        slider.valueChanged.connect(self.__on_light_threshold_change)

        self.slider_lightning_thr = slider

        slider_layout = QVBoxLayout()
        slider_layout.addWidget(slider)
        slider_layout.addWidget(slider_lbl)

        layout.addLayout(slider_layout)


        checkbox = QCheckBox()
        checkbox.setText('Show Measurements')
        checkbox.setChecked(True) 
        checkbox.stateChanged.connect(self.__on_checkbox_checked)
        self.checkbox_measurements = checkbox
        layout.addWidget(self.checkbox_measurements)

        checkbox = QCheckBox()
        checkbox.setText('Show Prev Measurements')
        checkbox.setChecked(True) 
        checkbox.stateChanged.connect(self.__on_checkbox_checked)
        self.checkbox_prev_measurements = checkbox
        layout.addWidget(self.checkbox_prev_measurements)

        checkbox = QCheckBox()
        checkbox.setText('Show Future Measurements')
        checkbox.setChecked(True) 
        checkbox.stateChanged.connect(self.__on_checkbox_checked)
        self.checkbox_future_measurements = checkbox
        layout.addWidget(self.checkbox_future_measurements)


        checkbox = QCheckBox()
        checkbox.setText('Show Velocities')
        checkbox.setChecked(False) 
        checkbox.stateChanged.connect(self.__on_checkbox_checked)
        self.checkbox_velocities = checkbox
        layout.addWidget(self.checkbox_velocities)

        checkbox = QCheckBox()
        checkbox.setText('Show Particles')
        checkbox.setChecked(False) 
        checkbox.stateChanged.connect(self.__on_checkbox_checked)
        self.checkbox_particles = checkbox
        layout.addWidget(self.checkbox_particles)

        widget.setLayout(layout)
        return widget

    def __on_checkbox_checked(self):
        self._simulate_step()
    
    def __on_threshold_change(self):
        sender = self.sender()
        val = float(sender.value()) / 10.0

        self.lbl_thr.setText(f'{val} Thr(%)')

        self._simulate_step()

    def __on_light_threshold_change(self):
        sender = self.sender()
        val = float(sender.value()) / 10.0
        self.lbl_lightning_thr.setText(f'{val} Lightning thr(%)')
        self._simulate_step()

    # Sliders
    def _create_simulation_widget(self):

        layout = QVBoxLayout()
        widget = QGroupBox("Estimation & Prediction")
        widget.setStyleSheet("QGroupBox { border: 2px solid gray; margin-top: 10px; } " \
                            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px;}")
        widget.setLayout(layout)

        # Dualslider
        slider = QRangeSlider(Qt.Horizontal)
                # Connect the valuesChanged signal to our slot
        slider.valuesChanged.connect(self.__on_date_range_change)
        slider.sliderReleased.connect(self.__on_slider_released)
        # slider.setTickPosition(QSlider.TicksBelow)

        self.slider_date_range = slider

        lbl_layout = QHBoxLayout()
        lbl = QLabel('Dates Range')
        self.lbl_from_date = QLabel(str(slider.value()[0]))
        self.lbl_to_date = QLabel(str(slider.value()[1]))
        lbl_layout.addWidget(lbl)
        lbl_layout.addWidget(self.lbl_from_date)
        lbl_layout.addWidget(self.lbl_to_date)

        slider_layout = QVBoxLayout()
        slider_layout.addLayout(lbl_layout)
        slider_layout.addWidget(slider)

        layout.addLayout(slider_layout)

        # Date slider
        slider = QSlider(Qt.Horizontal)
        slider.setSingleStep(self.settings.dt.seconds)
        slider.sliderPressed.connect(self.__on_slider_pressed)
        slider.valueChanged.connect(self.__on_curr_date_change)
        slider.sliderReleased.connect(self.__on_slider_released)
        # slider.setTickPosition(QSlider.TicksBelow)
        self.slider_current_date = slider

        lbl_layout = QHBoxLayout()
        lbl = QLabel('Current Date')
        self.lbl_curr_date = QLabel(str(slider.value()))
        lbl_layout.addWidget(lbl)
        lbl_layout.addWidget(self.lbl_curr_date)
        
        slider_layout = QVBoxLayout()
        slider_layout.addLayout(lbl_layout)
        slider_layout.addWidget(slider)
        layout.addLayout(slider_layout)

        # Step slider
        slider = QSlider(Qt.Horizontal)
        slider.sliderPressed.connect(self.__on_slider_pressed)
        slider.valueChanged.connect(self.__on_step_change)
        slider.setTickPosition(QSlider.TicksBelow)
        self.slider_step = slider
        self.lbl_step = QLabel(str(slider.value())+" Hours")

        # slider.setMaximum(6);slider.setMinimum(1);slider.setValue(1)
        # slider.setTickPosition(QSlider.TicksBelow)
        # slider.setTickInterval(1)
        lbl_layout = QHBoxLayout()
        lbl = QLabel('Prediction Step (Hours)')
        lbl_layout.addWidget(lbl)
        lbl_layout.addWidget(self.lbl_step)
        
        slider_layout = QVBoxLayout()
        slider_layout.addLayout(lbl_layout)
        dummy_layout = QHBoxLayout()
        dummy_layout.addWidget(QWidget(),4)
        dummy_layout.addWidget(slider,3)
        dummy_layout.addWidget(QWidget(),4)
        # slider_layout.addWidget(slider)
        slider_layout.addLayout(dummy_layout)

        layout.addLayout(slider_layout)

        return widget
    
    def __on_slider_released(self):
        # self._simulate_measurements()
        pass

    def __on_slider_pressed(self):
        slider = self.sender()
        slider.setFocusPolicy(Qt.StrongFocus) # Often default, but good to ensure
        slider.setFocus() # Set initial focus

    def __on_date_range_change(self):
        sender = self.sender()
        low,high = sender.value()

        _from = datetime.fromtimestamp(low)
        _to = datetime.fromtimestamp(high)

        self.lbl_from_date.setText(str(_from))
        self.lbl_to_date.setText(str(_to))

        self.date_range = (_from,_to)
        
        self._update_date_slider(low,high)

    def __on_curr_date_change(self):
        sender = self.sender()
        val = float(sender.value())

        _curr = datetime.fromtimestamp(val)
        self.lbl_curr_date.setText(str(_curr))

        self._simulate_measurements()
        
    def __on_step_change(self):
        sender = self.sender()
        val = float(sender.value())

        self.lbl_step.setText(str(val)+" Hours")
        self._simulate_step()

    # Data Parameters
    def _create_data_widget(self):
        layout = QVBoxLayout()
        widget = QGroupBox("Data Parameters")
        widget.setStyleSheet("QGroupBox { border: 2px solid gray; margin-top: 10px; } " \
                            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px;}")
        widget.setLayout(layout)

        items_list = [
            ('History',' Window(minutes)', self.settings.history_window, QIntValidator()),
            ('Time',' Delta(minutes)', self.settings.dt, QIntValidator())
        ]
        for lbl,lbl_postfix,defulat_value,validator in items_list:
            qline = QLineEdit()
            qline.setObjectName(lbl.lower())
            qline.setText(str(defulat_value))
            qline.setValidator(validator)
            qline.returnPressed.connect(self.__on_data_param_change)

            qlbl = QLabel(lbl+lbl_postfix)
            box = QVBoxLayout()
            box.addWidget(qlbl)
            box.addWidget(qline)
            layout.addLayout(box)
        
        return widget

    def __on_data_param_change(self):
        sender = self.sender()
        name = sender.objectName()
        text = sender.text()
        # print(name,text)

        if name == 'history':
            # self.history_window = int(text) if len(text) > 0 else 0
            self.settings.history_window = timedelta(minutes=int(text))
            sender.setText(str(self.settings.history_window))
        if name == 'time':
            # self.dt = int(text) if len(text) > 0 else 0
            self.settings.dt = timedelta(minutes=int(text))
            sender.setText(str(self.settings.dt))
            

        # TODO Implement reset simulation
        self.settings._save()
        self._update_sliders()
        self._simulate_measurements()

    # Tracker Parameters
    def _create_tracker_widget(self):
        layout = QVBoxLayout()
        widget = QGroupBox("Tracker Parameters")
        widget.setStyleSheet("QGroupBox { border: 2px solid gray; margin-top: 10px; } " \
                            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px;}")
        widget.setLayout(layout)

        items_list = [
            ('Max (∠)','Distance', self.settings.max_dist, QDoubleValidator()),
            ('Min ','Samples', self.settings.min_samples, QIntValidator())
        ]
        for lbl_prefix,lbl,defulat_value,validator in items_list:
            qline = QLineEdit(self)
            qline.setObjectName(lbl.lower())
            qline.setText(str(defulat_value))
            qline.setValidator(validator)
            qline.returnPressed.connect(self.__on_tracker_change)

            qlbl = QLabel(lbl_prefix+lbl)
            box = QVBoxLayout()
            box.addWidget(qlbl)
            box.addWidget(qline)
            layout.addLayout(box)
        
        return widget
    
    def __on_tracker_change(self):
        sender = self.sender()
        number = float(sender.text())

        max_dist = float(self.findChild(QLineEdit,'distance').text())
        min_samples = int(self.findChild(QLineEdit,'samples').text())

        self.settings.max_dist = max_dist
        self.settings.min_samples = min_samples

        self.pf.init_tracker(max_dist,min_samples)

        # print(sender)
        # print(self.findChild(QLineEdit,'distance'))
        # print(self.findChild(QLineEdit,'samples'))

        # TODO Implement reset simulation
        self.settings._save()
        self._simulate_measurements()

    # NSWE Widget
    def _create_nswe_widget(self):
        # Main widget
        layout = QGridLayout()
        widget = QGroupBox("NSWE Limits")
        widget.setStyleSheet("QGroupBox { border: 2px solid gray; margin-top: 10px; } " \
                            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px;}")
        widget.setLayout(layout)

        # items_list = [('Grid size(km^2)',0,1,self.settings.contour_resolution_km)
        #               ('North',0,1,self.settings.nswe['n']),
        #               ('South',2,1,self.settings.nswe['s']),
        #               ('West',1,0, self.settings.nswe['w']),
        #               ('East',1,2, self.settings.nswe['e']),
        #               ('Expand',1,1,self.settings.expand)]

        items_list = [('Grid size(km)',0,1,self.settings.contour_resolution_km),
                ('North',1,1,self.settings.nswe['n']),
                ('South',3,1,self.settings.nswe['s']),
                ('West',2,0, self.settings.nswe['w']),
                ('East',2,2, self.settings.nswe['e']),
                ('Expand',2,1,self.settings.expand)]
        
        for lbl,x,y,defulat_value in items_list:
            qline = QLineEdit()
            qline.setObjectName(lbl.lower())
            qline.setText(str(float(defulat_value)))
            qline.setValidator(QDoubleValidator())
            qline.returnPressed.connect(self.__on_nswe_change)

            qlbl = QLabel(lbl)
            box = QHBoxLayout()
            box.addWidget(qlbl)
            box.addWidget(qline)
            layout.addLayout(box, x, y)

        return widget

    def __on_nswe_change(self):

        sender = self.sender()
        number = float(sender.text())
        expand = 0
        # print(sender.objectName())
        mapping_dict = {'north':'n','south':'s','west':'w','east':'e'}
        # key = sender.objectName()
        try:
            self.settings.nswe[mapping_dict[sender.objectName()]] = number
            # self.settings.nswe[key] = number
        except:
            if sender.objectName() == 'expand':
                expand = number
                self.settings.expand = expand
            else:
                self.settings.contour_resolution_km = number
                
        x_range = self.settings.nswe['e'] - self.settings.nswe['w'] + 2 * self.settings.expand
        self.settings.contour_resolution_x = int(x_range * Settings.KM_PER_DEGREE // self.settings.contour_resolution_km)
        y_range = self.settings.nswe['n'] - self.settings.nswe['s'] + 2 * self.settings.expand
        self.settings.contour_resolution_y = int(y_range * Settings.KM_PER_DEGREE // self.settings.contour_resolution_km)
        


        self._init_estimator()
        extent = self.settings.nswe.values()
        extent = [e + sign * expand for e,sign in zip(extent,[-1,1,-1,1])]
        for ax in self.axs:
            ax.set_extent(extent)
        self.canvas.draw()


        text = str(number)
        sender.setText(text)    

        self.settings._save()    
        # self._simulate_step()

    # Menu Bar
    def _create_menu_bar(self):
        menubar = self.menuBar()
        # File Menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Load Lightning Data...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        # file_menu.addSeparator()

        # exit_action = QAction("Exit", self)
        # exit_action.triggered.connect(self.close)
        # file_menu.addAction(exit_action)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File")
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    self.frame = pd.read_excel(file_path,index_col=False,sheet_name='Sheet1')
                    self.frame['UTC'] = pd.to_datetime(self.frame['UTC'],format='ISO8601')
                    self.df_sample = self.frame
                elif file_path.endswith('.csv'):
                    self.frame = pd.read_csv(file_path,index_col=False)
                    self.frame['UTC'] = pd.to_datetime(self.frame['UTC'],format='ISO8601')
                    self.df_sample = self.frame
                self._prepare_dataframe_and_widgets()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not open file: {e}")
                print(e)

    def _prepare_dataframe_and_widgets(self):
        frame = self.frame
        # mask = (frame['lat'] <= self.nswe['n']) & (frame['lat'] >= self.nswe['s']) & \
        # (frame['lon'] <= self.nswe['e']) & (frame['lon'] >= self.nswe['w'])
        self.frame = frame.sort_values('UTC',ignore_index=True)
        
        self._init_estimator()
        
        self._update_sliders()
        
    
    def _update_sliders(self):
        if not hasattr(self,'frame'): return
        dftime = self.frame['UTC']
        # print(dftime.iloc[0],dftime.iloc[-1])
        # print(type(dftime.iloc[0]),type(self.settings.dt),type(self.settings.history_window))
        
        print('-----------------------')
        print(type(dftime.iloc[0]),type(self.settings.dt),type(self.settings.history_window))
        _from = dftime.iloc[0] + self.settings.dt + self.settings.history_window
        _to   = dftime.iloc[-1] - timedelta(hours=6)
        print(_from,_to)
        _s = _from.to_pydatetime().timestamp()
        _e =   _to.to_pydatetime().timestamp()

        self.slider_date_range.setMinimum(_s)
        self.slider_date_range.setMaximum(_e)
        self.slider_date_range.setValue((_s,_e))

        self.slider_step.setMaximum(6)
        self.slider_step.setMinimum(0)
        self.slider_step.setValue(1)

        self.slider_current_date.setSingleStep(self.settings.dt.seconds)
        
        self._simulate_measurements()

    def _update_date_slider(self,low,high):
        _s,_e = low,high
        self.slider_current_date.setMinimum(int(_s))
        self.slider_current_date.setMaximum(int(_e))
        # self.slider_current_date.setValue(int(_s))
        ########################################################################### SET BACK TO NORMAL ###########################################3
<<<<<<< HEAD
        # self.slider_current_date.setValue(int(1706538868))
=======
        self.slider_current_date.setValue(int(1706538868))
        self.slider_current_date.setValue(int(1706686012))
>>>>>>> revision

    def _get_current_date(self):
        timestamp = self.slider_current_date.value()
        print(timestamp)
        timestamp = float(timestamp)
        date = datetime.fromtimestamp(timestamp)
        return date
    
    def create_mask_from_lon_lat(self,lon_lat_points, X, Y):
        """
        Creates a binary mask (same shape as X, Y) with 1s at locations
        corresponding to the given lon/lat points, and 0s elsewhere.

        Args:
            lon_lat_points (np.ndarray): A 2D array of shape (N, 2) where N is the
                                        number of points, and each row is [longitude, latitude].
            X (np.ndarray): 2D array of longitudes from meshgrid.
            Y (np.ndarray): 2D array of latitudes from meshgrid.

        Returns:
            np.ndarray: A 2D binary mask (0s and 1s) with the same shape as X and Y.
        """
        Z_new = np.zeros_like(X, dtype=int)

        for target_lon, target_lat in lon_lat_points:
            # Calculate the squared difference for efficiency
            diff_lon = X - target_lon
            diff_lat = Y - target_lat
            distances_sq = diff_lon**2 + diff_lat**2

            # Find the index of the minimum distance
            min_idx_flat = np.argmin(distances_sq)
            row_idx, col_idx = np.unravel_index(min_idx_flat, distances_sq.shape)

            # Set the corresponding cell in the new mask to 1
            Z_new[row_idx, col_idx] = 1
        # Z_new = np.expand_dims(Z_new,0)
        return Z_new

    def _get_measurements(self):

        
        date = self._get_current_date()
        step = self.slider_step.value()
        dt_gap = self.settings.dt
        history_gap = self.settings.history_window

        # self.nswe = {'w': 32.95, 'e': 37.05, 's': 29.4, 'n': 33.5 }
        xminmax = tuple(list(self.settings.nswe.values())[:2])
        yminmax = tuple(list(self.settings.nswe.values())[2:])
        expand  = self.settings.expand

        region_mask =  ((self.frame['lon'] >= xminmax[0]-expand) & (self.frame['lon'] <= xminmax[1]+expand))
        region_mask &= ((self.frame['lat'] >= yminmax[0]-expand) & (self.frame['lat'] <= yminmax[1]+expand))
        df_sample = self.frame[region_mask]
        df_sample = self.frame



        time_mask = (df_sample.UTC >= date - history_gap) \
            & (df_sample.UTC <= date)
        time_mask_prev = (df_sample.UTC >= date - dt_gap - history_gap) \
            & (df_sample.UTC <= date - dt_gap)
        truth_time_mask = (df_sample.UTC > date + timedelta(hours=step-1)) \
            & (df_sample.UTC <= date + timedelta(hours=step))

        measurements = df_sample[time_mask][['lon', 'lat']].to_numpy()
        measurements_prev = df_sample[time_mask_prev][['lon', 'lat']].to_numpy()
        future_measurements = df_sample[truth_time_mask][['lon', 'lat']].to_numpy()

        return measurements,measurements_prev,future_measurements

    def _simulate_measurements(self):
        if not hasattr(self,'frame'): return
        measurements,measurements_prev,future_measurements = self._get_measurements()
        # step = self.slider_step.value()
        # print(len(measurements),len(measurements_prev))
        if len(measurements) > 0 and len(measurements_prev) > 0:
            # self._init_estimator()
            dt = self.settings.dt
            self.pf.init_tracker(self.settings.max_dist,self.settings.min_samples)
            self.pf.update_all(measurements,measurements_prev,dt)

            self._simulate_step()

    def _simulate_step(self):
        if not hasattr(self,'frame'): return
        measurements,measurements_prev,future_measurements = self._get_measurements()
        step = self.slider_step.value()
        dt = self.settings.dt
        # print(step,dt)
        # for _ in range(step):
        self.pf.predict(n_steps=step,dt=dt)
        self._plot_simulation(measurements,future_measurements,measurements_prev)

    def _plot_simulation(self,measurements,
                         future_measurements,
                         measurements_prev = None):
        ax = self.axs[0]
        particles = self.pf.get_all_particles()
        # print(particles.shape)
        if self.checkbox_particles.checkState():
            if self.particles_scatter is None:
                self.particles_scatter = ax.scatter(*particles[:,:2].T,s=3,c='green',label = 'Particles',alpha=.1)
            else:
                self.particles_scatter.set_offsets(particles[:,:2])
        else:
            try:
                self.particles_scatter.remove()
                self.particles_scatter = None
            except:pass

                                             # KDE Plot
        if not self.contour_plot is None:
            self.contour_plot.get_paths().clear()
        try:        
            self.cbar.remove()
        except Exception as ee:pass

        if self.checkbox_contour_plot.checkState():

            grid_coords = np.vstack([self.X.ravel(), self.Y.ravel()]).T
            val = float(self.slider_thr.value()) / 10.
            # print(val)
            Z = self.pf.get_gaussian_estimation(grid_coords,
                                                self.settings.contour_resolution_x, 
                                                self.settings.contour_resolution_y,
                                                thr = val)
            



            area_template = np.zeros_like(Z)
            area_template[1:-1,1:-1] = 1
            cnts,_  = cv2.findContours(area_template.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cnt = cnts[0]
            cnt = np.concat((cnt,np.expand_dims(cnt[0],axis=0)),axis=0)
            total_area = cv2.contourArea(cnt)
            contours, hierarchy = cv2.findContours(Z.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            ta = 0
            for cnt in contours:
                # print(cv2.contourArea(cnt))
                ta += cv2.contourArea(cnt)
            # print(ta / 330**2,ta,total_area,'----------------------------------------')
            
            # if not self.contour_plot is None:
            #     self.contour_plot.get_paths().clear()             
            Z = np.ma.masked_array(Z, Z < 0.01)
            self.contour_plot = ax.contourf(self.X, self.Y, Z, levels=25, cmap='plasma',alpha=.33)
            
            # self.cbar = self.figure.colorbar(self.contour_plot,fraction=0.046, pad=0.1)
            # lonlatmask = self.create_mask_from_lon_lat(future_measurements,self.X,self.Y)
            # print(Z)
            # self.contour_plot = ax.contourf(self.X, self.Y, Z , levels=1, cmap='plasma',alpha=.33)
            # ax.imshow(Z,alpha=1)
            # self.contour_plot = ax.imshow(Z,alpha=.5)

             # Scatter Plots

        if not self.contour_lightning_plot is None:
            self.contour_lightning_plot.get_paths().clear()
        try:        
            self.cbar_lightning.remove()
        except Exception as ee:pass

        if self.checkbox_light_contour_plot.checkState() and len(future_measurements) > 0:

            pf = ParticleFilterTracker()
            pf.init_tracker(self.settings.max_dist,self.settings.min_samples)
            pf.update_all(future_measurements,future_measurements,self.settings.dt)
            # Z=pf.get_gaussian_estimation(grid_coords,
            #                                     self.settings.contour_resolution_x, 
            #                                     self.settings.contour_resolution_y,
            #                                     thr = val)
            val = float(self.slider_lightning_thr.value()) / 10.
            grid_coords = np.vstack([self.X.ravel(), self.Y.ravel()]).T
            Z_lightning = pf.get_gaussian_estimation(grid_coords,
                                                           self.settings.contour_resolution_x, 
                                                           self.settings.contour_resolution_y,
                                                           thr=val,
                                                           )
            
            Z_lightning = np.ma.masked_array(Z_lightning, Z_lightning < 0.01)
            # print(Z_lightning)
            self.contour_lightning_plot = ax.contourf(self.X, self.Y, Z_lightning, levels=25, cmap='winter',alpha=.33)
            # self.cbar_lightning = self.figure.colorbar(self.contour_lightning_plot,fraction=0.046, pad=0.1)
    
        if self.checkbox_measurements.checkState():
            if self.measurements_scatter is None:
                self.measurements_scatter = ax.scatter(*measurements[:, :2].T, c='green', marker='x', alpha=1,label='Current Measurements')
            else:
                self.measurements_scatter.set_offsets(measurements[:, :2])
        else:
            try:
                self.measurements_scatter.remove()
                self.measurements_scatter = None
            except:pass


        if self.checkbox_future_measurements.checkState():
            if self.future_measurements_scatter is None:
                self.future_measurements_scatter = ax.scatter(*future_measurements[:, :2].T, c='red', marker='o', alpha = 1,label='Future Measurements')
            else:
                self.future_measurements_scatter.set_offsets(future_measurements[:, :2])
        else:
            try:
                self.future_measurements_scatter.remove()
                self.future_measurements_scatter = None
            except:pass
        if self.checkbox_prev_measurements.checkState():
            if not measurements_prev is None:
                # print(measurements_prev)
                if self.measurements_scatter_prev is None:
                    self.measurements_scatter_prev = ax.scatter(*measurements_prev[:, :2].T, c='blue', marker='x',alpha = 1, label='Past Measurements')
                else:
                    self.measurements_scatter_prev.set_offsets(measurements_prev[:, :2])
        else:
            try:
                self.measurements_scatter_prev.remove()
                self.measurements_scatter_prev = None
            except:pass

        ax.legend(loc=3)

        ax = self.axs[1]
        # if self.checkbox_particles.checkState():
        #     if self.particles_scatter is None:
        #         self.particles_scatter = ax.scatter(*self.pf.particles[:,:2].T,s=3,c='green',label = 'Particles')
        #     else:
        #         self.particles_scatter.set_offsets(self.pf.particles[:,:2])
        # else:
        #     try:
        #         self.particles_scatter.remove()
        #         self.particles_scatter = None
        #     except:pass

        for arrow in self.arrows_plot_list:
            arrow.remove()
        self.arrows_plot_list.clear()

        if self.checkbox_velocities.checkState():
            for p in particles:
                x,y,dx,dy = p
                scale = 2
                a = ax.arrow(x,y,dx * scale,dy * scale,width=.005)
                self.arrows_plot_list.append(a)


        for t in self.tracker_artists_list:
            for a in t: a.remove()
        self.tracker_artists_list.clear()

        for id,pf in self.pf.particle_filters.items():
            color = COLORS[id % len(COLORS)] 
            values = pf.particles[:,:2]
            s = ax.scatter(*values.T,marker='o',label=f'Cluster {id}',c=color,s=.1)
            t = ax.annotate(str(id),np.mean(values,axis=0),weight='bold')
            
            x,y = np.mean(values,axis=0)
            # dx,dy = self.pf.velocities[id]
            dx,dy = np.mean(pf.particles[:,2:],axis=0) * 5
            norm = np.linalg.norm([dx,dy])
            # if dx <= 0 and dy <= 0: continue
            a = ax.arrow(x,y,dx,dy,width=norm)
            # print('velocity:',id,dx,dy)

            self.tracker_artists_list.append((s,t,a))

        
        if len(self.pf.particle_filters) > 0:
            ax.legend(loc=3)
        self.canvas.draw()
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QCheckBox, QGroupBox, QLineEdit, QLabel { font-size: 12pt; font-weight: bold; }")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

