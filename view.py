import sys
import numpy as np
import pylab
import time

from PyQt5 import QtGui,QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import pyqtgraph
from pyqtgraph import PlotWidget, PlotItem, ImageView

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from model import *
import random



class MainApp(QtGui.QMainWindow):
    """ QMainwindow that contains a Videoview on top and a DataListView on bottom"""

    def __init__(self, data_models, video, parent=None):
        """ Initializes the Main window
            Args:
                data_models: List of DataModels
                video_models: List of VideoModels
        """
        super(MainApp, self).__init__(parent)
        self.video = video

        #Draw User Interface
        pyqtgraph.setConfigOption('background', 'w') #White background
        self.resize(1280, 640)
        self.setup_ui(data_models, video)


    def setup_ui(self, data_models, video_model):
        """ Sets up outermost level of GUI elements and layouts
            Args:
                datamodels: List of DataModels
                video: VideoModel
        """
        # 1. Create Widgets
        self.upperwidget = QtGui.QWidget(self)
        self.lowerwidget = DataListView(data_models, video_model)

        #2. Create a couple of elements
        self.video_view = VideoView(self.video)#VideoPlot(self.video)
        self.video_view.update()

        self.sld = QSlider(Qt.Horizontal, self)
        self.sld.setRange(0, self.video.get_amount_of_frames()-1)
        self.sld.setTickPosition(QSlider.TicksAbove)

        #3.Create Layout
        self.verticalLayout = QtGui.QVBoxLayout(self.upperwidget)

        #4. Add elements to layout
        self.verticalLayout.addWidget(self.video_view)#ADD VIDEO HERE
        self.verticalLayout.addWidget(self.sld)

        #5. Connect navigation elements to respective slots
        self.sld.valueChanged.connect(self.video.set_framenumber)
        self.video.framenumber.connect(self.sld.setValue)

        #6.Add upper and lowerwidget to QSplitter
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.upperwidget)
        self.splitter.addWidget(self.lowerwidget)

        self.setCentralWidget(self.splitter)
        self.setWindowTitle(QtGui.QApplication.translate("EEG Viewer", "EEG Viewer", None))

        #7. menuBar
        self.add_menubar()

    def add_menubar(self):
        """ Adds a menubar to the application"""
        extractAction = QtGui.QAction("&Exit", self)
        #extractAction.setShortcut("Ctrl+Q")
        #extractAction.setStatusTip('Leave The App')
        extractAction.triggered.connect(self.close_application)

        self.mainMenu = self.menuBar()
        self.fileMenu = self.mainMenu.addMenu('&File')
        self.fileMenu.addAction(extractAction)

        databaseManager = QtGui.QAction("&Start Database Manager", self)
        self.databaseMenu = self.mainMenu.addMenu('&Database')
        self.databaseMenu.addAction(databaseManager)

        self.motionSelectorMenu = self.mainMenu.addMenu('&Motion Selector')
        motionSelector = QtGui.QAction("&Select Motion ROI", self)
        self.choose_ROI = self.motionSelectorMenu.addAction(motionSelector)
        motionSelector.triggered.connect(self.video_view.choose_ROI)

    def close_application(self):
        """ Closses and exits application"""
        self.close()
        QtGui.QApplication.exit()

class DataListView(QScrollArea):
    """ Scroll area that contains several DataViews"""
    def __init__(self, datamodels = [], videomodel = None, parent = None):
        """ Initializes the DataListView
            Args:
                datamodels: List of DataModels
                videomodel: VideoModel that is used in the DataModels to locate the indicator
                parent: Parent QtWidget
        """
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

        self.videomodel = videomodel
        self.datamodels = []#Use self.add_data_display() tQtGui.QListWidget(self.centralwidget)o append
        self.dataviews = []
        pyqtgraph.setConfigOption('background', 'w') #White background
        self.setup_ui(datamodels)

    def wheelEvent(self, ev):
        """ Handles the wheel event and turns mouse wheel control off for the Scroll area"""
        if ev.type() == QtCore.QEvent.Wheel:
            ev.ignore()

    def setup_ui(self, datamodels):
        """ Sets up the user interface
        Args:
            datamodels: List of DataModels
        """
        self.datamodels = []#Use self.add_data_display() to append
        self.dataviews = []

        self.centralwidget = QtGui.QWidget(self)
        #self.centralwidget.setMinimumSize(600, 400)

        #self.scrollarea = QScrollArea(self)
        self.setWidgetResizable(True)

        #Create layout
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)

        #Create couple of elements
        self.addDataDisplays = QtGui.QPushButton(self) #Draw (+) button to add data displays
        self.addDataDisplays.setText("+")

        #self.verticalLayout.addWidget(self.addDataDisplays)

        for model in datamodels:
            self.add_data_display(model)
            #self.datamodels.append(model)

        self.addDataDisplays.clicked.connect(self.add_data_display)

        #add elements to layout
        self.setGeometry(self.geometry())
        self.setWidget(self.centralwidget)

    def delete(self, datadisplay):
        """ Deletes a model from the widget (by reinitializing the centralwidget) and the list of models  """
        idx = self.dataviews.index(datadisplay)

        l = len(self.datamodels.copy())#For assertion

        readd_models = self.datamodels.copy()
        del readd_models[idx]

        assert len(readd_models) == l -1

        self.datamodels = []#Use self.add_data_display() to append
        self.dataviews = []

        self.centralwidget = QtGui.QWidget(self)
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)

        for model in readd_models:
            assert isinstance(model, DataModel)
            self.add_data_display(model)
            #self.datamodels.append(model)

        self.setWidget(self.centralwidget)

        #redraw Add button at bottom
        self.add_button()

    def add_data_display(self, model = None):
        """ Adds datadisplay
            Args:
                model: Model to be added to centralwidget as well as to list of models
        """
        #Create new MODEL and append it to self.models
        if type(model) != type(DataModel()):#Via button (+) press some message (False) arrives
            model = DataModel()

        self.datamodels.append(model)

        #Create VIEW and add it (widget) to layout
        optiondisplay = DataView(model, video_model = self.videomodel, container = self)#such that child can tell parent it's dead
        self.dataviews.append(optiondisplay)
        self.verticalLayout.addWidget(optiondisplay)

        #Subscribe
        self.videomodel.eeg_pos.connect(optiondisplay.data_plot.update)
        self.videomodel.framenumber.connect(optiondisplay.mothist_plot.update)
        model.channeldata.connect(optiondisplay.data_plot.print_data)
        model.mothistmap.connect(optiondisplay.mothist_plot.print_data)

        self.add_button()

    def add_button(self):
        """ Adds button at end of list that may be used to add additional DataDisplays """
        if len(self.datamodels) == 0:#In this case the button is deleted completely as there is no reference to it
            self.addDataDisplays = QtGui.QPushButton(self) #Draw (+) button to add data displays
            self.addDataDisplays.setText("+")
            self.addDataDisplays.clicked.connect(self.add_data_display)
        self.verticalLayout.removeWidget(self.addDataDisplays)
        self.verticalLayout.addWidget(self.addDataDisplays)


class VideoView(QWidget):
    """ Displays a video and a control panel for it"""
    def __init__(self, video_model = None, parent=None):
        """ Initializes the View and parse_filepath_attributes
            Args:
                video_model: VideoModel that wraps access to video frames
                parent: Parent QtWidget
        """
        #1. Create the widget i.e. self by calling superclass
        super(QtGui.QWidget,self).__init__(parent)
        self.video_model = video_model

        #2. Create a couple of elements
        self.video_plot = VideoPlot(self.video_model, self)
        self.options_group = self.createOptionsGroup()

        # 2a: Create Window that will open upon button press
        self.dialog = MotionWindowSelector(video=video_model, parent = self)

        #3. Create and set layout
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSpacing(20)
        self.setLayout(self.horizontalLayout)####IMPORTANT
        self.setMinimumSize(120, 120)

        #4. Add elements to widget
        self.horizontalLayout.addWidget(self.video_plot)
        self.horizontalLayout.addWidget(self.options_group)

        #Make additional connections
        video_model.frame.connect(self.video_plot.update)
        video_model.title_signal.connect(self.set_title)

    def change_video(self, n_vid):
        """ Changes the underlying video file the videomodel wraps"""
        self.video_model.change_video(self.video_model.dyad, n_vid)

    def set_number(self, n = 0):
        """ deprecated ...... """
        assert False
        self.video_plot.update()

    def set_title(self, title):
        """ Sets the title being displayed in the Optionsbox"""
        self.l1.setText(title)

    def createOptionsGroup(self):
        """ Creates the options Group that gives access to control framenumber, play etc. """
        #1. Create a widget (here: QGroupBox)
        self.groupBox = QGroupBox()
        self.groupBox.setAlignment(4)

        #2. Create a couple of elements
        self.load_button = QtGui.QPushButton()
        self.load_button.setText("Load...")
        self.close_button = QtGui.QPushButton()
        self.open_dialog = QtGui.QPushButton()
        self.open_dialog.setText("Specify Motion ROI")
        self.frame_forward = QtGui.QPushButton()
        self.frame_back = QtGui.QPushButton()
        self.play = QtGui.QPushButton()
        self.reverse_play = QtGui.QPushButton()
        self.stop_vid = QtGui.QPushButton()

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./icons/frame_back.png"))
        self.frame_back.setIcon(icon)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./icons/play.png"))
        self.play.setIcon(icon)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./icons/stop.png"))
        self.stop_vid.setIcon(icon)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./icons/frame_forward.png"))
        self.frame_forward.setIcon(icon)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./icons/reverse_play.png"))
        self.reverse_play.setIcon(icon)

        self.close_button.setText("Close")
        self.l1 = QLabel("Video")
        self.spin_box = QSpinBox()

        #3. Add them to a QVBoxLayout (Vertical)
        vbox = QVBoxLayout()
        vbox.addWidget(self.l1)
        vbox.addWidget(self.spin_box)
        vbox.addWidget(self.load_button)
        vbox.addWidget(self.close_button)
        vbox.addWidget(self.open_dialog)

        vbox.addWidget(self.stop_vid)#
        vbox.addWidget(self.play)#
        vbox.addWidget(self.reverse_play)#
        vbox.addWidget(self.frame_forward)#
        vbox.addWidget(self.frame_back)#
        vbox.addStretch(1)#Add empty QSpacerItem that pushes the buttons upwards

        #4. Add layout to widget
        self.groupBox.setLayout(vbox)

        #5. connect
        self.open_dialog.clicked.connect(self.choose_ROI)
        self.frame_forward.clicked.connect(self.video_model.frame_forward)
        self.frame_back.clicked.connect(self.video_model.frame_back)
        self.play.clicked.connect(self.video_model.start_play)
        self.stop_vid.clicked.connect(self.video_model.stop_play)
        self.spin_box.valueChanged.connect(self.change_video)

        return self.groupBox

    def choose_ROI(self):
        """ Opens the dialog that allows the selection of a Motion Region of interest"""
        self.dialog.show()

class DataView(QWidget):
    """ A widget that displays a DataView and a Group Box that gives access to loading new data"""
    def __init__(self, model, video_model = None, container = None, parent=None):#Parent is container
        """ Initializes the DataListView
            Args:
                model: DataModel that represents 1d or 2d data
                video_model: VideoModel that represents a video File
                container: QWidget that contains self. Delete is called at container if self is obsoleteself
                parent: Parent QWidget
        """
        super(QtGui.QWidget,self).__init__(parent)
        self.model = model
        self.video_model = video_model
        self.container = container#datalistview that contains self

        self.data_plot = InteractiveDataPlot(self.model, self.video_model, self)
        self.mothist_plot = InteractiveImagePlot(self.model, self.video_model, self)
        self.optionsgroup = self.createOptionsGroup()

        self.setMinimumSize(150, 150)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSpacing(20)
        self.setLayout(self.horizontalLayout)####IMPORTANT

        self.draw_elements()
        self.toggle_view(self.model.datatype)

        self.model.datatype_signal.connect(self.set_datatype)

    def draw_elements(self):
        self.stack = QStackedWidget(self)
        self.stack.addWidget(self.data_plot)
        self.stack.addWidget(self.mothist_plot)

        self.horizontalLayout.addWidget(self.stack)
        self.horizontalLayout.addWidget(self.optionsgroup)

        self.stack.setCurrentIndex(0)

    def toggle_view(self, datatype):
        if datatype == "eeg":
            self.stack.setCurrentIndex(0)
        elif datatype == "motion":
            self.stack.setCurrentIndex(1)

    def set_title(self, title):
        """ Sets the title of the group box
            Args:
                channel: Datachannel that defines current 1D data.
        """
        self.groupBox.setTitle(title)

    def set_channel(self, channel):
        """ Sets current Data channel
            Args:
                channel: Datachannel that defines current 1D data.
        """
        self.l1.setText("Channel: " + str(channel))

    def createOptionsGroup(self):
        """ Creates a QGroupBox that contains navigation elements for the video """
        self.groupBox = QGroupBox(self.model.get_title())
        self.groupBox.setAlignment(4)

        self.load_button = QtGui.QPushButton()
        self.close_button = QtGui.QPushButton()

        self.l1 = QLabel("Channel: " + str(self.model.get_channel()))
        self.spin_box = QSpinBox()
        self.spin_box.setMinimumHeight(22)

        vbox = QVBoxLayout()
        vbox.addWidget(self.l1)
        vbox.addWidget(self.spin_box)
        vbox.addWidget(self.load_button)
        vbox.addWidget(self.close_button)

        self.load_button.setText("Load...")
        self.close_button.setText("Close")

        #USE EEG DISPLAY CONTROLLER TO HAVE THE Models LOAD ITS DATA
        loader = DataController(self.model)
        self.load_button.clicked.connect(loader)
        loader.title.connect(self.set_title)

        #LET THE MODEL COMMUNICATE IT'S DEAD
        self.close_button.clicked.connect(self.delete)

        #Use spin box to switch through channels
        self.spin_box.valueChanged.connect(self.model.set_channel)
        self.spin_box.valueChanged.connect(self.set_channel)

        vbox.addStretch(1)
        self.groupBox.setLayout(vbox)

        return self.groupBox

    def set_datatype(self, datatype, update = True):
        self.toggle_view(datatype)

    def delete(self):
        if self.container:
            self.container.delete(self)

class VideoPlot(QLabel):
    """ Displays a video-frame as a QLabel
        Drawing is relaized such that the aspect-ratio is kept constant
        and the image fills up all the available space in the layout the VideoPlot is contained in"""
    def __init__(self, video, parent=None, centered = True):
        super(VideoPlot, self).__init__(parent)

        self.video = video

        #set background to black and border to 0
        self.setStyleSheet("background-color: rgb(0,0,0); margin:0px; border:0px solid rgb(0, 255, 0); ")

        self.setMinimumSize(320, 180)#Set minimum size
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)# Set size policy to expanding
        self.setAlignment(Qt.AlignCenter)
        self.update()

    def resizeEvent(self, event):
        """ Rescales the Pixmap that contains the image when QLabel changes size
            Args:
                event: QEvent
        """
        size = self.size()
        size = QSize(int(size.width()),int(size.height()))

        scaledPix = self.pixmap.scaled(size, Qt.KeepAspectRatio, transformMode = Qt.FastTransformation )
        self.setPixmap(scaledPix)

    def update(self, frame = None):
        """ Upates the pixmap when a new frame is to be displays. Triggers the Qt eventpipeline.
        """
        if type(frame) == type(None):
            frame = self.video.get_frame()
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap = QtGui.QPixmap(image)
        size = self.size()
        scaledPix = self.pixmap.scaled(size, Qt.KeepAspectRatio, transformMode = Qt.FastTransformation)
        self.setPixmap(scaledPix)

        QtCore.QCoreApplication.processEvents()


class DataPlot(FigureCanvas):
    """ Allows to draw a Dataplot using matplotlib """
    def __init__(self, model, parent=None, width=5, height=4, dpi=100):
        """ Initializes view
            Args:
                model: DataModel that contains 1d data
                parent: Parent QtWidget
                width: Width of matplotlib.pyplot in inches. Resulting plot (image) might be scaled to match SizePolicy needs.
                height: Height of matplotlib.pyplot in inches. Resulting plot (image) might be scaled to match SizePolicy needs.
                dpi: Dots per inch of Height of matplotlib.pyplot in inches. Resulting plot (image) might be scaled to match SizePolicy needs.

        """
        self.model = model
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()

    def plot(self):
        data = self.model.get_data()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(data, 'r-')
        #ax.set_title(self.model.get_title())
        self.draw()


class InteractiveImagePlot(ImageView):
    """ Allows to draw 2d numpy array in an interactive plot using a colormap"""
    def __init__(self, model, video_model = None, parent=None, colormap = "inferno"):
        """ Sets the datamodel and optionally a videomodel that is used to print a time indicator according to its current frameself.
        Args:
            model:  DataModel wrapping either 1d or 2d data
            video_model: VideoModel wrapping a video file
            parent: Parent QtWidget
            colormap: String describing matplotlib colormap
        """
        self.view= PlotItem()
        super().__init__(view = self.view)

        self.model = model
        self.video_model = video_model
        self.print_indicator = False
        self.indicator = None

        # Get a colormap
        colormap = plt.cm.get_cmap(colormap)  # cm.get_cmap("CMRmap")nipy_spectral
        colormap._init()
        self.lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()

        initial_data = self.model.get_data()
        if not isinstance(initial_data, type(None)):#Draw imagedata if possible
            try:
                self.print_data(initial_data)
            except:#It might be that the model contains data that is not displayable e.g. 1d
                pass

    def print_data(self, data):
        """ Prints data to dataplot
            Args:
                data: 2d float.64 numpy arrays containing values between 0 and 1
        """
        self.print_indicator = True
        self.imagedata = data
        self.setImage(self.imagedata)

        self.indicator_min = -200
        self.indicator_max = 200

        if self.video_model != None:
            pos = int(self.video_model.get_pos(datatype = "motion"))
            self.indicator = self.view.plot([pos,pos],[self.indicator_min,self.indicator_max],pen=pyqtgraph.mkPen(color=pyqtgraph.hsvColor(2),width=1))

    def update(self, pos):
        """ Updates indicator position
            Args:
                pos: Int describing the current position of the indicator
        """
        if self.print_indicator and self.indicator and not self.video_model == None:
            C=pyqtgraph.hsvColor(1)
            pen=pyqtgraph.mkPen(color=C,width=1)
            pos = int(self.video_model.get_pos(datatype = self.model.get_datatype()))
            self.indicator.setData([pos,pos],[self.indicator_min,self.indicator_max])

    def updateImage(self, autoHistogramRange=True):
        """ Updates the image, setting the colormap"""
        super().updateImage(autoHistogramRange=autoHistogramRange)
        self.getImageItem().setLookupTable(self.lut)

    def setImage(self, *args, **kwargs):
        """ Sets the image and adjusts the initial view/zoom in the plot"""
        super().setImage(*args, **kwargs)
        self.view.disableAutoRange(axis = 0)
        self.view.enableAutoRange(axis = 1)

class InteractiveDataPlot(PlotWidget):
    """ Allows to draw a 1d interactive Dataplot """
    def __init__(self, model, video_model = None, parent=None):
        super().__init__(parent)
        self.main_plot = None
        self.indicator = None
        self.video_model = video_model
        self.model = model

        self.print_indicator = False#As long as there is no data we don't want to print the indicator
        initial_data = self.model.get_data()
        if not isinstance(initial_data, type(None)):
            try:
                self.print_data(initial_data)
            except:#It might be that the model contains data that is not displayable e.g. 2d data
                pass

        self.plot()

    def print_data(self, data):
        """ Prints 1D-Data represented in a numpy array.
            Args:
                data: 1D Numpy Array containing values to be displayed.
        """
        self.print_indicator = True #As soon as there is data we want to print the indicator upon update
        self.plot_item = self.getPlotItem()

        C=pyqtgraph.hsvColor(1)
        pen=pyqtgraph.mkPen(color=C,width=1)

        X=np.arange(len(data))
        self.indicator_min = int(np.nanmin(data))
        self.indicator_max = int(np.nanmax(data))
        self.main_plot = self.plot_item.plot(X,data,pen=pen,clear=True)

        if self.video_model != None:
            pos = int(self.video_model.get_pos(datatype = self.model.get_datatype()))
            self.indicator = self.plot_item.plot([pos,pos],[self.indicator_min,self.indicator_max],pen=pyqtgraph.mkPen(color=pyqtgraph.hsvColor(2),width=1))


    def update(self, pos = 0, msg = ""):
        """ Updates the image, setting the colormap"""
        if self.print_indicator and self.indicator and not self.video_model == None:
            C=pyqtgraph.hsvColor(1)
            pen=pyqtgraph.mkPen(color=C,width=1)
            data = np.zeros(10)

            pos = int(self.video_model.get_pos(datatype = self.model.get_datatype()))
            self.indicator.setData([pos,pos],[self.indicator_min,self.indicator_max]) #= self.plot_item.plot([pos,pos],[self.indicator_min,self.indicator_max],pen=pyqtgraph.mkPen(color=pyqtgraph.hsvColor(2),width=1))


class MotionWindowSelector(QtGui.QMainWindow):
    """ Window that allows selecting a region of interest to be stored in database """
    def __init__(self, video, parent=None):
        """ Initializes the view and establishes connections of the GUI elements to class methods.
            Args:
                video: Video model to be displayed in background
                parent: Parent QtWidget
        """

        super(MotionWindowSelector, self).__init__(parent)
        self.video = video
        self.setWindowTitle(QtGui.QApplication.translate("Motion Selector", "Motion Selector", None))

        # All the info that is written to the database upon button press
        self.framenumber = 0
        self.coordinates = QRect(0,0,0,0)
        self.comment_text = ""

        # 1. Create Widget
        self.centralwidget = QtGui.QWidget(self)
        self.setFixedSize(1200, 600)

        #2. Create a couple of elements
        self.box = SelectBoxOverlay()

        self.video_plot = VideoPlot(self.video, centered=False)
        self.video_plot.setFixedSize(960,540)#Hardcoded: Width and height are half of 16x9 HD videos

        #self.video.frame.connect(self.video_plot.update)# Uncomment to have
        self.video_plot.installEventFilter(self.box)

        #3.Create Layout
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)

        self.horizontalLayout.addWidget(self.video_plot)#ADD VIDEO HERE
        self.horizontalLayout.addWidget(self.createOptionsGroup())

        #4. Make remaining connections
        self.box.coordinates.connect(self.print_coordinates_change)
        self.box.coordinates.connect(self.set_coordinates)

        video.frame.connect(self.video_plot.update)
        video.title_signal.connect(self.set_title)

        self.video.framenumber.connect(lambda frame: self.current_frame.setText(str(frame)))#Lamda to convert to string
        self.video.framenumber.connect(self.set_framenumber)#
        self.comment.textChanged.connect(self.set_comment)
        self.save_to_database.clicked.connect(self.save)

        self.setLayout(self.horizontalLayout)#First add to layout THEN setLayout THEN setCentralWidget that was used to create layout
        self.setCentralWidget(self.centralwidget)#Essential

    def set_title(self, title):
        title = "Dyad: "+str(self.video.get_dyad())+ "\t\t" + title
        self.l1.setText(title)

    def save(self):
        """ Writes the collected data to the database"""
        dyad = self.video.dyad
        comment = self.comment_text
        video_number = self.video.camera
        start_frame = self.framenumber #correct for default half size pic for selection:
        coordinates = [int(self.coordinates.top()*2),int(self.coordinates.left()*2),int(self.coordinates.bottom()*2),int(self.coordinates.right()*2)]
        if coordinates[0] > coordinates[2]:#ensure to have coordinates correct
            v = coordinates[0]
            coordinates[0] = coordinates[2]
            coordinates[2] = v
        if coordinates[1] > coordinates[3]:
            v = coordinates[1]
            coordinates[1] = coordinates[3]
            coordinates[3] = v
        mother = self.mother
        child = self.child
        self.video.database.add_roi(dyad, video_number, start_frame, coordinates, mother, child, comment)
        self.video.database.save_as_json()
        QMessageBox.about(self, "Saved successfully", "The data was added to the database")

    def print_coordinates_change(self, coordinates):#Qrect
        """ Updates the values of the QLabels displaying the coordinates of the selected region of interest"""
        self.coordinates1.setText(str(coordinates.top()*2))
        self.coordinates2.setText(str(coordinates.left()*2))
        self.coordinates3.setText(str(coordinates.bottom()*2))
        self.coordinates4.setText(str(coordinates.right()*2))

    def set_coordinates(self, coordinates):
        """ Sets the coordinates of the reion of interest"""
        self.coordinates = coordinates

    def set_framenumber(self, framenumber):
        """ Sets the framenumber"""
        self.framenumber = framenumber

    def set_comment(self, comment):
        """ Sets the value for the comment"""
        self.comment_text = str(comment)

    def set_mother(self,b):
        """ Sets attribute mother to True and child to false if checkbox is selected"""
        if b.isChecked() == True:
            self.mother = True
            self.child = False
        else:
            self.mother = False
            self.child = True

    def set_child(self,b):
        """ Sets attribute child to True and mother to false if checkbox is selected"""
        if b.isChecked() == True:
            self.mother = False
            self.child = True
        else:
            self.mother = True
            self.child = False

    def createOptionsGroup(self):
        """ Creates an returns a widget with input/display elements for the data to be stored"""
        #1. Create a widget (here: QGroupBox)
        self.groupBox = QGroupBox()
        self.groupBox.setAlignment(4)

        #2. Create a couple of elements
        self.save_to_database = QtGui.QPushButton()
        self.save_to_database.setText("Save to database")

        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        #Mother or child select

        mother_child = QtGui.QHBoxLayout()
        self.mother_btn = QRadioButton("Mother")
        self.mother_btn.setChecked(True)
        self.mother = True
        mother_child.addWidget(self.mother_btn)
        self.mother_btn.toggled.connect(lambda:self.set_mother(self.mother_btn))


        self.child_btn = QRadioButton("Child")
        self.child_btn.setChecked(False)
        self.child = False
        mother_child.addWidget(self.child_btn)
        self.child_btn.toggled.connect(lambda:self.set_child(self.child_btn))

        #Coordinates display
        self.coordinates1 = QLineEdit()
        self.coordinates1.setFixedWidth(40)
        self.coordinates2 = QLineEdit()
        self.coordinates2.setFixedWidth(40)
        self.coordinates3 = QLineEdit()
        self.coordinates3.setFixedWidth(40)
        self.coordinates4 = QLineEdit()
        self.coordinates4.setFixedWidth(40)

        self.title = "Dyad: "+str(self.video.get_dyad())+ "\t\t Video: " + str(self.video.get_camera())
        self.l1 = QLabel(self.title)
        self.l2 = QLabel("Comment (optional)")
        self.l3 = QLabel("Coordinates")
        self.l4 = QLabel("Current frame")

        self.comment = QLineEdit()
        self.current_frame = QLineEdit()

        #3. Add them to a QVBoxLayout (Vertical)
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.coordinates1)
        hbox.addWidget(self.coordinates2)
        hbox.addWidget(self.coordinates3)
        hbox.addWidget(self.coordinates4)

        vbox = QVBoxLayout()
        vbox.addWidget(self.l1)
        vbox.addWidget(self.line)
        vbox.addWidget(self.l2)
        vbox.addWidget(self.comment)
        vbox.addLayout(mother_child)
        vbox.addWidget(self.l4)
        vbox.addWidget(self.current_frame)
        vbox.addWidget(self.l3)
        vbox.addLayout(hbox)
        vbox.addWidget(self.save_to_database)


        vbox.addStretch(1)#Add empty QSpacerItem that pushes the buttons upwards

        #4. Add layout to widget
        self.groupBox.setLayout(vbox)

        return self.groupBox

class SelectBoxOverlay(QObject):
    """ An event filter that allows the GUI-user to draw a rectangle and retrieval of coordinates"""
    coordinates = pyqtSignal(QRect)

    def __init__(self, parent = None):
        """ Inits the Event Filter
            Args:
                parent: Parent QtWidget
        """
        QObject.__init__(self, parent)
        self.overlay = None# assume overlay does not exist
        self.box_coordinates = [0,0,0,0]

    def eventFilter(self, w, event):
        """ Inits the event filterself.
            Args:
                w: QWidget the event filter is installed at and the select box is drawn in front of
                event: Event to be processed. Here meaningful events are redirected to SelectBoxOverlay.Overlay
        """
        if w.isWidgetType(): #Go through event types if a widget is passed

            #If overlay doesn't exist create it
            if not self.overlay:
                self.overlay = SelectBoxOverlay.Overlay(w.parentWidget())

            #Redirect event types:
            if event.type() == QEvent.MouseButtonPress:
                self.overlay.mousePressEvent(event)

            if event.type() == QEvent.MouseMove:
                self.overlay.mouseMoveEvent(event)

            if event.type() == QEvent.MouseButtonRelease:
                self.overlay.mouseReleaseEvent(event)

            elif event.type() == QEvent.Resize:#Upon resize
                if self.overlay: #If overlay exists (Python evaluates if None as False)
                    self.overlay.setGeometry(w.geometry())#set its geometry to widgets geometry which also causes paintEvent call

            #Set coordinates
            self.box_coordinates = self.overlay.get_box_coordinates()
            self.coordinates.emit(self.box_coordinates)


        return False

    def get_box_coordinates(self):
        """ Returns the coordinates of the select box"""
        return self.box_coordinates

    def toggle_box(self):
        """ Toggles visibility of the select box"""
        if self.overlay:
            self.overlay.show_box()

    class Overlay(QWidget):
        """ Draws a semitransparent select box at the coordinates the GUI-user defines by mouse drag"""
        def __init__(self, parent = None):
            """ Initializes the overlay
            Args:
                Parent: Parent QtWidget
            """
            QWidget.__init__(self, parent)
            self.setAttribute(Qt.WA_NoSystemBackground)
            self.setAttribute(Qt.WA_TransparentForMouseEvents)
            self.begin = QtCore.QPoint(0,0)
            self.end = QtCore.QPoint(0,0)

            self.box_begin = QtCore.QPoint(0,0)#For the final selection
            self.box_end = QtCore.QPoint(0,0)

            self.permanent_show = True

        def get_box_coordinates(self):
            """ returns the coordinates of the current selection"""
            return QRect(self.box_begin,self.box_end)

        def show_box(self):
            """ Toggles visibility of the select box"""
            self.permanent_show = not self.permanent_show

        def paintEvent(self, event):
            """ Draws a semitransparent select box at current coordinates
                Args:
                    event: GUI event
            """
            qp = QtGui.QPainter(self)
            br = QtGui.QBrush(QtGui.QColor(100, 10, 10, 40))
            qp.setBrush(br)
            qp.drawRect(QtCore.QRect(self.begin, self.end))

        def mousePressEvent(self, event):
            """ Resets coordinates of the select box. Sets beginning point to mouse pos.
                Args:
                    event: GUI event
            """
            self.begin = event.pos()
            self.end = event.pos()
            self.update()

        def mouseMoveEvent(self, event):
            """ Sets end point to mouse pos. Updates the select_box overlay.
                Args:
                    event: GUI event
            """
            self.end = event.pos()
            self.update()

        def mouseReleaseEvent(self, event):
            """ Copies the current coordinates to respective attributes.
                If permanent_show is set to false, deletes select_box view.
                Args:
                    event: GUI event
            """
            self.box_begin = self.begin
            self.box_end = event.pos()
            self.begin = event.pos()
            self.end = event.pos()
            if not self.permanent_show:
                self.update()

class DataController(QDialog):
        """ QDialog that represents an interface to the database and allows the user to choose dyads data"""
        NumGridRows = 3
        NumButtons = 4
        title = pyqtSignal(str)

        def __call__(self):
            """ Shows UI when called"""
            self.show()

        def __init__(self, model):
            """ Initializes the Controller
                Args:
                    model: DataModel that represents the data that may chosen to be displayed
            """
            super().__init__()
            self.model = model
            self.setup_ui()
            self.database = self.model.database

            self.datatype = "eeg"
            self.dyad = 1
            self.channel_or_video = 1# Integer for channel or number of video

        def setup_ui(self):
            """ Sets up the User interface and calls make_connections"""
            #super(Dialog, self).__init__()
            self.createFormGroupBox()

            buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttonBox.accepted.connect(self.check_input)
            buttonBox.rejected.connect(self.reject)

            mainLayout = QVBoxLayout()
            mainLayout.addWidget(self.formGroupBox)
            mainLayout.addWidget(buttonBox)
            self.setLayout(mainLayout)

            self.make_connections()

            self.setWindowTitle("Load Data ...")

        def make_connections(self):
            """ Connects the interface elements to the getter / setter methods"""
            try:
                self.datatype.currentIndexChanged.connect(self.set_datatype)
                self.dyad.valueChanged.connect(self.set_dyad)#
                self.vid_or_channel.valueChanged.connect(self.set_channel_or_vid)
            except Exception as e:
                QMessageBox.about(self, str(e))

        def set_datatype(self, datatype):
            """ Sets the datatype of the data to be loaded. Currently eeg and motion are supported
                Args:
                    datatype: Message string that specifies the datatype
            """
            if(datatype == 0):
                self.datatype = "eeg"
            elif(datatype == 1):
                self.datatype = "motion"
            else:
                raise NotImplementedError("EEG and Motion-Data supported only")

        def set_dyad(self, n):
            """ Sets the dyad
                Args:
                    n: dyad number
            """
            try:
                self.database.dictionary[str(n)]
                self.dyad = n
            except KeyError:
                QMessageBox.about(self, "Invalid value", "Setting dyad failed. The datapoint was not found in the database.\nPlease adjust your selection")

            self.title.emit("Dyad: " + str(n))


        def set_channel_or_vid(self, n):
            """ Sets the channel in the case an eeg is loaded or the videocamera respectively"""
            if(self.datatype == "eeg"):
                if((n < 0) or (n >= 64)):
                    raise ValueError("No such channel")
                else:
                    self.channel_or_video = n

            elif(self.datatype == "motion"):
                try:
                    self.database.dictionary[str(self.dyad)]["video"][str(n)]["motion"]["in_roi"]["1"]["path"]
                    self.channel_or_video = n
                except KeyError:
                    QMessageBox.about(self, "Invalid value", "Setting channel failed. The datapoint was not found in the database.\nPlease adjust your selection")
                    #raise ValueError("No such value")
            else:
                raise NotImplementedError


        def createFormGroupBox(self):
            """ Creates UI that allows the selection of data"""
            self.formGroupBox = QGroupBox("Select from database")
            layout = QFormLayout()

            self.dyad = QSpinBox()
            layout.addRow(QLabel("Dyad:"), self.dyad)

            self.datatype = QComboBox()

            self.datatype.addItem("EEG")
            self.datatype.addItem("Motion")

            self.vid_or_channel =  QSpinBox()

            layout.addRow(QLabel("Data Type:"), self.datatype)
            layout.addRow(QLabel("Video/Channel:"), self.vid_or_channel)

            #layout.addRow(QLabel("Name:"), QLineEdit())
            self.formGroupBox.setLayout(layout)

        def check_input(self):
            """ Checks the input for validity and causes data to be loaded. Terminates the dialog i.e. accepts the input."""
            try:
                if(self.datatype == "eeg"):
                    self.model.set_datatype(self.datatype)
                    self.model.set_dyad(self.dyad)
                    self.model.set_channel(self.channel_or_video)#causes loading of data
                elif(self.datatype == "motion"):
                    self.model.set_datatype(self.datatype)
                    self.model.set_filepath(self.database.dictionary[str(self.dyad)]["video"][str(self.channel_or_video)]["motion"]["in_roi"]["1"]["path"])#TODO NOT ALWAYS 1
                    self.model.set_channel(self.channel_or_video)
                else:
                    QMessageBox.about(self, "Incorrect selection", "Choose datatype")
                self.accept()
            except KeyError as e:
                QMessageBox.about(self, "Incorrect selection", "Please choose wisely" + str(e))
