import os
import sys
import cv2
import numpy as np
import tifffile as tiff
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QLineEdit,
    QCheckBox, QSlider, QFileDialog, QListWidget, QListWidgetItem, QFrame,
    QMessageBox, QDoubleSpinBox, QProgressBar, QScrollArea, QSizePolicy, QGraphicsEllipseItem
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect, QTimer
from collections import Counter
import time
import copy
import pickle
import os
import re
import json
import threading
import multiprocessing
 
import json

import gui
import utils
