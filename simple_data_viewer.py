import tkinter
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import os
import analysis_util_functions as uf
import pandas as pd
from read_roi import read_roi_zip
import matplotlib.backends.backend_tkagg as mptk  # import the matplotlib tk backend
import time
import csv


def import_stimulation_file(file_dir):
    # Check for header and delimiter
    with open(file_dir) as csv_file:
        some_lines = csv_file.read(512)
        dialect = csv.Sniffer().sniff(some_lines)
        delimiter = dialect.delimiter

    # Load data and assume the first row is the header
    data = pd.read_csv(file_dir, decimal='.', sep=delimiter, header=0)
    # Chek for correct header
    try:
        a = float(data.keys()[0])  # this means no headers in the file
        data = pd.read_csv(file_dir, decimal='.', sep=delimiter, header=None)
        data.columns = ['Time', 'Volt']
        return data
    except ValueError:
        data = data.drop(data.columns[0], axis=1)
        return data


def import_f_raw(file_dir):
    # Check for header and delimiter
    with open(file_dir) as csv_file:
        dialect = csv.Sniffer().sniff(csv_file.read(32))
        delimiter = dialect.delimiter

    # Load data and assume the first row is the header
    data = pd.read_csv(file_dir, decimal='.', sep=delimiter, header=0).reset_index(drop=True)
    # Chek for correct header
    try:
        a = float(data.keys()[0])
        data = pd.read_csv(file_dir, decimal='.', sep=delimiter, header=None)
    except ValueError:
        a = 0

    header_labels = []
    for kk in range(data.shape[1]):
        header_labels.append(f'roi_{kk + 1}')
    data.columns = header_labels
    return data


class DataViewer:

    def __init__(self, data_path, data, stimulus, fr_data, fr_stimulus, ref_img, rois_draw):
        # Get Input Variables
        self.data_path = data_path
        self.rois_draw = rois_draw
        # self.ref_img = np.invert(ref_img)
        self.ref_img = ref_img
        self.raw_data = data
        self.data = self.raw_data / np.max(self.raw_data, axis=0)
        self.stimulus = stimulus
        self.fr_data = fr_data
        self.fr_stimulus = fr_stimulus
        self.rois = self.data.keys()
        self.id = 0
        # Convert Samples to Time
        self.time_data = np.linspace(0, len(self.data) / self.fr_data, len(self.data))
        self.time_stimulus = np.linspace(0, len(self.stimulus) / self.fr_stimulus, len(self.stimulus))
        # Compute Delta F over F
        self.fbs = np.percentile(self.raw_data, 5, axis=0)
        self.data = (self.raw_data - self.fbs) / self.fbs

        self.root = tkinter.Tk()
        self.window_title = os.path.split(self.data_path)[1]
        self.root.wm_title(self.window_title)
        # Set Window to Full Screen
        self.root.state('zoomed')

        # Compute Ref Images
        self._compute_ref_images()

        # FIGURE START
        self.fig, self.ax = plt.subplots(2, 1)
        self.ax_ref_image = self.ax[0]
        self.ax_trace = self.ax[1]
        # Plot Reference Image
        self.ref_img_obj = self.ax_ref_image.imshow(self.roi_images[self.id], cmap='YlOrBr')
        self.ax_ref_image.axis('off')  # clear x-axis and y-axis
        self.ref_img_text = self.ax_ref_image.text(
            0, 0, f'{self.id + 1 }', fontsize=12, color=(1, 0, 0),
            horizontalalignment='center', verticalalignment='center'
        )
        self.ref_img_text.set_position(self.roi_pos[self.id])

        # Plot Response and Stimulus Traces
        self.data_to_plot, = self.ax_trace.plot(self.time_data, self.data[self.rois[self.id]], 'k')
        if not self.stimulus.empty:
            self.stimulus_norm = (self.stimulus / np.max(self.stimulus)) * np.max(self.data.max())
            self.stimulus_to_plot, = self.ax_trace.plot(self.time_stimulus, self.stimulus_norm, 'b', alpha=0.25)
        self.title_obj = self.ax_trace.set_title(f'roi: {self.rois[self.id]} / {len(self.rois)}')
        self.ax_trace.set_ylim([-0.5, np.max(self.data.max())])
        # FIGURE END

        # Add Menu
        self.menu_bar = tkinter.Menu(self.root)
        self.file_menu = tkinter.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Open", command=self.open_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self._quit)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.root.config(menu=self.menu_bar)

        self.frame = tkinter.Frame(master=self.root)
        self.frame.pack(side=tkinter.TOP)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar = CustomToolbar(self.canvas, self.root)
        self.toolbar.update()
        # side: align to top (put it below the other)
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # Add Buttons
        self.button = tkinter.Button(master=self.root, text="Quit", command=self._quit)
        self.button.pack(side=tkinter.BOTTOM)

        self.button_1 = tkinter.Button(master=self.root, text="PREVIOUS", command=self._previous)
        self.button_1.pack(side=tkinter.LEFT)

        self.button_2 = tkinter.Button(master=self.root, text="NEXT", command=self._next)
        self.button_2.pack(side=tkinter.RIGHT)

        self.canvas.mpl_connect("key_press_event", self.on_key_press)

    def _compute_ref_images(self):
        # Compute Ref Image
        self.roi_images = []
        self.roi_pos = []
        for ii, active_roi in enumerate(self.rois_draw):
            if self.rois_draw[active_roi]['type'] == 'freehand':
                x_center = np.mean(self.rois_draw[active_roi]['x'])
                y_center = np.mean(self.rois_draw[active_roi]['y'])
                self.roi_pos.append((x_center, y_center))
            else:
                self.roi_pos.append((self.rois_draw[active_roi]['left'] + self.rois_draw[active_roi]['width'] // 2,
                                self.rois_draw[active_roi]['top'] + self.rois_draw[active_roi]['height'] // 2))
            self.roi_images.append(uf.draw_rois_zipped(
                img=self.ref_img, rois_dic=self.rois_draw, active_roi=self.rois_draw[active_roi],
                r_color=(0, 0, 255), alp=0.5, thickness=1)
            )

    def open_file(self):
        # file_types example: [('Recording Files', 'raw.txt')]
        f_file_name = filedialog.askopenfilename(filetypes=[('Recording Files', 'raw.txt')])
        f_rec_dir = os.path.split(f_file_name)[0]
        f_rec_name = os.path.split(f_rec_dir)[1]


    def on_key_press(self, event):
        if event.key == 'down':
            # Decrease Index by 1
            self._previous()
            # self.redbutton['text'] = 'This was LEFT'
            # self.redbutton['bg'] = 'green'
        elif event.key == 'up':
            self._next()
            # self.redbutton['text'] = 'This was RIGHT'
            # self.redbutton['bg'] = 'black'

        elif event.key == 'q':
            self._quit()

        # print("you pressed {}".format(event.key))
        key_press_handler(event, self.canvas, self.toolbar)

    def _quit(self):
        self.root.quit()     # stops mainloop
        # this is necessary on Windows to prevent Fatal Python Error: PyEval_RestoreThread: NULL state
        self.root.destroy()

    def _next(self):
        self.id += 1
        self.id = self.id % self.data.shape[1]
        self._turn_page()

    def _previous(self):
        self.id -= 1
        self.id = self.id % self.data.shape[1]
        self._turn_page()

    def _turn_page(self):
        self.data_to_plot.set_ydata(self.data[self.rois[self.id]])
        self.title_obj = self.ax_trace.set_title(f'roi: {self.rois[self.id]} / {len(self.rois)}')
        # Update Ref Image
        self.ref_img_obj.set_data(self.roi_images[self.id])
        # Update Ref Image Label
        self.ref_img_text.set_text(f'{self.id + 1}')
        self.ref_img_text.set_position(self.roi_pos[self.id])

        if self.id == len(self.rois) - 1:
            plt.setp(self.title_obj, color='r')
        else:
            plt.setp(self.title_obj, color='k')
        self.canvas.draw()


class CustomToolbar(mptk.NavigationToolbar2Tk):  # subclass NavigationToolbar2Tk
    # This class inherits from NavigationToolbar2Tk
    # Than just copy the draw_rubberband fuction and change the line with the color
    def __init__(self, figcanvas, parent):
        super().__init__(figcanvas, parent)  # init the base class as usual

    # copy the method 'draw_rubberband()' right from NavigationToolbar2Tk
    # change only one line (color from black to whatever)
    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.remove_rubberband()
        height = self.canvas.figure.bbox.height
        y0 = height - y0
        y1 = height - y1
        # this is the bit we want to change...
        self.lastrect = self.canvas._tkcanvas.create_rectangle(x0, y0, x1, y1, outline='red')
        # hex color strings e.g.'#BEEFED' and named colors e.g. 'gainsboro' both work


if __name__ == "__main__":
    # Select Directory and get all files
    file_name = uf.select_file([('Recording Files', 'raw.txt')])
    data_file_name = os.path.split(file_name)[1]
    rec_dir = os.path.split(file_name)[0]
    rec_name = os.path.split(rec_dir)[1]
    uf.msg_box(rec_name, f'SELECTED RECORDING: {rec_name}', '+')

    # Import Reference Image
    img_ref = plt.imread(f'{rec_dir}/{rec_name}_ref.tif', format='tif')
    # Import ROIS from Imagej
    rois_in_ref = read_roi_zip(f'{rec_dir}/{rec_name}_RoiSet.zip')

    # Import Raw Values
    f_raw = import_f_raw(f'{rec_dir}/{rec_name}_raw.txt')
    # From Resolution ---> Frame Rate
    file_list = os.listdir(rec_dir)
    stimulation_file = [s for s in file_list if 'stimulation' in s]
    if stimulation_file:
        # Get Stimulus
        # stimulation = import_stimulation_file(f'{rec_dir}/{rec_name}_stimulation.txt')
        stimulation = pd.read_csv(f'{rec_dir}/{rec_name}_stimulation.txt')
        if stimulation['Volt'].max() <= 2:
            stimulation['Volt'] = stimulation['Volt'] * -1
        # Get Raw Values
        fr_rec = uf.estimate_sampling_rate(data=f_raw, f_stimulation=stimulation, print_msg=True)
        stimulation_trace = stimulation['Volt']
    else:
        stimulation_trace = pd.DataFrame()
        fr_rec = uf.pixel_resolution_to_frame_rate(img_ref.shape[0])
    # Start Data Viewer
    app = DataViewer(
        data_path=rec_dir,
        data=f_raw,
        stimulus=stimulation_trace,
        fr_data=fr_rec,
        fr_stimulus=1000,
        ref_img=img_ref,
        rois_draw=rois_in_ref
    )
    tkinter.mainloop()
    # If you put root.destroy() here, it will cause an error if the window is
    # closed with the window manager.()
