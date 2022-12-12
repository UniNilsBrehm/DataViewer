import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import tkinter.messagebox
import os
import csv
import cv2
import pandas as pd
import numpy as np
from dataclasses import dataclass
from IPython import embed
from read_roi import read_roi_zip
import matplotlib.pyplot as plt
# Implement the default Matplotlib key bindings.
# from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import NavigationToolbar2

# plt.rcParams['toolbar'] = 'toolmanager'


@dataclass
class Browser:
    data_file: str
    stimulus_file: str
    reference_image_file: str
    roi_file: str


class MainApplication(tk.Frame):

    def __init__(self, master):
        self.master = master
        tk.Frame.__init__(self, self.master)
        # Initialize class parameters
        # Prepare an empty data frame for recording data
        self.data_raw = pd.DataFrame()
        self.data_df = pd.DataFrame()
        self.data_z = pd.DataFrame()
        self.data_time = pd.DataFrame()
        self.data_fr = 0
        self.data_rois = pd.DataFrame()
        self.data_id = 0
        self.data_id_prev = 0

        self.stimulus_found = False
        self.stimulus = pd.DataFrame()
        self.first_start_up = True

        self.ref_img_found = False
        # self.ref_img = plt.imread('startup_logo.jpg')
        self.ref_img = None
        self.rois_in_ref = None
        self.ref_img_obj = None
        self.ref_img_text = None

        # Create boolean to state if new data was loaded
        self.new_data_loaded = False

        # Create a dataclass to store directories for browsing files
        self.browser = Browser
        self.browser.data_file = ''
        self.browser.stimulus_file= ''
        self.browser.reference_image_file = ''
        self.browser.roi_file = ''

        # Add the file menu
        self.add_file_menu()

        # Add The Main Windows in the App
        # Frame for ToolBar
        self.frame_toolbar_ref = tk.Frame(self.master)
        self.frame_toolbar_ref.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        self.frame_toolbar = tk.Frame(self.master)
        self.frame_toolbar.grid(row=2, column=0, padx=5, pady=5, sticky='ew')

        # Frame for the Reference Image
        self.frame_ref_img = tk.Frame(self.master)
        self.frame_ref_img.grid(row=1, column=0, padx=5, pady=5, sticky='ew')

        # Frame for ROIs List
        self.frame_rois_list = tk.Frame(self.master)
        self.frame_rois_list.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        self.list_items = tk.Variable(value=[])

        self.listbox = tk.Listbox(self.frame_rois_list, listvariable=self.list_items, height=10, selectmode='SINGLE',
                                  takefocus=0)
        self.listbox.grid(row=0, column=0)
        # Add scrollbar
        self.scroll_bar = tk.Scrollbar(self.frame_rois_list)
        self.scroll_bar.grid(row=0, column=1, sticky=tk.N+tk.S+tk.W)

        self.listbox.config(yscrollcommand=self.scroll_bar.set)
        self.scroll_bar.config(command=self.listbox.yview)

        # Frame for the Data Figure
        self.frame_traces = tk.Frame(self.master)
        self.frame_traces.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky='ew')
        # self.frame_traces.bind('<Enter>', lambda event: self.enter_frame_traces())

        # Key Bindings
        self.master.bind('<Left>', self._previous)
        self.master.bind('<Right>', self._next)
        self.master.bind('q', self._quit)
        self.listbox.bind('<Down>', self.do_nothing)
        self.listbox.bind('<Up>', self.do_nothing)

        # Prepare Figures
        # Ref Image with ROIs
        self.fig_ref, self.axs_ref = plt.subplots()
        self.axs_ref.axis('off')

        # Data Plot
        self.fig, self.axs = plt.subplots()
        # self.fig = plt.figure()

        # Those will contain the actual data that is plotted later on
        self.stimulus_plot, = self.axs.plot(0, 0)
        self.data_plot, = self.axs.plot(0, 0)
        self.import_data_text = self.axs.text(0.5, 0.5, 'Click here to import Data', ha='center', va='center', transform=self.axs.transAxes)
        self.axs.axis('off')

        # Prepare Canvas
        # Add Canvas for MatPlotLib Figures
        self.canvas_ref = FigureCanvasTkAgg(self.fig_ref, master=self.frame_ref_img)  # A tk.DrawingArea.
        self.canvas_ref.draw()
        self.canvas_ref.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_traces)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # self.canvas._tkcanvas.bind("<Key>", self.print_something)
        # self.canvas_ref._tkcanvas.bind("<Key>", self.print_something)

        # Catch Mouse Clicks on Figure Canvas
        self.canvas.mpl_connect('button_press_event', self.canvas_on_mouse_click)
        self.canvas_ref.mpl_connect("button_press_event", self.select_roi_with_mouse)
        self.canvas_ref.mpl_connect("key_press_event", self.detect_key)

        # # Add Toolbars
        # self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame_toolbar)
        # self.toolbar.pack(side=tk.TOP)
        # self.toolbar_ref = NavigationToolbar2Tk(self.canvas_ref, self.frame_toolbar_ref)
        # self.toolbar_ref.pack(side=tk.TOP)
        self.toolbar = CustomNavigationToolbar2Tk(self.canvas, self.frame_toolbar)
        self.toolbar.pack(side=tk.TOP)
        self.toolbar_ref = CustomNavigationToolbar2Tk(self.canvas_ref, self.frame_toolbar_ref)
        self.toolbar_ref.pack(side=tk.TOP)

        # List information about toolbar buttons
        # print(self.toolbar.toolitems)

        # Remove some tools from the toolbar
        # self.toolbar.children['!button2'].pack_forget()
        # self.toolbar.children['!button3'].pack_forget()
        # self.toolbar_ref.children['!button2'].pack_forget()
        # self.toolbar_ref.children['!button3'].pack_forget()

        # Add Vertical Toolbar
        # self.toolbar_ref = VerticalNavigationToolbar2Tk(self.canvas_ref, self.frame_toolbar_ref)
        # self.toolbar_ref.update()
        # self.toolbar_ref.pack(side=tk.TOP, fill=tk.Y)

        self.configure_gui()

    def detect_key(self, event):
        if event.key == 'left':
            self.print_something(event)
        if event.key == 'right':
            self.print_something(event)

    def select_roi_with_mouse(self, event):
        click_x = event.xdata
        click_y = event.ydata
        error = []
        for k in self.roi_pos:
            x_diff = (k[0] - click_x)**2
            y_diff = (k[1] - click_y)**2
            error.append(np.sqrt(x_diff + y_diff))
        pixel_th = 10
        if np.min(error) > pixel_th:
            # There is no ROI located close enough to the click
            return "break"
        idx_min = np.where(error == np.min(error))[0][0]
        _data_id = idx_min
        # Update Listbox
        self.listbox.selection_clear(self.data_id)
        self.listbox.activate(_data_id)
        self.listbox.select_set(self.listbox.index(tk.ACTIVE))
        self.listbox.event_generate("<<ListboxSelect>>")
        self.listbox.see(_data_id)

    def canvas_on_mouse_click(self, event):
        if event.dblclick and not self.new_data_loaded:
            self._import_data()

    def _move_scrollbar_down(self):
        pos_top, pos_bottom = self.scroll_bar.get()
        ds = 0.05
        if self.listbox.index(tk.ACTIVE) > 10:
            # Adjust scrollbar position
            self.scroll_bar.set(pos_top+ds, pos_bottom-ds)

    def configure_gui(self):
        # Set Window to Full Screen
        self.master.state('zoomed')
        self.master.grid_columnconfigure(0, weight=1)

    def enter_frame_traces(self):
        print('ENTERED!')

    def add_file_menu(self):
        # Add Menu
        menu_bar = tk.Menu(self.master)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Data File", command=self._import_data)
        file_menu.add_command(label="Import Stimulus", command=self._open_stimulus_file)
        file_menu.add_command(label="Calcium Impulse Response Function", command=self._cirf_creater)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._exit_app)
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.master.config(menu=menu_bar)

    def _cirf_creater(self):
        # Pop up a new window
        # time_axis = np.linspace(0, 10, 1000)
        self.dt = 0.0001
        data, time_axis = self.cirf_double_tau(tau1=1, tau2=2, a=1, dt=self.dt)
        self.cirf_fig, self.cirf_ax = plt.subplots()
        self.cirf_plot, = self.cirf_ax.plot(time_axis, data, 'k', lw=2)
        self.cirf_ax.set_xlabel('Time [s]')
        self.cirf_ax.set_ylabel('Y-Value [a.u.]')

        tau2_max = 10.0
        self.cirf_ax.set_xlim((-1, 5 * tau2_max))
        self.cirf_ax.set_ylim((0, 1))

        self.cirf_window = tk.Toplevel(self.master)
        self.cirf_window.title('Calcium Impulse Response Function')
        self.cirf_window.geometry("800x800")

        # Label
        slider_font_size = 14
        cirf_label = tk.Label(self.cirf_window,
                              text='Calcium Impulse Response Function: cirf = A * (1-exp{-t/tau1}) * exp{-t/tau2}',
                              font=("Arial", 16))
        cirf_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Sliders
        tau1_label = tk.Label(self.cirf_window, text='Tau 1 [s]', font=("Arial", slider_font_size))
        tau1_label.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.N, tk.E))
        self.tau1_current_value = tk.DoubleVar()
        self.tau1_current_value.set(1)
        tau1_slider = tk.Scale(self.cirf_window, from_=0.01, to=10.0, orient=tk.HORIZONTAL,
                               resolution=0.01, variable=self.tau1_current_value, command=self.change_cirf)
        tau1_slider.grid(row=2, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))

        tau2_label = tk.Label(self.cirf_window, text='Tau 2 [s]', font=("Arial", slider_font_size))
        tau2_label.grid(row=3, column=0, padx=5, pady=5, sticky=(tk.N, tk.E))
        self.tau2_current_value = tk.DoubleVar()
        self.tau2_current_value.set(2)
        tau2_slider = tk.Scale(self.cirf_window, from_=0.01, to=tau2_max, orient=tk.HORIZONTAL, resolution=0.01,
                               variable=self.tau2_current_value, command=self.change_cirf)
        tau2_slider.grid(row=3, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))

        a_label = tk.Label(self.cirf_window, text='Amplitude', font=("Arial", slider_font_size))
        a_label.grid(row=4, column=0, padx=5, pady=5, sticky=(tk.N, tk.E))
        self.a_current_value = tk.DoubleVar()
        self.a_current_value.set(1)
        a_slider = tk.Scale(self.cirf_window, from_=0.1, to=5, orient=tk.HORIZONTAL, resolution=0.1,
                               variable=self.a_current_value, command=self.change_cirf)
        a_slider.grid(row=4, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))

        self.cirf_canvas = FigureCanvasTkAgg(self.cirf_fig, master=self.cirf_window)  # A tk.DrawingArea.
        # self.cirf_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.cirf_canvas.get_tk_widget().grid(row=1, column=0, columnspan=2)
        self.cirf_canvas.draw()

    def change_cirf(self, event):
        tau1 = self.tau1_current_value.get()
        tau2 = self.tau2_current_value.get()
        a = self.a_current_value.get()
        c, t = self.cirf_double_tau(tau1, tau2, a, dt=self.dt)
        self.cirf_plot.set_xdata(t)
        self.cirf_plot.set_ydata(c)
        # Adapt x and y limits of the figure
        if np.max(c) > 1:
            if np.max(c) <= 5:
                self.cirf_ax.set_ylim((0, 5))
            else:
                self.cirf_ax.set_ylim((0, 10))
        else:
            self.cirf_ax.set_ylim((0, 1))
        # self.cirf_ax.set_xlim((0, np.max(t) + (np.max(t) * 0.05)))
        # self.cirf_ax.set_ylim((0, np.max(c) + (np.max(c) * 0.05)))
        self.cirf_canvas.draw()

    def _compute_ref_images(self):
        # Compute Ref Image
        self.roi_images = []
        self.roi_pos = []
        for ii, active_roi in enumerate(self.rois_in_ref):
            if self.rois_in_ref[active_roi]['type'] == 'freehand':
                x_center = np.mean(self.rois_in_ref[active_roi]['x'], axis=0)
                y_center = np.mean(self.rois_in_ref[active_roi]['y'], axis=0)
                self.roi_pos.append((x_center, y_center))
            else:
                self.roi_pos.append((self.rois_in_ref[active_roi]['left'] + self.rois_in_ref[active_roi]['width'] // 2,
                                self.rois_in_ref[active_roi]['top'] + self.rois_in_ref[active_roi]['height'] // 2))
            self.roi_images.append(self.draw_rois_zipped(
                img=self.ref_img, rois_dic=self.rois_in_ref, active_roi=self.rois_in_ref[active_roi],
                r_color=(0, 0, 255), alp=0.5, thickness=1)
            )

    def _open_stimulus_file(self):
        # Pop up a new window
        self.import_window = tk.Toplevel(self.master)
        self.import_window.title('Import Stimulus File')
        self.import_window.geometry("600x600")
        # import_window.attributes('-topmost', True)

        # Add Text Label (Description)
        self.description_text = tk.Label(self.import_window, text="Please select a stimulus file and adjust the settings:")
        self.description_text.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Add Open File Button
        open_button = tk.Button(self.import_window, text="Open File ...", command=self._browse_file)
        open_button.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Add Label for File Name
        self.file_name_label = tk.Label(self.import_window, text="No File selected")
        self.file_name_label.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Delimiter Settings
        self.delimiter_label = tk.Label(self.import_window, text="Delimiter: ")
        self.delimiter_label.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))
        delimiter_options = ['Comma', 'Tab', 'Semicolon', 'Space', 'Colon']
        self.delimiter_variable = tk.StringVar(self.import_window)
        self.delimiter_variable.set(delimiter_options[0])  # default value
        self.delimiter_dropdown = tk.OptionMenu(self.import_window, self.delimiter_variable, *delimiter_options)
        self.delimiter_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))
        self.delimiter_dropdown.config(width=10, font=('Arial', 8))

        # Which column contains data?
        self.cols_label = tk.Label(self.import_window, text="Column Data: ")
        self.cols_label.grid(row=3, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))
        self.cols = tk.Entry(self.import_window)
        self.cols.grid(row=3, column=1)

        self.cols_time_label = tk.Label(self.import_window, text="Column Time Axis (none: 0): ")
        self.cols_time_label.grid(row=4, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))
        self.cols_time = tk.Entry(self.import_window)
        self.cols_time.grid(row=4, column=1)

        # Add Header Options
        self.has_header = tk.IntVar()
        self.has_header_button = tk.Checkbutton(self.import_window, text="File has Column Headers: ",
                                                variable=self.has_header, onvalue=0, offvalue=1)
        self.has_header_button.grid(row=5, column=0)

        self.import_button = tk.Button(self.import_window, text='Done', command=self._collect_info)
        self.import_button.grid(row=7, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))

    def _collect_info(self):
        # All inputs are strings!
        delimiter = self.convert_to_delimiter(self.delimiter_variable.get())
        data_column_number = self.cols.get()
        time_column_number = self.cols_time.get()
        header_button_state = self.has_header.get()
        if header_button_state == 0:
            header = 0
        else:
            header = None

        # Check if entries are correct
        try:
            data_column_number = int(data_column_number) - 1
            time_column_number = int(time_column_number) - 1
            if time_column_number == data_column_number:
                tk.messagebox.showerror(title='Wrong Input',
                                        message='Time and Data Column cannot be the same!')
                self.import_window.lift()
                return None

            stimulus_file = pd.read_csv(self.browser.file_dir, delimiter=delimiter, header=header)
            if data_column_number > stimulus_file.shape[1]:
                tk.messagebox.showerror(title='Wrong Input',
                                        message='Data Column number exceeds size of selected file!')
                self.import_window.lift()
                return None
            if time_column_number >= 0:
                if time_column_number > stimulus_file.shape[1]:
                    tk.messagebox.showerror(title='Wrong Input',
                                            message='Time Column number exceeds size of selected file!')
                    self.import_window.lift()
                    return None
                else:
                    self.stimulus['Time'] = stimulus_file.iloc[:, time_column_number]
            self.stimulus['Volt'] = stimulus_file.iloc[:, data_column_number]
        except ValueError:
            tk.messagebox.showerror(title='Wrong Input', message='Only Integer Number are allowed for Column numbers!')
            # Bring Stimulus Import Window back to top
            self.import_window.lift()
            return None
        # Now Update The Figure
        print(self.stimulus)
        self.update_stimulus_trace()

    def update_stimulus_trace(self):
        self.axs.set_visible(True)
        self.stimulus_plot.set_xdata(self.stimulus['Time'])
        self.stimulus_plot.set_ydata(self.stimulus['Volt'])
        self.canvas.draw()
        print('Stimulus updated')

    def initialize_stimulus_plot(self):
        self.axs.set_visible(True)
        self.stimulus_plot, = self.axs.plot(self.stimulus['Time'], self.stimulus['Volt'], 'b')
        self.canvas.draw()
        print('Stimulus drawn to Canvas')

    def initialize_recording_plot(self):
        self.axs.set_visible(True)
        self.data_plot, = self.axs.plot(self.data_time, self.data_df[self.data_rois[self.data_id]], 'k')
        self.canvas.draw()
        print('Recording drawn to Canvas')

    def update_recording_trace(self):
        self.axs.set_visible(True)
        self.data_plot.set_ydata(self.data_df[self.data_rois[self.data_id]])
        self.data_plot.set_xdata(self.data_time)
        self.canvas.draw()
        print('Recording updated')

    def update_reference_image(self):
        self.axs_ref.set_visible(True)

    def _browse_file(self, file_type, file_extension):
        # file_extension must be: [('Recording Files', '.txt')]
        file_dir = filedialog.askopenfilename(filetypes=file_extension)
        file_name = os.path.split(file_dir)[1]
        if file_type == 'data':
            self.browser.data_file = file_dir
            # Update Label
            self.data_file_name_label.config(text=file_name)
        elif file_type == 'stimulus':
            self.browser.stimulus_file = file_dir
            # Update Label
            self.stimulus_file_name_label.config(text=file_name)
        elif file_type == 'reference':
            self.browser.reference_image_file = file_dir
            # Update Label
            self.reference_file_name_label.config(text=file_name)
        elif file_type == 'rois':
            self.browser.roi_file = file_dir
            # Update Label
            self.rois_file_name_label.config(text=file_name)
        else:
            print('ERROR: Invalid File Type')
            return "break"
        # Bring Stimulus Import Window back to top
        self.import_window.lift()

    def _next(self, event):
        self.listbox.selection_clear(self.data_id)
        self.data_id_prev = self.data_id
        _data_id = self.data_id + 1
        _data_id = _data_id % self.data_raw.shape[1]

        # Update Listbox
        self.listbox.activate(_data_id)
        self.listbox.select_set(self.listbox.index(tk.ACTIVE))
        self.listbox.event_generate("<<ListboxSelect>>")
        self.listbox.see(_data_id)

    def _previous(self, event):
        # print(event)
        self.listbox.selection_clear(self.data_id)
        _data_id = self.data_id - 1
        _data_id = _data_id % self.data_raw.shape[1]

        # Update Listbox
        self.listbox.activate(_data_id)
        self.listbox.select_set(self.listbox.index(tk.ACTIVE))
        self.listbox.event_generate("<<ListboxSelect>>")
        self.listbox.see(_data_id)

    def _turn_page(self):
        # Update Recording Plot Data
        self.data_plot.set_ydata(self.data_df[self.data_rois[self.data_id]])

        # Update Ref Image
        if self.import_rois_variable.get() == 1:
            self.ref_img_obj.set_data(self.roi_images[self.data_id])
            # Update Ref Image Label
            self.ref_img_text[self.data_id_prev].set_color((1, 1, 1))
            self.ref_img_text[self.data_id].set_color((1, 0, 0))
            self.canvas_ref.draw()
        self.canvas.draw()

    def _update_plotting_data(self):
        self.data_plot.set_ydata(self.data_df[self.data_rois[self.data_id]])

    def list_items_selected(self, event):
        self.data_id_prev = self.data_id
        self.data_id, = self.listbox.curselection()
        self._turn_page()

    def _open_file(self):
        self.import_data_text.set_visible(False)
        f_file_name = filedialog.askopenfilename(filetypes=[('Recording Files', '.txt')])
        f_rec_dir = os.path.split(f_file_name)[0]
        # f_rec_name = os.path.split(f_rec_dir)[1]
        #
        # f_file_name, f_rec_dir, _ = self.browse_file()

        # Get Stimulus File
        file_list = os.listdir(f_rec_dir)
        stimulation_file = [s for s in file_list if 'stimulation' in s]
        if stimulation_file:
            stimulus_file_dir = f'{f_rec_dir}/{stimulation_file[0]}'
            self.stimulus_found = True
            self.stimulus = self.import_stimulation_file(stimulus_file_dir)
        else:
            print('COULD NOT FIND STIMULUS FILE!')
            self.stimulus = pd.DataFrame()
            self.stimulus_found = False

        # Get Reference Image
        ref_file = [s for s in file_list if 'ref' in s]
        if ref_file:
            self.ref_img_found = True
            ref_img_file_dir = f'{f_rec_dir}/{ref_file[0]}'
            self.ref_img = plt.imread(ref_img_file_dir)
        else:
            print('COULD NOT FIND REFERENCE IMAGE FILE!')
            self.ref_img_found = False
            self.ref_img = None

        # Get Image ROIs
        roi_file = [s for s in file_list if 'Roi' in s]
        if roi_file:
            self.rois_in_ref = read_roi_zip(f'{f_rec_dir}/{roi_file[0]}')
        else:
            print('COULD NOT FIND IMAGEJ ROIS!')
            self.rois_in_ref = None
        # Now compute Reference Images with Rois drawn on top
        self._compute_ref_images()

        # Get Recording Data
        self.data_raw = self.import_f_raw(file_dir=f_file_name)
        self.new_data_loaded = True

        # Get ROIs
        self.data_rois = self.data_raw.keys()
        self.data_id = 0

        # Convert to delta f over f
        self.data_df = self.convert_raw_to_df_f(self.data_raw)
        # Compute z-scores
        self.data_z = self.compute_z_score(self.data_df)
        # Estimate Data Frame Rate, with not specified
        if self.stimulus_found:
            self.data_fr = self.estimate_sampling_rate(data=self.data_raw, f_stimulation=self.stimulus, print_msg=True)
        else:
            print('PLEASE SPECIFY SAMPLING RATE ...')
            self.data_fr = 2.034514712575680
        # Compute data time axis
        self.data_time = self.convert_samples_to_time(sig=self.data_raw, fr=self.data_fr)

        # Fill List with rois
        # First clear listbox
        self.listbox.delete(0, tk.END)
        rois_numbers = np.linspace(1, len(self.data_rois.to_list()), len(self.data_rois.to_list()), dtype='int')
        rois_numbers = np.char.mod('%d', rois_numbers)
        self.listbox.insert("end", *rois_numbers)
        self.listbox.bind('<<ListboxSelect>>', self.list_items_selected)

        # INITIALIZE DATA AND REFERENCE FIGURES
        # If this is the first time that any data was loaded, create the plot
        if self.first_start_up:
            self.first_start_up = False
            # Plot Ref Image
            if self.ref_img_found:
                self.ref_img_obj = self.axs_ref.imshow(self.roi_images[self.data_id])
                self.ref_img_text = []
                for i, v in enumerate(self.roi_pos):
                    if i == self.data_id:
                        color = (1, 0, 0)
                    else:
                        color = (1, 1, 1)
                    self.ref_img_text.append(self.axs_ref.text(
                        v[0], v[1], f'{i + 1}', fontsize=10, color=color,
                        horizontalalignment='center', verticalalignment='center'
                    ))

                # self.ref_img_text = self.axs_ref.text(
                #     0, 0, f'{self.data_id + 1}', fontsize=12, color=(1, 0, 0),
                #     horizontalalignment='center', verticalalignment='center'
                # )
                # self.ref_img_text.set_position(self.roi_pos[self.data_id])
                self.axs_ref.axis('off')
            # Plot Data
            if not self.stimulus.empty:
                self.stimulus_plot, = self.axs.plot(self.stimulus['Time'], self.stimulus['Volt'], 'b')
            self.data_plot, = self.axs.plot(self.data_time, self.data_df[self.data_rois[self.data_id]], 'k')
        self.axs.set_visible(True)
        self.axs_ref.set_visible(True)
        self.canvas.draw()
        self.canvas_ref.draw()

    def _switch_stimulus_import(self):
        switch_var = self.import_stimulus_variable.get()
        if switch_var == 0:  # this means no stimulus file is available
            # No Stimulus File is selected
            self.select_stimulus_button.config(state='disable')
            self.stimulus_file_name_label.config(state='disable')
            self.delimiter_dropdown.config(state='disable')
            self.stimulus_delimiter_label.config(state='disable')
            self.stimulus_has_header_button.config(state='disable')
            self.cols.config(state='disable')
            self.cols_time.config(state='disable')
            self.cols_label.config(state='disable')
            self.cols_time_label.config(state='disable')

            self.stimulus_frame_rate_label.config(state='disable')
            self.stimulus_frame_rate_input.config(state='disable')
            self.stimulus_time_axis_checkbutton.config(state='disable')

            self.stimulus_time_axis_checker.set(value=0)

            # Turn on manual frame rate input for the recording
            self.frame_rate_input.config(state='normal')
            print(switch_var)
        if switch_var == 1:
            # Stimulus File is selected
            self.select_stimulus_button.config(state='normal')
            self.stimulus_file_name_label.config(state='normal')
            self.delimiter_dropdown.config(state='normal')
            self.stimulus_delimiter_label.config(state='normal')
            self.stimulus_has_header_button.config(state='normal')
            self.cols.config(state='normal')
            self.cols_time.config(state='normal')
            self.cols_label.config(state='normal')
            self.cols_time_label.config(state='normal')

            self.stimulus_time_axis_checkbutton.config(state='normal')
            self.stimulus_time_axis_checker.set(value=1)
            # Turn off manual frame rate input for the recording
            self.frame_rate_input.config(state='disabled')
            print(switch_var)

    def _stimulus_sampling_rate_switcher(self):
        switch_var = self.stimulus_time_axis_checker.get()
        if switch_var == 0:  # this means no time axis
            self.cols_time.config(state='disable')
            self.cols_time_label.config(state='disable')
            self.stimulus_frame_rate_label.config(state='normal')
            self.stimulus_frame_rate_input.config(state='normal')
            if self.import_data_variable.get() == 1:
                self.frame_rate_input.config(state='normal')
                self.frame_rate_label.config(state='normal')
            else:
                self.frame_rate_input.config(state='disable')
                self.frame_rate_label.config(state='disable')
        if switch_var == 1:
            self.cols_time.config(state='normal')
            self.cols_time_label.config(state='normal')
            self.stimulus_frame_rate_label.config(state='disable')
            self.stimulus_frame_rate_input.config(state='disable')
            self.frame_rate_input.config(state='disable')
            self.frame_rate_label.config(state='disable')

    def _import_data_switcher(self):
        switch_var = self.import_data_variable.get()
        if switch_var == 0:  # this means no data selected
            self.select_recording_button.config(state='disable')
            self.data_file_name_label.config(state='disable')
            self.data_delimiter_label.config(state='disable')
            self.delimiter_data_dropdown.config(state='disable')
            self.data_has_header_button.config(state='disable')
            self.frame_rate_input.config(state='disabled')
            self.frame_rate_label.config(state='disable')

        if switch_var == 1:
            self.select_recording_button.config(state='normal')
            self.data_file_name_label.config(state='normal')
            self.data_delimiter_label.config(state='normal')
            self.delimiter_data_dropdown.config(state='normal')
            self.data_has_header_button.config(state='normal')
            if self.stimulus_time_axis_checker.get() == 0:
                self.frame_rate_input.config(state='normal')
                self.frame_rate_label.config(state='normal')

    def _import_reference_switch(self):
        switch_var = self.import_reference_variable.get()
        if switch_var == 0:  # this means no reference image selected
            self.select_reference_button.config(state='disable')
            self.reference_file_name_label.config(state='disable')
            self.import_rois_checkbutton.config(state='disable')
            self.select_rois_button.config(state='disable')
            self.rois_file_name_label.config(state='disable')
        if switch_var == 1:
            self.select_reference_button.config(state='normal')
            self.reference_file_name_label.config(state='normal')
            self.import_rois_checkbutton.config(state='normal')
            self.select_rois_button.config(state='normal')
            self.rois_file_name_label.config(state='normal')

    def _import_rois_switcher(self):
        switch_var = self.import_rois_variable.get()
        if switch_var == 0:  # this means no rois selected
            self.select_rois_button.config(state='disable')
            self.rois_file_name_label.config(state='disable')
        if switch_var == 1 and self.import_reference_variable.get() == 1:
            self.select_rois_button.config(state='normal')
            self.rois_file_name_label.config(state='normal')

    def _import_data(self):
        # Remove the text on the Canvas
        self.import_data_text.set_visible(False)
        # Pop up a new window
        self.import_window = tk.Toplevel(self.master)
        self.import_window.title('Import...')
        self.import_window.geometry("700x850")
        # import_data_window.attributes('-topmost', True)

        # GLOBAL LAYOUT SETTINGS
        header_label_size = 12
        separator_symbol = '-'
        separator_count = 40

        # ==============================================================================================================
        # IMPORT DATA SETTINGS
        # --------------------------------------------------------------------------------------------------------------
        self.import_data_window = tk.Frame(self.import_window)
        self.import_data_window.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Label
        data_window_label = tk.Label(self.import_data_window, text='IMPORT DATA FILE', font=('Arial', header_label_size))
        data_window_label.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))

        # CheckButton if you want to import data?
        button_w = 32
        button_h = 16
        self.on_image = tk.PhotoImage(width=button_w, height=button_h)
        self.off_image = tk.PhotoImage(width=button_w, height=button_h)
        # (x_start, y_start, x_end, y_end), x start is left, y start is top
        self.on_image.put(("green",), to=(0, 0, button_w//2, button_h))
        self.off_image.put(("red",), to=(button_w//2, 0, button_w, button_h))

        self.import_data_variable = tk.IntVar(value=1)
        self.import_data_checkbutton = tk.Checkbutton(
            self.import_data_window, image=self.off_image, selectimage=self.on_image, indicatoron=False,
            onvalue=1, offvalue=0, variable=self.import_data_variable, command=self._import_data_switcher)

        self.import_data_checkbutton.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Add Open File Button
        # Set data type:
        data_file_extension = [('Recording Files', ['.txt', '.csv'])]
        self.select_recording_button = tk.Button(
            self.import_data_window, text="Open File ...", command=lambda: self._browse_file('data', data_file_extension))
        self.select_recording_button.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Add Label for Data File Name
        self.data_file_name_label = tk.Label(self.import_data_window, text="No File selected")
        self.data_file_name_label.grid(row=2, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Delimiter Settings
        delimiter_options = ['Comma', 'Tab', 'Semicolon', 'Space', 'Colon']
        self.data_delimiter_label = tk.Label(self.import_data_window, text="Delimiter: ")
        self.data_delimiter_label.grid(row=3, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))
        self.delimiter_data_variable = tk.StringVar(self.import_data_window)
        self.delimiter_data_variable.set(delimiter_options[0])  # default value
        self.delimiter_data_dropdown = tk.OptionMenu(self.import_data_window, self.delimiter_data_variable, *delimiter_options)
        self.delimiter_data_dropdown.grid(row=3, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))
        self.delimiter_data_dropdown.config(width=10, font=('Arial', 8))

        # Add Header Options
        self.data_has_header = tk.IntVar(value=0)
        self.data_has_header_button = tk.Checkbutton(
            self.import_data_window, text="File has Column Headers: ",
            variable=self.data_has_header, onvalue=0, offvalue=1)
        self.data_has_header_button.grid(row=4, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Frame Rate (Switches on when there is no stimulus file)
        self.frame_rate_label = tk.Label(self.import_data_window, text='Recording Sampling Rate [Hz]: ')
        self.frame_rate_label.grid(row=5, column=0)

        self.frame_rate_input = tk.Entry(self.import_data_window)
        self.frame_rate_input.grid(row=5, column=1)
        self.frame_rate_input.config(state='disabled')

        # Add Separator
        separator = tk.Label(self.import_window, text=[separator_symbol] * separator_count)
        separator.grid(row=1, column=0, padx=5, pady=10, sticky=(tk.N, tk.W))

        # ==============================================================================================================
        # IMPORT STIMULUS SETTINGS
        # --------------------------------------------------------------------------------------------------------------
        self.import_stimulus_window = tk.Frame(self.import_window)
        self.import_stimulus_window.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Label
        stimulus_window_label = tk.Label(self.import_stimulus_window, text='IMPORT STIMULUS FILE', font=('Arial', header_label_size))
        stimulus_window_label.grid(row=0, column=0, padx=5, pady=10, sticky=(tk.N, tk.W))

        # CheckButton if you want to import stimulus?
        self.import_stimulus_variable = tk.IntVar(value=1)
        self.import_stimulus_checkbutton = tk.Checkbutton(
            self.import_stimulus_window, image=self.off_image, selectimage=self.on_image, indicatoron=False,
            onvalue=1, offvalue=0, variable=self.import_stimulus_variable, command=self._switch_stimulus_import)
        self.import_stimulus_checkbutton.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Add Open File Button
        stimulus_file_extension = [('Stimulus Files', ['.txt', '.csv'])]
        self.select_stimulus_button = tk.Button(
            self.import_stimulus_window, text="Open File ...", command=lambda: self._browse_file('stimulus', stimulus_file_extension))
        self.select_stimulus_button.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Add Label for Data File Name
        self.stimulus_file_name_label = tk.Label(self.import_stimulus_window, text="No File selected")
        self.stimulus_file_name_label.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Delimiter Settings
        self.stimulus_delimiter_label = tk.Label(self.import_stimulus_window, text="Delimiter: ")
        self.stimulus_delimiter_label.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))
        self.delimiter_stimulus_variable = tk.StringVar(self.import_stimulus_window)
        self.delimiter_stimulus_variable.set(delimiter_options[0])  # default value
        self.delimiter_dropdown = tk.OptionMenu(self.import_stimulus_window, self.delimiter_stimulus_variable, *delimiter_options)
        self.delimiter_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))
        self.delimiter_dropdown.config(width=10, font=('Arial', 8))

        # Add Header Options
        self.stimulus_has_header = tk.IntVar(value=0)
        self.stimulus_has_header_button = tk.Checkbutton(
            self.import_stimulus_window, text="File has Column Headers: ",
            variable=self.stimulus_has_header, onvalue=0, offvalue=1)
        self.stimulus_has_header_button.grid(row=3, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Which column contains data?
        self.cols_label = tk.Label(self.import_stimulus_window, text="Column Data: ")
        self.cols_label.grid(row=4, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))
        self.cols = tk.Entry(self.import_stimulus_window)
        self.cols.insert(0, "2")
        self.cols.grid(row=4, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Is there a time axis? Else specify a sampling rate for the stimulus  trace
        self.cols_time_label = tk.Label(self.import_stimulus_window, text="Column Time Axis: ")
        self.cols_time_label.grid(row=5, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))
        self.cols_time = tk.Entry(self.import_stimulus_window)
        self.cols_time.insert(0, "1")
        self.cols_time.grid(row=5, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))

        button_w2 = 32
        button_h2 = 16
        self.on_image2 = tk.PhotoImage(width=button_w2, height=button_h2)
        self.off_image2 = tk.PhotoImage(width=button_w2, height=button_h2)
        # (x_start, y_start, x_end, y_end), x start is left, y start is top
        self.on_image2.put(("blue",), to=(0, 0, button_w2 // 2, button_h2))
        self.off_image2.put(("blue",), to=(button_w2 // 2, 0, button_w2, button_h2))
        self.stimulus_time_axis_checker = tk.IntVar(value=1)
        self.stimulus_time_axis_checkbutton = tk.Checkbutton(
            self.import_stimulus_window, image=self.off_image2, selectimage=self.on_image2, indicatoron=False,
            onvalue=1, offvalue=0, variable=self.stimulus_time_axis_checker, command=self._stimulus_sampling_rate_switcher)
        self.stimulus_time_axis_checkbutton.grid(row=5, column=2, padx=5, pady=5, sticky=(tk.N, tk.W))

        self.stimulus_frame_rate_label = tk.Label(self.import_stimulus_window, text='Sampling Rate [Hz]')
        self.stimulus_frame_rate_label.grid(row=5, column=3, padx=5, pady=10, sticky=(tk.N, tk.W))
        self.stimulus_frame_rate_input = tk.Entry(self.import_stimulus_window)
        self.stimulus_frame_rate_input.grid(row=5, column=4, padx=5, pady=10, sticky=(tk.N, tk.W))
        self.stimulus_frame_rate_label.config(state='disable')
        self.stimulus_frame_rate_input.config(state='disable')

        # Add Separator
        separator = tk.Label(self.import_window, text=[separator_symbol] * separator_count)
        separator.grid(row=3, column=0, padx=5, pady=10, sticky=(tk.N, tk.W))

        # ==============================================================================================================
        # IMPORT REFERENCE IMAGE
        # --------------------------------------------------------------------------------------------------------------
        self.import_reference_window = tk.Frame(self.import_window)
        self.import_reference_window.grid(row=4, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))
        # Label
        reference_window_label = tk.Label(self.import_reference_window, text='IMPORT REFERENCE IMAGE',
                                         font=('Arial', header_label_size))
        reference_window_label.grid(row=0, column=0, padx=5, pady=10, sticky=(tk.N, tk.W))

        # CheckButton if you want to import stimulus?
        self.import_reference_variable = tk.IntVar(value=1)
        self.import_reference_checkbutton = tk.Checkbutton(
            self.import_reference_window, image=self.off_image, selectimage=self.on_image, indicatoron=False,
            onvalue=1, offvalue=0, variable=self.import_reference_variable, command=self._import_reference_switch)
        self.import_reference_checkbutton.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Add Open File Button
        reference_file_extension = [('Reference Image', ['.jpg', '.jpeg', '.png', '.tif', '.tiff'])]
        self.select_reference_button = tk.Button(
            self.import_reference_window, text="Open File ...", command=lambda: self._browse_file('reference', reference_file_extension))
        self.select_reference_button.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Add Label for Data File Name
        self.reference_file_name_label = tk.Label(self.import_reference_window, text="No File selected")
        self.reference_file_name_label.grid(row=2, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Add Separator
        separator = tk.Label(self.import_window, text=[separator_symbol] * separator_count)
        separator.grid(row=5, column=0, padx=5, pady=10, sticky=(tk.N, tk.W))

        # ==============================================================================================================
        # IMPORT ROIs
        # --------------------------------------------------------------------------------------------------------------
        self.import_rois_window = tk.Frame(self.import_window)
        self.import_rois_window.grid(row=6, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))
        # Label
        rois_window_label = tk.Label(self.import_rois_window, text='IMPORT ROIS',
                                          font=('Arial', header_label_size))
        rois_window_label.grid(row=0, column=0, padx=5, pady=10, sticky=(tk.N, tk.W))

        # CheckButton if you want to import stimulus?
        self.import_rois_variable = tk.IntVar(value=1)
        self.import_rois_checkbutton = tk.Checkbutton(
            self.import_rois_window, image=self.off_image, selectimage=self.on_image, indicatoron=False,
            onvalue=1, offvalue=0, variable=self.import_rois_variable, command=self._import_rois_switcher)
        self.import_rois_checkbutton.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Add Open File Button
        rois_file_extension = [('ROIs', ['.zip'])]
        self.select_rois_button = tk.Button(
            self.import_rois_window, text="Open File ...",
            command=lambda: self._browse_file('rois', rois_file_extension))
        self.select_rois_button.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Add Label for Data File Name
        self.rois_file_name_label = tk.Label(self.import_rois_window, text="No File selected")
        self.rois_file_name_label.grid(row=2, column=1, padx=5, pady=5, sticky=(tk.N, tk.W))

        # Add Separator
        separator = tk.Label(self.import_window, text=[separator_symbol] * separator_count)
        separator.grid(row=7, column=0, padx=5, pady=10, sticky=(tk.N, tk.W))

        self.import_done_window = tk.Frame(self.import_window)
        self.import_done_window.grid(row=8, column=0, padx=5, pady=5, sticky=(tk.N, tk.W))
        import_data_button = tk.Button(self.import_done_window, text='Import', height=1, width=5, command=self._collect_import_data)
        import_data_button.grid(row=0, column=0, padx=5, pady=10, sticky=(tk.N, tk.W, tk.E, tk.S))
        cancel_button = tk.Button(self.import_done_window, text='Cancel', height=1, width=5, command=self.import_window.destroy)
        cancel_button.grid(row=0, column=1, padx=5, pady=10, sticky=(tk.N, tk.W, tk.E, tk.S))

    def pop_up_error(self, tlt, msg):
        tk.messagebox.showerror(title=tlt, message=msg)
        self.window_dummy.lift()

    def _collect_import_data(self):
        # All inputs are strings!
        # Data Info
        print('IMPORTING ...')
        stimulus_selected = self.import_stimulus_variable.get()
        data_selected = self.import_data_variable.get()
        reference_selected = self.import_reference_variable.get()
        if reference_selected:
            rois_selected = self.import_rois_variable.get()
        else:
            rois_selected = False
        self.window_dummy = self.import_window

        # ==============================================================================================================
        # First Check Stimulus File
        # --------------------------------------------------------------------------------------------------------------
        if stimulus_selected:  # this means user has selected a stimulus file
            # Check if you can find this file
            file_found = os.path.exists(self.browser.stimulus_file)
            if not file_found:
                self.pop_up_error('ERROR', 'Could not find stimulus file')
                return "break"
            # Now try to import the stimulus file
            # Check settings
            stimulus_delimiter = self.convert_to_delimiter(self.delimiter_stimulus_variable.get())
            data_column_number = self.cols.get()
            time_column_number = self.cols_time.get()
            header_button_state = self.stimulus_has_header.get()
            if header_button_state == 0:
                stimulus_header = 0
            else:
                stimulus_header = None
            # Check if text entries are correct
            try:
                data_column_number = int(data_column_number) - 1
            except ValueError:
                self.pop_up_error('ERROR', 'Stimulus Data Column Number must be Integer')
                return "break"
            # Now finally import the stimulus file
            stimulus_file = pd.read_csv(self.browser.stimulus_file, delimiter=stimulus_delimiter, header=stimulus_header)
            self.stimulus['Volt'] = stimulus_file.iloc[:, data_column_number]

            # Check for Time Axis Data
            if self.stimulus_time_axis_checker.get() == 1:
                try:
                    time_column_number = int(time_column_number) - 1
                except ValueError:
                    self.pop_up_error('ERROR', 'Time Axis Column Number must be Integer')
                    return "break"

                if time_column_number == data_column_number:
                    self.pop_up_error('ERROR', 'Stimulus Data Column and Time Axis Column cannot be the same')
                    return "break"
                self.stimulus['Time'] = stimulus_file.iloc[:, time_column_number]
            else:
                stimulus_fr = self.stimulus_frame_rate_input.get()
                s_size = len(self.stimulus['Volt'])
                self.stimulus['Time'] = np.linspace(0, s_size * stimulus_fr, s_size)

        # ==============================================================================================================
        # Second Check Recording Data File
        # --------------------------------------------------------------------------------------------------------------
        if data_selected == 1:
            # Check if you can find this file
            file_found = os.path.exists(self.browser.data_file)
            if not file_found:
                self.pop_up_error('ERROR', 'Could not find stimulus file')
                return "break"
            # Now try to import the recording data file
            # Check settings
            rec_delimiter = self.convert_to_delimiter(self.delimiter_data_variable.get())
            rec_header_button_state = self.data_has_header.get()
            if rec_header_button_state == 0:
                # Take the first row as headers
                rec_header = 0
            else:
                # Create new headers after importing the file
                rec_header = None
            # Now Import Data File
            self.data_raw = pd.read_csv(self.browser.data_file, delimiter=rec_delimiter, header=rec_header)
            # Create new headers if there are none
            if rec_header is None:
                header_labels = []
                for kk in range(self.data_raw.shape[1]):
                    header_labels.append(f'roi_{kk + 1}')
                self.data_raw.columns = header_labels

            # Get ROIs from columns
            self.data_rois = self.data_raw.keys()
            self.data_id = 0

            # Convert to delta f over f
            self.data_df = self.convert_raw_to_df_f(self.data_raw)

            # Compute z-scores
            self.data_z = self.compute_z_score(self.data_df)

            # Frame Rate
            # Now if there is a stimulus file with a time axis, estimate recording frame rate based on that.
            # If there is no time axis, then user had to specify a sampling rate.
            if self.stimulus_time_axis_checker.get() == 1:
                # There is a time axis, so use the maximum time to estimate the recordings frame rate
                self.data_fr = self.estimate_sampling_rate(
                     data=self.data_raw, f_stimulation=self.stimulus, print_msg=True)
            else:
                self.data_fr = self.frame_rate_input.get()
            # Compute Time Axis for Recording
            data_dummy = self.data_raw[self.data_rois[self.data_id]]
            self.data_time = np.linspace(0, len(data_dummy) / self.data_fr, len(data_dummy))

            # Fill List with rois
            # First clear listbox
            self.listbox.delete(0, tk.END)
            rois_numbers = np.linspace(1, len(self.data_rois.to_list()), len(self.data_rois.to_list()), dtype='int')
            rois_numbers = np.char.mod('%d', rois_numbers)
            self.listbox.insert("end", *rois_numbers)
            self.listbox.bind('<<ListboxSelect>>', self.list_items_selected)

        # ==============================================================================================================
        # Third check reference image and imagej rois files
        # --------------------------------------------------------------------------------------------------------------
        # Reference Image and ROIs Info
        if reference_selected == 1:
            self.ref_img = plt.imread(self.browser.reference_image_file)
            if rois_selected == 1:
                self.rois_in_ref = read_roi_zip(self.browser.roi_file)
                self._compute_ref_images()

        # ==============================================================================================================
        # Now check the imported files for consistency
        # --------------------------------------------------------------------------------------------------------------
        # Check if stimulus and recording data have the same duration
        if stimulus_selected and data_selected:
            if np.round(self.stimulus['Time'].max(), 2) == np.round(self.data_time.max(), 2):
                print('Recording Data and Stimulus Duration match!')
            else:
                do_continue = tk.messagebox.askyesno(
                    title='WARNING',
                    message='Recording Data Duration and Stimulus Duration do NOT match\n Do you want to continue?!')
                self.window_dummy.lift()
                if not do_continue:
                    self.window_dummy.lift()
                    return "break"

        # Check if the number of rois in the imagej rois is the same as the number of rois in the recording data
        if data_selected and rois_selected and reference_selected:
            if len(self.data_rois) == len(self.rois_in_ref):
                print('Rois found in Data Recording and in Imagej file do match!')
            else:
                do_continue = tk.messagebox.askyesno(
                    title='WARNING',
                    message='Rois found in Data Recording and in Imagej file do NOT match!\n Do you want to continue?!')
                self.window_dummy.lift()
                if not do_continue:
                    self.window_dummy.lift()
                    return "break"

        # ==============================================================================================================
        # Now Update The Figure and close the Import Window
        # --------------------------------------------------------------------------------------------------------------
        if stimulus_selected:
            self.initialize_stimulus_plot()
        if data_selected:
            self.initialize_recording_plot()
        if rois_selected:
            self.ref_img_obj = self.axs_ref.imshow(self.roi_images[self.data_id])
            self.ref_img_text = []
            for i, v in enumerate(self.roi_pos):
                if i == self.data_id:
                    color = (1, 0, 0)
                else:
                    color = (1, 1, 1)
                self.ref_img_text.append(self.axs_ref.text(
                    v[0], v[1], f'{i + 1}', fontsize=10, color=color,
                    horizontalalignment='center', verticalalignment='center'
                ))

            self.axs_ref.axis('off')
            self.axs_ref.set_visible(True)
            self.canvas_ref.draw()
        else:
            if reference_selected:
                self.ref_img_obj = self.axs_ref.imshow(self.ref_img)
                self.axs_ref.axis('off')
                self.axs_ref.set_visible(True)
                self.canvas_ref.draw()

        self.new_data_loaded = True
        self.import_window.destroy()

    def _quit(self, event):
        self.master.quit()
        self.master.destroy()

    def _exit_app(self):
        self.master.quit()
        self.master.destroy()

    def draw_rois_zipped(self, img, rois_dic, active_roi, r_color, alp, thickness=1):
        """
            It will draw the ROI as transparent window with border
            Args:
                img   : Image which you want to draw the ROI
                rois_dic: dictionary of rois from zipped rois from ImageJ
                active_roi: Active roi that will be highlighted
                r_color : in RGB (R, G, B), Black = (0, 0, 0)
                alp   : Alpha or transparency value between 0.0 and 1.0
                thickness: thickness in px

            Returns:
                Retuns the processed image
                out = drawROI()
        """

        overlay = img.copy()
        image_out = img.copy()

        if self.dict_depth(rois_dic) == 1:
            print('SINGLE ROI')
            roi_type = rois_dic['type']
            # convert top, left to center coordinates
            x_coordinate_centered = rois_dic['left'] + rois_dic['width'] // 2
            y_coordinate_centered = rois_dic['top'] + rois_dic['height'] // 2
            center_coordinates = (x_coordinate_centered, y_coordinate_centered)

            # Convert total to half-width for plotting ellipse
            axes_length = (rois_dic['width'] // 2, rois_dic['height'] // 2)

            # Set rest of the parameters for ellipse
            angle = 0
            start_ngle = 0
            end_angle = 360

            # Using cv2.ellipse() method
            # Draw a ellipse with line borders
            cv2.ellipse(overlay, center_coordinates, axes_length,
                        angle, start_ngle, end_angle, r_color, thickness)

            # add ellipse overlay to image
            cv2.addWeighted(overlay, alp, image_out, 1 - alp, 0, image_out)
        else:
            for key in rois_dic:
                if key != active_roi['name']:  # ignore active roi
                    # get roi shape type
                    roi_type = rois_dic[key]['type']
                    if roi_type == 'freehand':
                        x_vals = np.array(rois_dic[key]['x']).astype('int')
                        y_vals = np.array(rois_dic[key]['y']).astype('int')
                        for kk in range(len(x_vals) - 1):
                            cv2.line(overlay, (x_vals[kk], y_vals[kk]), (x_vals[kk + 1], y_vals[kk + 1]),
                                     r_color, thickness, lineType=cv2.LINE_AA)
                    elif roi_type == 'rectangle':
                        x_start_pos = int(rois_dic[key]['left'])
                        y_start_pos = int(rois_dic[key]['top'])
                        x_end_pos = x_start_pos + int(rois_dic[key]['width'])
                        y_end_pos = y_start_pos + int(rois_dic[key]['height'])
                        cv2.rectangle(overlay, (x_start_pos, y_start_pos), (x_end_pos, y_end_pos), r_color, thickness)

                    else:
                        # convert top, left to center coordinates
                        x_coordinate_centered = int(rois_dic[key]['left'] + rois_dic[key]['width'] // 2)
                        y_coordinate_centered = int(rois_dic[key]['top'] + rois_dic[key]['height'] // 2)
                        center_coordinates = (x_coordinate_centered, y_coordinate_centered)

                        # Convert total to half-width for plotting ellipse
                        axes_length = (int(rois_dic[key]['width'] // 2), int(rois_dic[key]['height'] // 2))

                        # Set rest of the parameters for ellipse
                        angle = 0
                        start_angle = 0
                        end_angle = 360
                        # Using cv2.ellipse() method
                        # Draw a ellipse with line borders
                        cv2.ellipse(overlay, center_coordinates, axes_length,
                                    angle, start_angle, end_angle, r_color, thickness)

        # ACTIVE ROI
        # convert top, left to center coordinates
        active_roi_type = active_roi['type']
        if active_roi_type == 'freehand':
            x_vals = np.array(active_roi['x']).astype('int')
            y_vals = np.array(active_roi['y']).astype('int')
            r_color = (255, 0, 0)
            for kk in range(len(x_vals) - 1):
                cv2.line(overlay, (x_vals[kk], y_vals[kk]), (x_vals[kk + 1], y_vals[kk + 1]),
                         r_color, thickness, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, alp, image_out, 1 - alp, 0, image_out)
        elif active_roi_type == 'rectangle':
            x_start_pos = int(active_roi['left'])
            y_start_pos = int(active_roi['top'])
            x_end_pos = x_start_pos + int(active_roi['width'])
            y_end_pos = y_start_pos + int(active_roi['height'])
            r_color = (255, 0, 0)
            cv2.rectangle(overlay, (x_start_pos, y_start_pos), (x_end_pos, y_end_pos), r_color, thickness)
            # add overlays to image
            cv2.addWeighted(overlay, alp, image_out, 1 - alp, 0, image_out)
        else:
            x_coordinate_centered = active_roi['left'] + active_roi['width'] // 2
            y_coordinate_centered = active_roi['top'] + active_roi['height'] // 2
            center_coordinates = (x_coordinate_centered, y_coordinate_centered)

            # Convert total to half-width for plotting ellipse
            axes_length = (active_roi['width'] // 2, active_roi['height'] // 2)

            # Set rest of the parameters for ellipse
            angle = 0
            start_angle = 0
            end_angle = 360

            # Using cv2.ellipse() method
            # Draw a ellipse with line borders
            r_color = (255, 0, 0)
            cv2.ellipse(overlay, center_coordinates, axes_length,
                        angle, start_angle, end_angle, r_color, thickness)

            # add ellipse overlays to image
            cv2.addWeighted(overlay, alp, image_out, 1 - alp, 0, image_out)

        return image_out

    # STATIC METHODS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def do_nothing(event):
        return "break"

    @staticmethod
    def print_something(event):
        print('SOMETHING')
        return "break"

    @staticmethod
    def cirf_double_tau(tau1, tau2, a, dt):
        t_max = tau2 * 5  # in sec
        t_cif = np.arange(0, t_max, dt)
        cif = a * (1 - np.exp(-(t_cif / tau1))) * np.exp(-(t_cif / tau2))
        return cif, t_cif

    @staticmethod
    def convert_to_delimiter(d):
        if d == 'Comma':
            d_str = ','
        elif d == 'Tab':
            d_str = '\t'
        elif d == 'Semicolon':
            d_str = ';'
        elif d == 'Colon':
            d_str = ':'
        elif d == 'Space':
            d_str = '\s+'
        else:
            d_str = ','
        return d_str

    @staticmethod
    def convert_samples_to_time(sig, fr):
        t_out = np.linspace(0, len(sig) / fr, len(sig))
        return t_out

    @staticmethod
    def convert_raw_to_df_f(data):
        fbs = np.percentile(data, 5)
        return (data - fbs) / fbs

    @staticmethod
    def compute_z_score(data):
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    @staticmethod
    def estimate_sampling_rate(data, f_stimulation, print_msg):
        r""" Estimate the sampling rate via the total duration and sample count
            ----------
            data : pandas data frame, shape (N,)
                the values of all ROIs.
            f_stimulation : pandas data frame, shape (N,)
                stimulation recording (voltage trace and time trace).
            Returns
            -------
            fr : float
                the estimated sampling rate.
            Notes
            -----
            the stimulation data frame needs a column called 'Time' with sample time points.
        """
        if (type(data) is int) or (type(data) is float):
            data_size = data
        else:
            data_size = len(data)
        max_time = f_stimulation['Time'].max()
        fr = data_size / max_time
        if print_msg:
            print('')
            print('--------- INFO ---------')
            print(f'Estimated Frame Rate of Ca Imaging Recording: {fr} Hz')
            print('')

        return fr

    @staticmethod
    def import_stimulation_file(file_dir):
        # Automatic detection of delimiter and headers
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
            if len(data.keys()) > 2:
                data = data.drop(data.columns[0], axis=1)
            data.columns = ['Time', 'Volt']
            return data
        except ValueError:
            if len(data.keys()) > 2:
                data = data.drop(data.columns[0], axis=1)
            return data

    @staticmethod
    def import_f_raw(file_dir):
        # Automatic detection of delimiter and headers
        # Check for header and delimiter
        with open(file_dir) as csv_file:
            dialect = csv.Sniffer().sniff(csv_file.read(32))
            delimiter = dialect.delimiter

        # Load data and assume no header
        data = pd.read_csv(file_dir, decimal='.', sep=delimiter, header=None)
        if (data.iloc[:, 0].dtype == 'int64') or (data.iloc[:, 0].dtype == 'int32'):
            data = data.drop(columns=0)

        # Check for correct header
        is_string = isinstance(data.iloc[0, 0], str) # is the first entry a string or not?
        if is_string:
            # Load data once again, this time with the first row as header
            data = pd.read_csv(file_dir, decimal='.', sep=delimiter, header=0)
            # check if first column is data or index
            if (data.iloc[:, 0].dtype == 'int64') or (data.iloc[:, 0].dtype == 'int32'):
                data = data.drop(columns=0)
        else:
            # Since there is no header, create one
            header_labels = []
            for kk in range(data.shape[1]):
                header_labels.append(f'roi_{kk + 1}')
            data.columns = header_labels
        return data

    @staticmethod
    def dict_depth(dic):
        str_dic = str(dic)
        counter = 0
        for i in str_dic:
            if i == "{":
                counter += 1
        return counter


class VerticalNavigationToolbar2Tk(NavigationToolbar2Tk):
    def __init__(self, canvas, window):
        super().__init__(canvas, window, pack_toolbar=False)

    # override _Button() to re-pack the toolbar button in vertical direction
    def _Button(self, text, image_file, toggle, command):
        b = super()._Button(text, image_file, toggle, command)
        b.pack(side=tk.TOP) # re-pack button in vertical direction
        return b

    # override _Spacer() to create vertical separator
    def _Spacer(self):
        s = tk.Frame(self, width=26, relief=tk.RIDGE, bg="DarkGray", padx=2)
        s.pack(side=tk.TOP, pady=5) # pack in vertical direction
        return s

    # disable showing mouse position in toolbar
    def set_message(self, s):
        pass

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.remove_rubberband()
        height = self.canvas.figure.bbox.height
        y0 = height - y0
        y1 = height - y1
        # this is the bit we want to change...
        self.lastrect = self.canvas._tkcanvas.create_rectangle(x0, y0, x1, y1, outline='red')
        # hex color strings e.g.'#BEEFED' and named colors e.g. 'gainsboro' both work


class CustomNavigationToolbar2(NavigationToolbar2):
    def __init__(self, canvas):
        super().__init__(canvas)

    def back(self, *args):
        """
        Move back up the view lim stack.
        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
        pass
        # self._nav_stack.back()
        # self.set_history_buttons()
        # self._update_view()

    def forward(self, *args):
        """
        Move forward in the view lim stack.
        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
        pass
        # self._nav_stack.forward()
        # self.set_history_buttons()
        # self._update_view()


# https://github.com/matplotlib/matplotlib/blob/00ec4ec6321d85af8519e268d67807746207f7d2/lib/matplotlib/backends/_backend_tk.py#L369
class CustomNavigationToolbar2Tk(NavigationToolbar2Tk):
    def __init__(self, canvas, window):
        super().__init__(canvas, window, pack_toolbar=False)

    def set_history_buttons(self):
        pass
        # state_map = {True: tk.NORMAL, False: tk.DISABLED}
        # can_back = self._nav_stack._pos > 0
        # can_forward = self._nav_stack._pos < len(self._nav_stack._elements) - 1
        #
        # if "Back" in self._buttons:
        #     self._buttons['Back']['state'] = state_map[can_back]
        #
        # if "Forward" in self._buttons:
        #     self._buttons['Forward']['state'] = state_map[can_forward]

    def zoom(self, *args):
        super().zoom(*args)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.remove_rubberband()
        height = self.canvas.figure.bbox.height
        y0 = height - y0
        y1 = height - y1
        # this is the bit we want to change...
        self.lastrect = self.canvas._tkcanvas.create_rectangle(x0, y0, x1, y1, outline='red')
        # hex color strings e.g.'#BEEFED' and named colors e.g. 'gainsboro' both work


if __name__ == '__main__':
    root = tk.Tk()
    main_app = MainApplication(root)
    root.mainloop()
