import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from IPython import embed
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import Slider
import analysis_util_functions as uf
from read_roi import read_roi_zip
from tkinter import filedialog
from tkinter import messagebox
import sys


def data_viewer(f_rec_name, raw_data, sig_t, ref_im, st_rec, rec_protocol, rois_dic, good_scores, scores,
                cell_score_th, reg_trace, selected_cells):

    print('')
    print('---------- STARTING DATA VIEWER ----------')
    print('')

    # Get Stimulus Info
    st_on = rec_protocol['Onset_Time']
    st_off = rec_protocol['Offset_Time']
    cell_score_th = cell_score_th
    # Get Roi Names
    f_rois_all = raw_data.keys()
    # Normalize raw values to max = 1
    f_raw_norm = raw_data / raw_data.max()
    y_lim = 1.5
    y_lim_min = -0.1

    stimulus_reg = reg_trace
    reg_scores = scores
    good_scores = good_scores
    selected_cells = selected_cells

    # Select first data to plot: ROI 1
    sig, _, _ = uf.compute_df_over_f(
        f_values=raw_data,
        window_size=100,
        per=5,
        fast=True
    )
    sig = sig[f_rois_all[0]]
    # Create the initial figure ----------
    fig, ax = plt.subplots(2, 1)
    fig.canvas.manager.set_window_title(f'Recording: {f_rec_name}')

    re_im_obj = ax[0].imshow(ref_im)
    ax[0].axis('off')  # clear x-axis and y-axis

    # plot stimulus marker
    ax[1].plot(st_rec['Time'], (st_rec['Volt'] / st_rec['Volt'].max()) * y_lim, color='darkgreen', lw=2, alpha=0.3)
    marker_color = 'g'
    stim_text_obj = {}
    for k in range(len(st_on)):
        ax[1].fill_between([st_on[k], st_off[k]], [y_lim, y_lim], color=marker_color,  edgecolor=marker_color, alpha=0.3)

    uf.remove_axis(ax[1], box=False)
    # Plot Stimulus Trace
    ax[0].set_title(f'ROI: {1}')

    # Plot Response Traces
    l, = ax[1].plot(sig_t, sig, color='k')
    ax[1].set_xlabel('Time [s]')
    # ax[1].set_ylabel('dF/F')
    ax[1].set_ylabel('Raw Values')
    ax[1].set_ylim([y_lim_min, y_lim])

    # Plot Reg Trace
    fr_rec = uf.estimate_sampling_rate(data=raw_data, f_stimulation=st_rec, print_msg=False)
    t_cirf = uf.convert_samples_to_time(stimulus_reg, fr_rec)
    ax[1].plot(t_cirf, stimulus_reg, 'b', alpha=0.3)

    text_obj = ax[0].text(
        0, 0, '', fontsize=12, color=(1, 0, 0),
        horizontalalignment='center', verticalalignment='center'
    )

    text_score_obj_x = -(np.max(sig_t) // 4)
    text_score_obj_y = 0

    text_score_obj = ax[1].text(
        text_score_obj_x, text_score_obj_y, 'scores: ', fontsize=12, color=(0, 0, 0),
        horizontalalignment='center', verticalalignment='center'
    )

    # Set Window to Full Screen
    manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    manager.window.showMaximized()
    fig.subplots_adjust(left=0.2, top=0.9, bottom=0.2, right=0.9, wspace=0.1, hspace=0.25)

    class App:
        # Initialize parameter values
        if selected_cells.shape[0] <= 0:
            good_cells = good_scores
        else:
            good_cells = selected_cells
        show_raw = False
        switch_to_good_cells_status = False
        cells_by_score = good_scores
        th_reg = cell_score_th
        ind = 0
        df_win = 100
        df_per = 5
        data, _, _ = uf.compute_df_over_f(
                    f_values=raw_data,
                    window_size=df_win,
                    per=df_per,
                    fast=True
                    )
        y_lim = np.max(data.max())
        f_rois = f_rois_all
        selected_roi_nr = 0
        selected_roi = f_rois[selected_roi_nr]
        cell_color = 'red'

        # Compute Ref Image
        active_roi_nr = 1
        roi_images = []
        roi_pos = []
        for ii, active_roi in enumerate(rois_dic):
            roi_pos.append((rois_dic[active_roi]['left'] + rois_dic[active_roi]['width'] // 2,
                            rois_dic[active_roi]['top'] + rois_dic[active_roi]['height'] // 2))
            roi_images.append(uf.draw_rois_zipped(
                img=ref_im, rois_dic=rois_dic, active_roi=rois_dic[active_roi],
                r_color=(0, 0, 255), alp=0.5, thickness=1)
            )

        def turn_page(self):
            # Select new ROI Number
            self.selected_roi = self.f_rois[self.ind]
            self.selected_roi_nr = int(self.f_rois[self.ind][4:])

            # Update Data Display
            ydata = self.data[self.selected_roi]
            l.set_ydata(ydata)

            # Update title
            ax[0].set_title(f'ROI: {self.selected_roi_nr} ({self.ind+1} / {len(self.f_rois)})', color='black')

            # Update Ref Image
            re_im_obj.set_data(self.roi_images[self.selected_roi_nr - 1])

            # Update Ref Image Label
            text_obj.set_text(f'{self.selected_roi_nr}')
            text_obj.set_position(self.roi_pos[self.selected_roi_nr - 1])

            # Update Score Text
            if any(reg_scores[self.selected_roi] >= self.th_reg):
                text_score_obj.set_color((1, 0, 0))
            else:
                text_score_obj.set_color((0, 0, 0))
            score_text = reg_scores[['type', 'parameter', self.selected_roi]].round(2).to_string(header=None,
                                                                                                 index=None)
            text_score_obj.set_text(f'Reg. Scores \n{score_text}')

            # Set the line color of the last roi to red, just for visualization
            if self.ind + 1 == len(self.f_rois):
                l.set_color('red')
            else:
                l.set_color('black')

            if self.selected_roi in self.good_cells['roi'].to_numpy():
                # button name
                b_good.label.set_text("REMOVE")

            else:
                b_good.label.set_text("ADD")

            plt.draw()

        def next(self, event):
            # Increase Index by 1
            self.ind += 1
            self.ind = self.ind % self.data.shape[1]
            self.turn_page()

        def prev(self, event):
            # Decrease Index by 1
            self.ind -= 1
            self.ind = self.ind % self.data.shape[1]
            self.turn_page()

        def select_cell(self, event):
            if self.selected_roi in self.good_cells['roi'].to_numpy():  # this is a good cell
                # REMOVE CELL FROM GOOD CELLS
                idx = self.good_cells[self.good_cells['roi'] == self.selected_roi].index[0]
                self.good_cells = self.good_cells.drop(idx)
                b_good.label.set_text("BAD")
            else:
                # ADD CELL TO GOOD CELLS
                self.good_cells = pd.concat([self.good_cells, pd.DataFrame([self.selected_roi], columns=['roi'])], ignore_index=True)
                b_good.label.set_text("GOOD")

        def update_fbs_win_slider(self, val):
            # Get new parameter value
            self.df_win = int(val)
            # Compute data based on new parameter value
            self.compute_data()
            # Update y-lim
            self.y_lim = np.max(self.data.max())
            ax[1].set_ylim([y_lim_min, self.y_lim])
            l.set_ydata(self.data[self.f_rois[self.ind]])
            plt.draw()

        def update_fbs_per_slider(self, val):
            self.df_per = int(val)
            # Compute data based on new parameter value
            self.compute_data()
            # Update y-lim
            self.y_lim = np.max(self.data.max())
            ax[1].set_ylim([y_lim_min, self.y_lim])
            l.set_ydata(self.data[self.f_rois[self.ind]])
            plt.draw()

        def raw(self, event):
            self.show_raw = np.invert(self.show_raw)
            self.compute_data()
            # Update y-lim
            self.y_lim = np.max(self.data.max())
            ax[1].set_ylim([y_lim_min, self.y_lim])
            l.set_ydata(self.data[self.f_rois[self.ind]])

            if self.show_raw:
                b_raw.label.set_text("Show dF")
            else:
                b_raw.label.set_text("Show RAW")
            plt.draw()

        def export_good_cells(self, event):
            f = filedialog.asksaveasfile(mode='w', defaultextension=".csv", filetypes=[('csv files', '*.csv')],
                                         initialfile=f"{f_rec_name}_selected_cells")
            if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
                return
            self.good_cells.to_csv(f.name)
            f.close()
            messagebox.showinfo('INFO', 'Selected Cells stored to HDD')

        def compute_data(self):
            if self.show_raw:
                ax[1].set_ylabel('Raw Values')
                # self.data = uf.filter_raw_data(f_raw_norm, win=self.filter_win, o=self.filter_order)
                self.data = f_raw_norm
            else:
                ax[1].set_ylabel('dF/F')
                self.data, _, _ = uf.compute_df_over_f(
                    f_values=raw_data,
                    window_size=self.df_win,
                    per=self.df_per,
                    fast=True
                    )

    callback = App()

    # [left, bottom, width, height]

    # FBS Percentile Slider
    fbs_per_slider_ax = fig.add_axes([0.08, 0.4, 0.02, 0.4])
    fbs_per_slider = Slider(
        fbs_per_slider_ax,
        'fbs percentile [%]',
        valmin=0,
        valmax=50,
        valinit=5,
        valstep=1,
        orientation="vertical",
        color='black'
    )
    fbs_per_slider.on_changed(callback.update_fbs_per_slider)

    # FBS Window Size Slider
    # Create a plt.axes object to hold the slider
    fbs_slider_ax = fig.add_axes([0.05, 0.4, 0.02, 0.4])
    # Add a slider to the plt.axes object
    fbs_slider = Slider(
        fbs_slider_ax,
        'fbs win [s]',
        valmin=0,
        valmax=int(np.floor(sig_t[-1])/2),
        valinit=100,
        valstep=10,
        orientation="vertical",
        color='black'
    )
    fbs_slider.on_changed(callback.update_fbs_win_slider)

    # Add Buttons to Figure
    button_width = 0.05
    button_height = 0.03
    button_font_size = 6

    ax_next = fig.add_axes([0.5, 0.05, button_width, button_height])
    b_next = Button(ax_next, 'Next')
    b_next.label.set_fontsize(button_font_size)
    b_next.on_clicked(callback.next)

    ax_prev = fig.add_axes([0.4, 0.05, button_width, button_height])
    b_prev = Button(ax_prev, 'Previous')
    b_prev.label.set_fontsize(button_font_size)
    b_prev.on_clicked(callback.prev)

    ax_raw = fig.add_axes([0.3, 0.05, button_width, button_height])
    b_raw = Button(ax_raw, 'Show RAW')
    b_raw.label.set_fontsize(button_font_size)
    b_raw.on_clicked(callback.raw)

    ax_good = fig.add_axes([0.2, 0.05, button_width, button_height])
    b_good = Button(ax_good, 'Show RAW')
    b_good.label.set_fontsize(button_font_size)
    b_good.on_clicked(callback.select_cell)

    ax_export = fig.add_axes([0.1, 0.05, button_width, button_height])
    b_export = Button(ax_export, 'EXPORT')
    b_export.label.set_fontsize(button_font_size)
    b_export.on_clicked(callback.export_good_cells)

    plt.show()


if __name__ == '__main__':
    show_selection = False
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == '-good':
            show_selection = True

    # Select Directory and get all files
    file_name = uf.select_file([('Recording Files', 'raw.txt')])
    data_file_name = os.path.split(file_name)[1]
    rec_dir = os.path.split(file_name)[0]
    rec_name = os.path.split(rec_dir)[1]
    # rec_name = os.path.split(file_name)[1][0:uf.find_pos_of_char_in_string(os.path.split(file_name)[1], '_')[-1]]
    uf.msg_box(rec_name, f'SELECTED RECORDING: {rec_name}', '+')

    # Import stimulation trace
    stimulus = uf.import_txt_stimulation_file(f'{rec_dir}', f'{rec_name}_stimulation', float_dec='.')

    # Import Protocol
    protocol = pd.read_csv(f'{rec_dir}/{rec_name}_protocol.csv')

    # Import Reference Image
    img_ref = plt.imread(f'{rec_dir}/refs/{rec_name}_ROI.tif.jpg', format='jpg')

    # Import ROIS from Imagej
    rois_in_ref = read_roi_zip(f'{rec_dir}/refs/{rec_name}_ROI.tif_RoiSet.zip')

    # Import raw fluorescence traces (rois)
    # It is important that the header is equal to the correct ROI number
    header_labels = []
    for k, v in enumerate(rois_in_ref):
        header_labels.append(f'roi_{k + 1}')
    f_raw = pd.read_csv(f'{rec_dir}/{data_file_name}', decimal='.', sep='\t', index_col=0, header=None)
    f_raw.columns = header_labels

    # Estimate frame rate
    fr_rec = uf.estimate_sampling_rate(data=f_raw, f_stimulation=stimulus, print_msg=False)

    # Compute Calcium Impulse Response Function (CIRF)
    cirf = uf.create_cif(fr_rec, tau=10)

    # Compute time axis for rois
    roi_time_axis = uf.convert_samples_to_time(sig=f_raw, fr=fr_rec)

    # Get step and ramp stimuli
    step_parameters = protocol[protocol['Stimulus'] == 'Step']['Duration'].unique()
    ramp_parameters = protocol[protocol['Stimulus'] == 'Ramp']['Duration'].unique()

    stimulus_parameters = pd.DataFrame()
    stimulus_parameters['parameter'] = np.append(step_parameters, ramp_parameters)
    stimulus_parameters['type'] = np.append(['Step'] * len(step_parameters), ['Ramp'] * len(ramp_parameters))

    # Import Linear Regression Scoring
    good_cells_by_score_csv = pd.read_csv(f'{rec_dir}/{rec_name}_lm_good_score_rois.csv', index_col=0)
    final_mean_score = pd.read_csv(f'{rec_dir}/{rec_name}_lm_mean_scores.csv')
    all_cells = np.load(f'{rec_dir}/{rec_name}_lm_results.npy', allow_pickle=True).item()
    score_th = 0.15

    # Compute Regressor for the entire stimulus trace for plotting
    binary, reg, _, _ = uf.create_binary_trace(
        sig=f_raw, cirf=cirf, start=protocol['Onset_Time'], end=protocol['Offset_Time'],
        fr=fr_rec, ca_delay=0, pad_before=5, pad_after=20, low=0, high=1
    )

    reg = reg / np.max(reg)

    # Look for selected cells
    r_l = os.listdir(rec_dir)
    selected_cells_file_name = [s for s in r_l if 'selected_cells' in s]

    if show_selection:
        if selected_cells_file_name:
            selected_cells = pd.read_csv(f'{rec_dir}/{selected_cells_file_name[0]}', index_col=0)
            idx = selected_cells.transpose().to_numpy()[0]
            f_raw = f_raw[idx]
            print('FOUND SELECTED FILES')
        else:
            idx = good_cells_by_score_csv.transpose().to_numpy()[0]
            f_raw = f_raw[idx]

    # Call DataViewer
    data_viewer(f_rec_name=rec_name, raw_data=f_raw, sig_t=roi_time_axis, ref_im=img_ref, st_rec=stimulus,
                rec_protocol=protocol, rois_dic=rois_in_ref, good_scores=good_cells_by_score_csv,
                scores=final_mean_score, cell_score_th=score_th, reg_trace=reg, selected_cells=pd.DataFrame())


