import numpy as np
import pandas as pd
import customtkinter as ctk
import os
from natsort import natsorted
import openpyxl

def compute_vectors(csv_path):
    df = pd.read_csv(csv_path, header=None)
    points = df.to_numpy()
    vectors = []
    for i in range(len(points) - 1):
        x_i, y_i = points[i]
        x_next, y_next = points[i + 1]
        dx = x_next - x_i
        dy = y_next - y_i
        vectors.append([x_i, y_i, dx, dy])
    return np.array(vectors)

def compute_resultant_vector(vectors):
    resultant_x = np.sum(vectors[:, 2])
    resultant_y = np.sum(vectors[:, 3])
    return np.array([resultant_x, resultant_y])

def compute_magnitude(vectors):
    return np.sqrt(vectors[:, 2]**2 + vectors[:, 3]**2)

def compute_angle(vectors):
    return np.rad2deg(np.arctan2(vectors[:, 3], vectors[:, 2]))

def get_batch_mode():
    app = ctk.CTk()
    app.title("Select Processing Mode")
    batch_mode = ctk.BooleanVar(value=False)

    def on_toggle():
        batch_mode.set(not batch_mode.get())

    toggle_btn = ctk.CTkButton(app, text="Toggle Batch Mode", command=on_toggle)
    toggle_btn.pack(pady=10)

    def on_confirm():
        app.quit()

    confirm_btn = ctk.CTkButton(app, text="Confirm", command=on_confirm)
    confirm_btn.pack(pady=10)

    app.mainloop()
    app.destroy()
    return batch_mode.get()

def batch_process(csv_files):
    resultant_vectors = []
    average_magnitudes = []
    euclidean_distances = []
    path_lengths = []
    angles = []
    vectors = []

    for csv_file in natsorted(csv_files):
        vector = compute_vectors(csv_file)
        euclidean_distance = np.linalg.norm(vector[:, 2:4], axis=1)
        magnitudes = compute_magnitude(vector)
        angle = compute_angle(vector)
        resultant = compute_resultant_vector(vector)
        file_name = os.path.basename(csv_file)
        resultant_vectors.append((file_name, resultant))
        average_magnitudes.append((file_name, np.mean(magnitudes)))
        euclidean_distances.append((file_name, euclidean_distance))
        path_length = np.sum(euclidean_distance)
        path_lengths.append((file_name, path_length))
        angles.append((file_name, angle))
        vectors.append((file_name, vector))

    mean_resultant = np.mean([r[1] for r in resultant_vectors], axis=0)
    mean_magnitude = np.mean([m[1] for m in average_magnitudes])
    mean_path_length = np.mean([p[1] for p in path_lengths])

    return (
        resultant_vectors,
        average_magnitudes,
        mean_resultant,
        mean_magnitude,
        euclidean_distances,
        mean_path_length,
        angles,
        vectors,
        path_lengths
    )

def single_process(csv_path):
    if not csv_path:
        print("No file selected.")
        return
    vectors = compute_vectors(csv_path)
    magnitudes = compute_magnitude(vectors)
    angles = compute_angle(vectors)
    resultant = compute_resultant_vector(vectors)
    average_magnitude = np.mean(magnitudes)
    euclidean_distance = np.linalg.norm(vectors[:, 2:4], axis=1)
    path_length = np.sum(euclidean_distance)
    file_name = os.path.basename(csv_path)

    return (
        (file_name, vectors),
        (file_name, magnitudes),
        (file_name, angles),
        (file_name, resultant),
        (file_name, average_magnitude),
        (file_name, euclidean_distance),
        (file_name, path_length)
    )

def export_to_excel(batch_mode):
    if batch_mode:
        csv_dir = ctk.filedialog.askdirectory(title="Select Directory with CSV Files")
        if not csv_dir:
            print("No directory selected.")
            return
        os.chdir(csv_dir)
        csv_files = os.listdir(csv_dir)
        csv_files = [os.path.normpath(os.path.join(csv_dir, f)) for f in csv_files if f.endswith('.csv')]
        if not csv_files:
            print("No CSV files found in directory.")
            return
        (resultant_vectors, average_magnitudes, mean_resultant, mean_magnitude,
         euclidean_distances, mean_path_length, angles, vectors, path_lengths) = batch_process(csv_files=csv_files)
        print("Exporting batch results to Excel...")

        # Batch summary file
        summary_data = {
            'mean_resultant_x': [mean_resultant[0]],
            'mean_resultant_y': [mean_resultant[1]],
            'mean_magnitude': [mean_magnitude],
            'mean_path_length': [mean_path_length]
        }
        df_summary = pd.DataFrame(summary_data)
        df_path_lengths = pd.DataFrame(path_lengths, columns=['file', 'path_length'])

        with pd.ExcelWriter("batch_summary.xlsx", engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name="Summary", index=False, float_format="%.3f")
        print("Batch summary exported to batch_summary.xlsx")

        wb = openpyxl.load_workbook(output_filename_summary := "batch_summary.xlsx")
        for ws in wb.worksheets:
            for col in ws.columns:
                max_length = 0
                col_letter = openpyxl.utils.get_column_letter(col[0].column)
                for cell in col:
                    try:
                        cell_value = str(cell.value)
                        if len(cell_value) > max_length:
                            max_length = len(cell_value)
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = max_length + 10
        wb.save(output_filename_summary)
        print(f"Exported {output_filename_summary}")

        # Per-file Excel export
        for i, (fname, vec) in enumerate(vectors):
            xy = vec[:, :2]
            dxdy = vec[:, 2:]
            df_points = pd.read_csv(os.path.join(csv_dir, fname), header=None)
            last_xy = df_points.iloc[[-1]].to_numpy()
            all_xy = np.vstack([xy, last_xy])
            all_dxdy = np.vstack([dxdy, [[np.nan, np.nan]]])
            df_vectors = pd.DataFrame(np.hstack([all_xy, all_dxdy]), columns=['x', 'y', 'dx', 'dy'])
            df_vectors.insert(0, 'file', fname)

            # Find corresponding stats
            mag = [m for f, m in average_magnitudes if f == fname][0]
            ang = [a for f, a in angles if f == fname][0]
            res = [r for f, r in resultant_vectors if f == fname][0]
            eucl = [e for f, e in euclidean_distances if f == fname][0]
            path = [p for f, p in path_lengths if f == fname][0]

            # Find all magnitudes for this file
            all_mags = compute_magnitude(vec)
            df_magnitudes = pd.DataFrame({'file': [fname]*len(all_mags), 'magnitude': all_mags})

            df_angles = pd.DataFrame({'file': fname, 'angle': ang})
            df_resultant = pd.DataFrame([{'file': fname, 'resultant_x': res[0], 'resultant_y': res[1]}])
            df_avg_mag = pd.DataFrame({'file': [fname], 'average_magnitude': [mag]})
            df_path = pd.DataFrame({'file': [fname], 'total_path_length': [path]})
            df_eucl = pd.DataFrame({'file': fname, 'euclidean_distance': eucl})

            output_filename = f"{os.path.splitext(fname)[0]}_results.xlsx"
            with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                df_resultant.to_excel(writer, sheet_name="Resultant", index=False, float_format="%.3f")
                df_avg_mag.to_excel(writer, sheet_name="AverageMagnitude", index=False, float_format="%.3f")
                df_eucl.to_excel(writer, sheet_name="EuclideanDistances", index=False, float_format="%.3f")
                df_angles.to_excel(writer, sheet_name="Angles", index=False, float_format="%.3f")
                df_vectors.to_excel(writer, sheet_name="Vectors", index=False, float_format="%.3f")
                df_magnitudes.to_excel(writer, sheet_name="Magnitudes", index=False, float_format="%.3f")
                df_path.to_excel(writer, sheet_name="PathLengths", index=False, float_format="%.3f")

            # Auto-fit columns for all sheets
            wb = openpyxl.load_workbook(output_filename)
            for ws in wb.worksheets:
                for col in ws.columns:
                    max_length = 0
                    col_letter = openpyxl.utils.get_column_letter(col[0].column)
                    for cell in col:
                        try:
                            cell_value = str(cell.value)
                            if len(cell_value) > max_length:
                                max_length = len(cell_value)
                        except Exception:
                            pass
                    ws.column_dimensions[col_letter].width = max_length + 10
            wb.save(output_filename)
            print(f"Exported {output_filename}")

    else:
        csv_path = ctk.filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
        os.chdir(os.path.dirname(csv_path))
        if not csv_path:
            print("No file selected.")
            return
        single_results = single_process(csv_path=csv_path)
        if single_results is None:
            print("No results to export.")
            return
        vectors, magnitudes, angles, resultant, average_magnitude, euclidean_distance, path_length = single_results
        vectors = [vectors]
        magnitudes = [magnitudes]
        angles = [angles]
        resultant_vectors = [resultant]
        average_magnitudes = [average_magnitude]
        euclidean_distances = [euclidean_distance]
        path_lengths = [path_length]

        # Vectors: shape (N, 4)
        xy = vectors[0][1][:, :2]
        dxdy = vectors[0][1][:, 2:]

        # Get last x, y from original CSV
        df_points = pd.read_csv(csv_path, header=None)
        last_xy = df_points.iloc[[-1]].to_numpy()

        # Stack x, y and dx, dy
        all_xy = np.vstack([xy, last_xy])
        all_dxdy = np.vstack([dxdy, [[np.nan, np.nan]]])

        df_vectors = pd.DataFrame(
            np.hstack([all_xy, all_dxdy]),
            columns=['x', 'y', 'dx', 'dy']
        )
        df_vectors.insert(0, 'file', vectors[0][0])
        # Magnitudes: shape (N,)
        df_magnitudes = pd.DataFrame({'file': magnitudes[0][0], 'magnitude': magnitudes[0][1]})
        # Angles: shape (N,)
        df_angles = pd.DataFrame({'file': angles[0][0], 'angle': angles[0][1]})
        # Resultant: shape (2,)
        df_resultant = pd.DataFrame([{'file': resultant[0], 'resultant_x': resultant[1][0], 'resultant_y': resultant[1][1]}])
        # Average magnitude: scalar
        df_avg_mag = pd.DataFrame([{'file': average_magnitude[0], 'average_magnitude': average_magnitude[1]}])
        # Euclidean distance: shape (N,)
        df_eucl = pd.DataFrame({'file': euclidean_distance[0], 'euclidean_distance': euclidean_distance[1]})
        # Path length: scalar
        df_path = pd.DataFrame([{'file': path_length[0], 'path_length': path_length[1]}])

        # Example: Save to Excel with multiple sheets using pandas
        output_filename = f"{os.path.splitext(os.path.basename(euclidean_distance[0]))[0]}_results.xlsx"

        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            df_resultant.to_excel(writer, sheet_name="Resultant", index=False, float_format="%.3f")
            df_avg_mag.to_excel(writer, sheet_name="AverageMagnitude", index=False, float_format="%.3f")
            df_eucl.to_excel(writer, sheet_name="EuclideanDistances", index=False, float_format="%.3f")
            df_path.to_excel(writer, sheet_name="PathLengths", index=False, float_format="%.3f")
            df_angles.to_excel(writer, sheet_name="Angles", index=False, float_format="%.3f")
            df_vectors.to_excel(writer, sheet_name="Vectors", index=False, float_format="%.3f")
            df_magnitudes.to_excel(writer, sheet_name="Magnitudes", index=False, float_format="%.3f")

        # Auto-fit columns for all sheets
        wb = openpyxl.load_workbook(output_filename)
        for ws in wb.worksheets:
            for col in ws.columns:
                max_length = 0
                col_letter = openpyxl.utils.get_column_letter(col[0].column)
                for cell in col:
                    try:
                        cell_value = str(cell.value)
                        if len(cell_value) > max_length:
                            max_length = len(cell_value)
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = max_length + 10
        wb.save(output_filename)

        print("Single file results exported to", output_filename)

def main():
    app = ctk.CTk()
    app.title("Vector Extraction")

    batch_mode = ctk.BooleanVar(value=False)

    def on_toggle():
        batch_mode.set(not batch_mode.get())
        toggle_btn.configure(text=f"Batch Mode: {'ON' if batch_mode.get() else 'OFF'}")

    toggle_btn = ctk.CTkButton(app, text="Batch Mode: OFF", command=on_toggle)
    toggle_btn.pack(pady=10)

    def on_process():
        export_to_excel(batch_mode.get())

    process_btn = ctk.CTkButton(app, text="Process", command=on_process)
    process_btn.pack(pady=10)

    def delete_xlsx_files():
        cwd = ctk.filedialog.askdirectory(title="Select Directory to Delete XLSX Files")
        deleted = 0
        for f in os.listdir(cwd):
            if f.endswith('.xlsx'):
                try:
                    os.remove(os.path.join(cwd, f))
                    deleted += 1
                except Exception as e:
                    print(f"Could not delete {f}: {e}")
        print(f"Deleted {deleted} .xlsx files from {cwd}")

    delete_btn = ctk.CTkButton(app, text="Delete All XLSX Files", command=delete_xlsx_files)
    delete_btn.pack(pady=10)

    app.mainloop()

if __name__ == "__main__":
    main()
