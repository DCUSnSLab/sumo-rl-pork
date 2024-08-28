import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os


def plot_csv_files(file_paths):
    plt.figure(figsize=(10, 6))

    for file_path in file_paths:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # 파일 이름을 레전드로 사용
        file_name = os.path.basename(file_path)

        # system_total_co2 열의 변화를 시간에 따라 플롯
        plt.plot(df['step'], df['system_total_co2'], label=f"{file_name}")

        # system_total_co2의 총합 계산
        total_co2 = df['system_total_co2'].sum()

        # 각 파일의 총합을 그래프에 텍스트로 표시
        plt.text(df['step'].iloc[-1], df['system_total_co2'].iloc[-1], f"Total CO2: {total_co2:.2f}", fontsize=10,
                 color='black')

    # 그래프 제목 및 라벨 설정
    plt.title('System Total CO2 over Time')
    plt.xlabel('Time step (seconds)')
    plt.ylabel('Total CO2')
    plt.ylim(bottom=0)
    plt.grid(True)

    # 레전드 및 그래프 표시
    plt.legend()
    plt.show()


def select_files_and_plot():
    # Tkinter를 사용해 파일 다이얼로그를 연다
    root = Tk()
    root.withdraw()  # Tkinter 창을 숨김

    # 여러 개의 CSV 파일을 선택할 수 있게 설정
    file_paths = filedialog.askopenfilenames(
        title="Select CSV files",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )

    # 선택된 파일이 있으면 시각화
    if file_paths:
        plot_csv_files(file_paths)


# GUI를 통해 파일을 선택하고 시각화
select_files_and_plot()
