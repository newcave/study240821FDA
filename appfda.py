import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skfda import FDataGrid
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.representation.basis import BSpline, Fourier
from skfda.exploratory.stats import mean
from skfda.preprocessing.dim_reduction.projection import FPCA

def main():
    st.title('Functional Data Analysis of Temperature Data')

    # 데이터 입력
    st.sidebar.header('1. Enter Hourly Temperature Data')
    city1 = st.sidebar.text_input('Seoul (comma-separated, 24 values)', '3, 5, 8, 12, 15, 18, 20, 21, 20, 18, 13, 10, 7, 5, 4, 3, 2, 2, 3, 4, 6, 5, 4, 3')
    city2 = st.sidebar.text_input('Busan (comma-separated, 24 values)', '6, 7, 10, 14, 17, 20, 22, 23, 22, 21, 17, 15, 12, 10, 9, 8, 7, 7, 8, 9, 10, 9, 8, 7')
    city3 = st.sidebar.text_input('Daegu (comma-separated, 24 values)', '4, 6, 9, 13, 16, 19, 21, 22, 21, 19, 14, 11, 8, 6, 5, 4, 3, 3, 4, 5, 7, 6, 5, 4')

    # 기저 함수 선택
    st.sidebar.header('2. Select Basis Function for Smoothing')
    basis_type = st.sidebar.selectbox("Basis Function", ["BSpline", "Fourier"])
    n_basis = st.sidebar.slider("Number of Basis Functions", 3, 20, 7)

    # 입력 데이터 처리
    try:
        data_seoul = np.array(list(map(float, city1.split(','))))
        data_busan = np.array(list(map(float, city2.split(','))))
        data_daegu = np.array(list(map(float, city3.split(','))))
        data = np.array([data_seoul, data_busan, data_daegu])

        # FDataGrid 객체 생성
        time_points = np.linspace(0, 24, 24)
        fd = FDataGrid(data, time_points)

        # 스무딩
        if basis_type == "BSpline":
            basis = BSpline(n_basis=n_basis, order=4)
        else:
            basis = Fourier(n_basis=n_basis)
            
        smoother = BasisSmoother(basis, method='svd', return_basis=True)
        fd_smooth = smoother.fit_transform(fd)

        # 시각화
        plt.figure(figsize=(12, 6))
        plt.plot(time_points, fd.data_matrix[:, :, 0].T, 'o-', label='Original')
        plt.plot(time_points, fd_smooth.data_matrix[:, :, 0].T, '-', label='Smoothed')
        plt.title('Temperature Data Smoothing')
        plt.xlabel('Hour')
        plt.ylabel('Temperature')
        plt.legend()
        st.pyplot(plt)
    except Exception as e:
        st.sidebar.error('Please make sure the input format is correct.')

if __name__ == "__main__":
    main()
