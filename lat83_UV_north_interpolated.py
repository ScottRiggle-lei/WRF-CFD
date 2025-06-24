import pandas as pd
import netCDF4
import numpy as np

# 大气层温度假设常数
T0 = 288.15  # 近地表温度 (K)
L = 0.0065  # 温度递减率 (K/m)
R = 287.05  # 干空气气体常数 (J/(kg·K))
g = 9.80665  # 重力加速度 (m/s^2)
size_h = 39  # 设置数据提取层数
size_s = 1  # 设置数据开始位置（修正为110，因为最大索引为110）
size_e = 105# 设置数据结束位置
size_n = 121  # 设置数据开始位置（修正为110，因为最大索引为110）
size_w = 1 # 设置数据结束位置
lat_r = 111320   # 设置经度度系数
long_r = 102736.78   # 设置经度度系数

def calculate_heights(data, size_h, size_e):
    # 获取所需变量
    znu = data.variables['ZNU'][0, :size_h]  # ZNU层次，只取第一个时间步的前size_h层
    psfc = data.variables['PSFC'][:, size_s:size_n, size_w:size_e]  # 地面气压
    hgt = data.variables['HGT'][:, size_s:size_n, size_w:size_e]  # 地表高度
    ptop = data.variables['P_TOP'][0]  # P_TOP变量
    u_shape = data.variables['U'][:, :size_h, size_s:size_n, size_w:size_e].shape
    # 初始化高度数组，保持与 data.variables['U'][:, :size_h, :size_e, :size_e] 相同的形状
    heights = np.zeros(u_shape)

    # 计算每层气压并转换为高度
    for ti in range(u_shape[0]):  # 时间步
        psfc_ti = psfc[ti]
        p = znu[:, np.newaxis, np.newaxis] * (psfc_ti - ptop) + ptop
        heights[ti] = hgt[ti] + (T0 / L) * (1 - (p / psfc_ti) ** (R * L / g))

    return heights

def interpolate_within_same_time(df, num_points=4):
    interpolated_data = []

    unique_times = df['Repeated_time'].unique()

    for time in unique_times:
        section = df[df['Repeated_time'] == time]

        for i in range(len(section) - 1):
            row1 = section.iloc[i]
            row2 = section.iloc[i + 1]

            interpolated_data.append(row1.values.tolist())  # 保留每个区间的原始数据行

            # 仅对 x 坐标连续的行进行插值
            if row1['Longitude'] < row2['Longitude']:
                # 对高度、x坐标和风速进行插值
                interp_heights = np.linspace(row1['Repeated_height'], row2['Repeated_height'], num_points + 2)
                interp_x_coords = np.linspace(row1['Longitude'], row2['Longitude'], num_points + 2)
                interp_wind_Uspeeds = np.linspace(row1['lat_U'], row2['lat_U'], num_points + 2)
                interp_wind_Vspeeds = np.linspace(row1['lat_V'], row2['lat_V'], num_points + 2)
                interp_wind_Wspeeds = np.linspace(row1['lat_W'], row2['lat_W'], num_points + 2)
                interp_wind_UST = np.linspace(row1['lat_UST'], row2['lat_UST'], num_points + 2)
                interp_wind_K = np.linspace(row1['lat_K'], row2['lat_K'], num_points + 2)
                interp_wind_OMG = np.linspace(row1['lat_OMG'], row2['lat_OMG'], num_points + 2)
                for j in range(1, num_points + 1):  # Exclude the first point to avoid duplicate
                    interpolated_data.append(
                        [row1['Repeated_time'], interp_heights[j], interp_x_coords[j], interp_wind_Uspeeds[j],
                         interp_wind_Vspeeds[j],interp_wind_Wspeeds[j], interp_wind_UST[j], interp_wind_K[j],interp_wind_OMG[j]])

        interpolated_data.append(section.iloc[-1].values.tolist())  # 保留每个区间的最后一行数据

    return pd.DataFrame(interpolated_data, columns=df.columns)

# 主程序
if __name__ == '__main__':
    # 读取 NetCDF 数据
    nc_file = ('F:/WRF_CFD/09_weihaifarm/dongxingfarm_WRF/wrfout_d03_2024-04-24_18_00_00.nc')  # 你的 NetCDF 文件路径
    data = netCDF4.Dataset(nc_file, 'r')
    variables = data.variables.keys()
    print(variables)
    print(data.variables['HGT'][0][0][0])
    # 计算高度
    height_z = calculate_heights(data, size_h, size_e)

    # 提取ZNU变量的每个时间步的前10层数据并重复每层110次
    num_time_steps = len(data.variables['Times'][:])

    time = []
    longitudes_x = []
    lat_y_U = []
    lat_y_V = []
    lat_y_W = []
    lat_y_UST = []
    lat_Z = []
    lat_y_K = []
    lat_y_OMG = []

    for t in range(num_time_steps):
        psfc_ti = data.variables['PSFC'][t, size_s-1:size_n+2, size_w-1:size_e+2]
        hgt_ti = data.variables['HGT'][t, size_s-1:size_n+2, size_w-1:size_e+2]
        znu = data.variables['ZNU'][t,:size_h]
        ust_data = data.variables['UST'][t, size_s-1:size_n + 2, size_w - 1:size_e + 2]
        ust_expanded = np.repeat(ust_data[np.newaxis, :, :], size_h, axis=0)
        p = znu[:, np.newaxis, np.newaxis] * (psfc_ti - data.variables['P_TOP'][0]) + data.variables['P_TOP'][0]
        heights_ti = hgt_ti + (T0 / L) * (1 - (p / psfc_ti) ** (R * L / g))-60
        u_data = data.variables['U'][t, :size_h, size_s-1:size_n+2, size_w-1:size_e+2]
        v_data = data.variables['V'][t, :size_h, size_s-1:size_n+2, size_w-1:size_e+2]
        w_data = data.variables['W'][t, :size_h, size_s - 1:size_n + 2, size_w - 1:size_e + 2]
        xlong_data = data.variables['XLONG'][t, size_s-1:size_n+2, size_w-1:size_e+2]
        k_data = np.where(
            heights_ti > 0,  # 条件：heights_ti 大于 0
            (0.2 * (10 / heights_ti)) ** 0.2 * (u_data ** 2 + v_data ** 2 + w_data ** 2),  # 满足条件时计算
            0  # 否则为 0
        )

        omg_data = np.where(
            heights_ti > 0,  # 条件：heights_ti 大于 0
            (ust_expanded ** 3) / (0.0417 * k_data * 0.41 * (heights_ti + 0.03)),  # 满足条件时计算
            0  # 否则为 0
        )
        # k_data = (0.2 * (10 / heights_ti)) ** 0.2*(u_data**2+v_data**2+w_data**2)
        # omg_data=(ust_expanded**3)/(0.0417*k_data*0.41*(heights_ti+0.03))
        for h in range(size_h):
            time.extend([t] * (size_e-size_w+2))
            lat_y_U.extend(u_data[h, size_n-size_s+1, :size_e-size_w+2])  # 取反
            lat_y_V.extend(v_data[h, size_n-size_s+1, :size_e-size_w+2])  # 取反
            lat_y_W.extend(w_data[h, size_n - size_s+1, :size_e - size_w +2])  # 取反
            lat_y_UST.extend(ust_expanded[h, size_n - size_s+1, :size_e - size_w + 2])  # 取
            lat_Z.extend(heights_ti[h, size_n-size_s+1, :size_e-size_w+2])
            longitudes_x.extend((xlong_data[size_n-size_s+1, :size_e-size_w+2] -xlong_data[size_n-size_s+1, size_w-size_w+1]) * long_r)
            lat_y_K.extend(k_data[h, size_n - size_s+1, :size_e - size_w + 2])
            lat_y_OMG.extend(omg_data[h, size_n - size_s+1, :size_e - size_w + 2])

            # omg_values = omg_data[h, size_n - size_s, :size_e - size_w + 1]//
            # omg_values = np.where(omg_values < 0.001, 0.001, omg_values) // # 将小于0.0001的值替换为1//
            # lat_y_OMG.extend(omg_values)
    print(f'Length of time: {len(time)}')
    print(f'Length of lat_Z: {len(lat_Z)}')
    print(f'Length of longitudes_y: {len(longitudes_x)}')
    print(f'Length of lat_y_U: {len(lat_y_U)}')
    print(data.variables['XLONG'][0][0][56])
    print(data.variables['XLAT'][0][55][0])

    # 将数据写入到 DataFrame，并保留数据的原始精度
    pd.set_option('display.float_format', lambda x: '%.15g' % x)
    print(f'Length of time: {len(time)}')
    output_df = pd.DataFrame({
        'Repeated_time': time,
        'Repeated_height': lat_Z,  # 使用计算后的 heights
        'Longitude': longitudes_x,
        'lat_U': lat_y_U,
        'lat_V': lat_y_V,
        'lat_W': lat_y_W,
        'lat_UST': lat_y_UST,
        'lat_K': lat_y_K,
        'lat_OMG': lat_y_OMG

    })

    # 执行插值
    interpolated_df = interpolate_within_same_time(output_df, num_points=1)

    # 保存插值后的结果到文件
    interpolated_df.to_csv('F:/WRF_CFD/09_weihaifarm/dongxingfarm_WRF/lath_north_UV20220219.txt', index=False, header=False, sep='\t', float_format='%.5f')
    # 关闭 NetCDF 文件
    data.close()

    # 显示结果
    print(interpolated_df.head(40)) # Display first 40 rows to verify
