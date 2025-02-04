# ライブラリー類のインポート

# 既知の実験データを読み込み前処理(標準化)を実行
data = pd.read_csv("zikken_data.csv", encoding="shift_jis")

# 実験における変数の数を定義
num_x = len(data.columns)-1

# 各変数の刻み幅(step),最小値(min),最大値(max),標準化後の刻み幅(norm_step),標準化後の定義域(norm_range)を定義
for i in range(num_x):
    exec("x"+str(i)+"_step = data.iloc[0,"+str(i)+"]")
    exec("x"+str(i)+"_min = data.iloc[1,"+str(i)+"]")
    exec("x"+str(i)+"_max = data.iloc[2,"+str(i)+"]")
    exec("norm_x"+str(i)+"_step = ((x"+str(i)+"_step)/((x"+str(i)+"_max)-(x"+str(i)+"_min)))")
    exec("norm_x"+str(i)+"_range = np.arange(0, (1+norm_x"+str(i)+"_step), norm_x"+str(i)+"_step)")

# MinMaxScalerを用いて標準化処理を実行（標準化前：data→標準化後：norm_dataにデータ名を変更）
# 読み込んだ既知の実験データに定義域外のデータ（最小値や最大値の範囲を超える設定値）がある場合はこの方法では上手くいかないので注意
# 読み込んだデータの3行目（最小値）以下のxの値のみを標準化を実行
data_before_norm = data.iloc[1:,:-1]
mmscaler = MinMaxScaler()
mmscaler.fit(data_before_norm)
norm_data_wo_y = mmscaler.transform(data_before_norm)
norm_data_wo_y_df = pd.DataFrame(norm_data_wo_y)
# norm_data_wo_y_dfの先頭2行（行番号0と1）は最小値と最大値なので取り除く
new_norm_data_wo_y = norm_data_wo_y_df.iloc[2:,:]
new_norm_data_wo_y_reset = new_norm_data_wo_y.reset_index(drop=True)
df_new_norm_data_wo_y_reset = pd.DataFrame(new_norm_data_wo_y_reset)

# 標準化処理したデータにもともとのyの値を結合していくためにyの値だけ格納されたデータフレームを作成していく
data_y = data.iloc[3:,-1]
# のちのデータフレームの結合を考えてindex番号をリセット
data_y_reset = data_y.reset_index(drop=True)
df_data_y_reset = pd.DataFrame(data_y_reset)

# 標準化したxのみのデータフレームとyのみのデータフレームを結合し、csvファイルに書き出し
norm_data = pd.concat([df_new_norm_data_wo_y_reset, df_data_y_reset], axis=1)
norm_data.columns = data.columns
norm_data.to_csv("norm_zikken_data.csv", index=False)
