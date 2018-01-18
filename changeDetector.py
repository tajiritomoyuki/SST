#-*-coding:utf-8-*-
from astropy.io import fits
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv
from scipy.sparse import linalg


class ProgressBar():
    def __init__(self, y):
        self.init_time = 0
        self.y = y
        self.gen = None

    def __gen__(self):
        for i in range(self.y):
            yield i
        return i

    def remaining_time(self, x):
        #時間を計測
        if self.init_time == 0:
            self.init_time = time.perf_counter()
        nowtime = time.perf_counter() - self.init_time
        tot_time = nowtime / 60.0 / float(x+1) * float(self.y)
        rem_time = tot_time - nowtime / 60
        return rem_time

    def progress_bar(self, x, unit=""):
        # 進捗表示のテンプレート
        bar_template = "\r[{0:25s}] {1:2d}% ( {2}{4} / {3}{4} ) 残り{5:3d}分"
        # 進捗バーの表示を作る
        bar = "#" * int(math.floor(25 * x / self.y))
        # パーセンテージの計算
        percent = int(math.floor(100 * x / self.y))
        #時間を計算
        rem_time = int(self.remaining_time(x))
        # 渡されたx,yがfloatなら出力を整形
        if type(x) == float and type(y) == float:
            x_str = "%4.2f" % x
            y_str = "%4.2f" % self.y
            sys.stdout.write(bar_template.format(bar, percent, x_str, y_str, unit, rem_time))
        else:
            sys.stdout.write(bar_template.format(bar, percent, x, self.y, unit, rem_time))

    def update(self):
        if self.gen == None:
            self.gen = self.__gen__()
        x = next(self.gen)
        self.progress_bar(x)


class MergeData():
    def __init__(self, datdir):
        self.datdir = datdir
        self.cadenceno_org = np.array([])
        self.flux_org = np.array([])
        self.base_array = np.array([])
        self.flux_array = np.array([])
        self.filled_flux = np.array([])
        self.split_point = []

    def __gen__(self):
        for cadenceno, flux in zip(self.cadenceno_org, self.flux_org):
            yield(cadenceno, flux)
        yield(0.0, 0.0)

    #fitsファイルからオリジナルデータをインポート
    def importData(self):
        fitslist = glob.glob(os.path.join(self.datdir, "*llc.fits"))
        for fitspath in fitslist:
            with fits.open(fitspath) as hdulist:
                cadenceno = hdulist["LIGHTCURVE"].data.field("CADENCENO").astype(np.uint32)
                flux_org = hdulist["LIGHTCURVE"].data.field("PDCSAP_FLUX")
                mid_flux = np.nanmedian(flux_org)
                flux = flux_org / mid_flux
                self.cadenceno_org = np.hstack((self.cadenceno_org, cadenceno))
                self.flux_org = np.hstack((self.flux_org, flux))

    #軸を設定
    def createBaseArray(self):
        maxval = np.max(self.cadenceno_org)
        minval = np.min(self.cadenceno_org)
        self.base_array = np.arange(minval, maxval + 1, dtype=np.uint32)

    #軸に沿ってfluxを配置
    def createFluxArray(self):
        self.flux_array = np.zeros(len(self.base_array))
        gen = self.__gen__()
        cadenceno, flux = next(gen)
        for i, base_no in enumerate(self.base_array):
            if base_no == cadenceno:
                self.flux_array[i] = flux
                cadenceno, flux = next(gen)
            else:
                self.flux_array[i] = np.nan

    #Nanを埋める
    def fillNan(self):
        tmparray = np.array([])
        last_flux = 0.0
        nan_init = 0
        nan_lim = 5
        for i, flux in enumerate(self.flux_array):
            #fluxがnanのとき
            if np.isnan(flux):
                tmparray = np.hstack((tmparray, np.array([np.nan])))
                #Nanの列開始のポイントを記憶
                if nan_init == 0 :
                    nan_init = i
            else:
                if len(tmparray) == 0:
                    self.filled_flux = np.hstack((self.filled_flux, np.array([flux])))
                elif 0 < len(tmparray) < nan_lim:
                    #平均の値を代入
                    tol = (flux - last_flux) / (len(tmparray) + 1)
                    if tol == 0.0:
                        cor_array = np.array([flux] * (len(tmparray) + 1))
                    else:
                        cor_array = np.arange(last_flux + tol, flux + tol, tol)[0 : len(tmparray) + 1]
                    self.filled_flux = np.hstack((self.filled_flux, cor_array))
                    tmparray = np.array([])
                else:
                    #そのままNanを代入
                    self.filled_flux = np.hstack((self.filled_flux, tmparray, np.array([flux])))
                    tmparray = np.array([])
                    #Nanのセクションを決定
                    nan_section = [nan_init, i]
                    self.split_point.extend(nan_section)
                last_flux = flux
                nan_init = 0


class ChangeDetect():
    def __init__(self, flux_array, window, hist_row, hist_pattern,
                 test_row, test_pattern, lag):
        self.flux_array = flux_array
        self.window = window
        self.hist_row = hist_row
        self.hist_pattern = hist_pattern
        self.test_row = test_row
        self.test_pattern = test_pattern
        self.lag = lag
        self.hist_mtrx = np.array([])
        self.test_mtrx = np.array([])
        self.abn_array = np.array([])
        self.total = 0
        self.num = 0

    def init_process(self):
        #トータルの計算量を計算
        self.total = len(self.flux_array)
        self.num = (self.total - self.lag + 1) - (self.window + self.test_row)
        #異常度配列の最初の方はNanで埋める
        nan_len = self.lag + self.window + self.test_row - 1
        self.abn_array = np.full(nan_len, np.nan)
        #最初の行列を作成
        for i in range(self.hist_row):
            hist_vec = self.flux_array[i : self.window + i].reshape(self.window, 1)
            if len(self.hist_mtrx) == 0:
                self.hist_mtrx = hist_vec
            else:
                self.hist_mtrx = np.hstack((self.hist_mtrx, hist_vec))
        for i in range(self.test_row):
            test_vec = self.flux_array[self.lag + i : self.window + self.lag + i].reshape(self.window, 1)
            if len(self.test_mtrx) == 0:
                self.test_mtrx = test_vec
            else:
                self.test_mtrx = np.hstack((self.test_mtrx, test_vec))

    def main_process(self, bar):
        for i in range(self.num):
            bar.update()
            #特異値分解
            l_vec_hist, s_hist, r_vec_hist = linalg.svds(self.hist_mtrx, k=self.hist_pattern)
            l_vec_test, s_test, r_vec_test = linalg.svds(self.test_mtrx, k=self.test_pattern)
            #最大特異値を求める
            _1, max_s_vec, _2 = linalg.svds(l_vec_hist.T.dot(l_vec_test), k=1)
            #異常度を計算
            abn_deg = 1. - max_s_vec[0]
            #格納
            self.abn_array = np.hstack((self.abn_array, np.array([abn_deg])))
            #一つずつ行列をずらす
            if i != self.num - 1:
                new_hist_vec = self.flux_array[self.hist_row + i : self.hist_row + self.window + i].reshape(self.window, 1)
                self.hist_mtrx = np.hstack((np.hsplit(self.hist_mtrx, [1])[1], new_hist_vec))
                new_test_vec = self.flux_array[self.test_row + self.lag + i : self.test_row + self.lag + self.window + i].reshape(self.window, 1)
                self.test_mtrx = np.hstack((np.hsplit(self.test_mtrx, [1])[1], new_test_vec))


class CaliculateAbnormality():
    def __init__(self, sys_name):
        self.sys_name = sys_name
        self.cadenceno = np.array([])
        self.flux_array = np.array([])
        self.abn_ord = np.array([])
        self.abn_rev = np.array([])
        self.param = {"window" : 100,
                      "hist_row" : 50,
                      "test_row" : 50,
                      "hist_pattern" : 5,
                      "test_pattern" : 5,
                      "lag" : 25
                      }
        self.min_len = self.param["window"] + self.param["test_row"] + self.param["lag"] - 2
        self.tot_num = 0
        self.bar = None

    def searchDir(self):
        sys_name_init = self.sys_name[0:4]
        datdir = os.path.join("D:", "data", sys_name_init, self.sys_name)
        return datdir

    #合計計算量を計算
    def caliculateTotalTimes(self, splited_flux_list):
        for splited_flux in splited_flux_list:
            #nanのときは0秒
            if np.isnan(splited_flux[0]):
                pass
            #基準以下の長さのときも0秒
            elif len(splited_flux) < self.min_len:
                pass
            else:
                num = len(splited_flux) - self.min_len - 1
                self.tot_num += 2 * num

    #各セクションに分けて異常度を計算
    def caliculateAbnDeg(self, splited_flux_list, reverse=False):
        abn_array = np.array([])
        for splited_flux in splited_flux_list:
            #Nanのセクションのときは、異常度を全てNanで登録
            if np.isnan(splited_flux[0]):
                abn_array = np.hstack((abn_array, splited_flux))
            #splitしたデータがあまりにも小さい場合はnanで登録
            elif len(splited_flux) < self.min_len:
                nan_list = np.full(len(splited_flux), np.nan)
                abn_array = np.hstack((abn_array, nan_list))
            # それ以外の場合は異常度を計算
            else:
                #異常は、時間軸を正順でとっても逆順でとっても異常であると仮定
                if reverse == False:
                    flux_array = splited_flux
                else:
                    #反転
                    flux_array = np.fliplr([splited_flux])[0]
                detect = ChangeDetect(flux_array, **self.param)
                detect.init_process()
                detect.main_process(self.bar)
                if reverse == False:
                    abn_array = np.hstack((abn_array, detect.abn_array))
                else:
                    abn_array = np.hstack((abn_array, np.fliplr([detect.abn_array])[0]))
        return abn_array

    def saveData(self):
        dstdir = "D:\\kepler\\abn\\dat"
        #dstdir = "C:\\Users\\tajiri tomoyuki\\school\\dat\\abn"
        basename = "sy{sys_name}wd{window}hr{hist_row}tr{test_row}hp{hist_pattern}tp{test_pattern}lg{lag}.npz"
        name = basename.format(sys_name=self.sys_name, **self.param)
        datpath = os.path.join(dstdir, name)
        np.savez(datpath,
                 cadenceno = self.cadenceno,
                 flux = self.flux_array,
                 abn_ord = self.abn_ord,
                 abn_rev = self.abn_rev
                 )

    def main(self):
        #systemのデータが入っているディレクトリを探す
        datdir = self.searchDir()
        #datdir = "C:\\Users\\tajiri tomoyuki\\school\\kepler\\%s" % self.sys_name
        #データをインポートして加工
        merge = MergeData(datdir)
        merge.importData()
        merge.createBaseArray()
        merge.createFluxArray()
        merge.fillNan()
        self.cadenceno = merge.base_array
        self.flux_array = merge.flux_array

        #異常度を計算
        splited_flux_list = np.split(merge.filled_flux, merge.split_point)
        self.caliculateTotalTimes(splited_flux_list)
        self.bar = ProgressBar(self.tot_num)
        self.abn_ord = self.caliculateAbnDeg(splited_flux_list)
        self.abn_rev = self.caliculateAbnDeg(splited_flux_list, reverse=True)
        self.saveData()
        print("completed!")


def main():
    csvpath = "C:\\Users\\tajiri tomoyuki\\Desktop\\kepler_eclipsing_binary.csv"
    with open(csvpath, "r" ,encoding="utf-8_sig") as f:
        r = csv.reader(f)
        for i in range(1621):
            header = next(r)
        for i, row in enumerate(r):
            sys_name = "%09d" % int(row[0])
            print(sys_name)
            try:
                calabn = CaliculateAbnormality(sys_name)
                calabn.main()
            except:
                print("error raised")


if __name__ == "__main__":
    #main()
    calabn = CaliculateAbnormality("003832716")
    calabn.main()
