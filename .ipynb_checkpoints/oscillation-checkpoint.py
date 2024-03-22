import pandas as pd
import numpy as np
from scipy.fft import *
from scipy.signal import convolve


##### バンドパスフィルター群 #####

def frq(k, num, hz=1000):   # num is data length
    return k * hz / num


def ini_kc(fr, num, hz=1000):
    return int(np.ceil(fr * num / hz))


def fin_kc(fr, num, hz=1000):
    return int(np.floor(fr * num / hz))


def _partialFFT(arr, k0, k1, rmDC=True):   # 波数k0〜k1のFFT(k0, k1は自然数),  add rmDC=True
    num = len(arr)
    Fk = fft(arr) / num
    # Fk = fft(arr, norm="forward")      #Scipy ver.1.6以上
    if rmDC == True:     # add
        Fk[0] = 0             # add
    Fk[:k0] = 0           # index = 0, 1, 2, ... k0-1 の要素を0に (低振動数カット...左片側)
    Fk[num-k0+1:] = 0   # index = len(arr)-k1+1, len(arr)-k1+2, ... ,len(arr)-1 の要素を0に (低振動数カット....右片側)
    Fk[k1+1:num-k1] = 0   # index = k1+1, k1+2, ... len(arr)-(k1-1) の要素を0に （高周波数カット）
    return Fk


def partialFFT(arr, ini_frq, fin_frq, hz=1000, rmDC=True):    # add rmDC=True
    """
    Return 周波数範囲を指定したFFTスペクトル1D-ndarray.
    
    Parameters
    ----------
    arr : 1D-ndarray
        サンプリングデータ配列. 1D-ndarray.
    ini_frq : float
        低周波カットオフ周波数. ini_fr以上のスペクトルを残す.
    fin_frq : float
        高周波カットオフ周波数. fin_fr以下のスペクトルを残す.
    hz : int, optional (1000)
        サンプル周波数. デフォルトで1000[Hz].
    
    """
    num = len(arr)
    k0 = ini_kc(ini_frq, num, hz=hz)
    k1 = fin_kc(fin_frq, num, hz=hz)
    return  _partialFFT(arr, k0, k1, rmDC=rmDC)   # add rmDC=rmDC


def bpfilter(arr, ini_frq, fin_frq, hz=1000, rmDC=True):   # 周波数i区間[ini_frq, fin_frq]のバンドパスフィルター
    """
    Retern バンドパスフィルタを施した1D-ndarray(デフォルト). 
        
    Parameters
    ----------
    arr : 1D-ndarray
        サンプリングデータ配列. 1D-numpy配列.
    ini_frq : float
        低周波カットオフ周波数. ini_fr以上のスペクトルを残す.
    fin_frq : float
        高周波カットオフ周波数. fin_fr以下のスペクトルを残す.
    hz : int, optional (1000)
        サンプル周波数. デフォルトで1000[Hz].
    """
    num = len(arr)
    Fk = partialFFT(arr, ini_frq, fin_frq, hz=hz, rmDC=rmDC)    # add rmDC=rmDC
    return np.real(ifft(Fk) * num)
    # return np.real(ifft(Fk, norm="forward"))   #Scipy ver.1.6以上


def frq_and_power(arr, ini_frq, fin_frq, hz=1000, rmDC=True):   #周波数とパワー  # add rmDC=True
    """
    Return tuple
        周波数の1D-ndarrayと，各周波数に対応するパワーの1D-ndarrayのタプル.
    
    Parameters
    ----------
    arr : 1D-ndarray
        サンプリングデータ配列. 1D-numpy配列.
    ini_frq : float
        低周波カットオフ周波数. ini_fr以上のスペクトルを残す.
    fin_frq : float
        高周波カットオフ周波数. fin_fr以下のスペクトルを残す.
    hz : int, optional (1000)
        サンプル周波数. デフォルトで1000[Hz].
    """
    Fk = partialFFT(arr, ini_frq, fin_frq, hz=hz, rmDC=rmDC)    # add rmDC=rmDC
    if len(arr) % 2 == 0:
        eff_num = len(arr) // 2
    else:
        eff_num = len(arr)//2 + 1
    return (fftfreq(len(arr), d=1/hz)[:eff_num], np.abs(Fk**2)[:eff_num])


def mpf(arr, ini_frq, fin_frq, hz=1000, rmDC=True):    # add rmDC=True
    """
    Return 1D-ndarray 
        平均パワー周波数.
    
    Parameters
    ----------
    arr : 1D-ndarray
        サンプリングデータ配列. 1D-numpy配列.
    ini_frq : float
        低周波カットオフ周波数. ini_fr以上のスペクトルを残す.
    fin_frq : float
        高周波カットオフ周波数. fin_fr以下のスペクトルを残す.
    hz : int, optional (1000)
        サンプル周波数. デフォルトで1000[Hz].
    """
    f =  frq_and_power(arr, ini_frq, fin_frq, hz=hz, rmDC=rmDC)[0]    # add rmDC=rmDC
    p = frq_and_power(arr, ini_frq, fin_frq, hz=hz, rmDC=rmDC)[1]    # add rmDC=rmDC
    return np.dot(f, p) / sum(p)


def mp(arr, ini_frq, fin_frq, hz=1000, rmDC=True):    # add rmDC=True
    """
    Return 1D-ndarray 
        平均パワー. 1D-numpy配列.
    
    Parameters
    ----------
    arr : 1D-ndarray
        サンプリングデータ配列. 1D-numpy配列.
    ini_frq : float
        低周波カットオフ周波数. ini_fr以上のスペクトルを残す.
    fin_frq : float
        高周波カットオフ周波数. fin_fr以下のスペクトルを残す.
    hz : int, optional (1000)
        サンプル周波数. デフォルトで1000[Hz].
    """
    p = frq_and_power(arr, ini_frq, fin_frq, hz=hz, rmDC=rmDC)[1]    # add rmDC=rmDC
    return np.mean(p)


def pp(arr, ini_frq, fin_frq, hz=1000, rmDC=True):
    p = frq_and_power(arr, ini_frq, fin_frq, hz=hz, rmDC=rmDC)[1]
    return np.max(p)


def ppf(arr, ini_frq, fin_frq, hz=1000, rmDC=True):
    idx = np.argmax(frq_and_power(arr, ini_frq, fin_frq, hz=hz, rmDC=rmDC)[1])
    return fftfreq(len(arr), d=1/hz)[idx]



##### 標準化・規格化 #####

def z_(arr):
    """
    Return 1D-ndarray
        z値. 1D-numpy配列.
    
    Parameters
    ----------
    arr: 1D-ndarray
    """    
    return (arr - np.mean(arr)) / np.std(arr)


def n_(arr):
    """
    Return 1D-ndarray
        データの規格化. 1D-numpy配列.
    
    Parameters
    ----------
    arr: 1D-ndarray
    """    
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    

##### 畳み込み計算 #####

# 移動平均(default 100)
def mvmu(arr0, arr1, points=100, position="middle"):
    """
    Return tuple as (1D-ndarray, 1D-ndarray) 
        時間と移動平均のタプル.
        
    Parameters
    ----------
    arr0: 1D-ndarray
        時刻配列
    arr1: 1D-ndarray
        各時刻のデータ配列
    points: int, optional (100)
        各時刻毎に平均をとるサンプル数.
    position: str ("middle" or some string)
        "left", "middle", "right"(other) のいずれかの文字列.
        移動平均点を左, 中央, 右に配置するかの指定.
    """
    ker = np.ones(points) / points
    if position == 'right':
        t = arr0[points - 1:]
    elif position == "left":
        t = arr0[: -points + 1:]
    else:
        t = np.convolve(arr0, ker, mode="valid")
    return (t, np.convolve(arr1, ker, mode="valid"))


# 移動標準偏差(default 100)
def mvsd(arr0, arr1, points=100, position="middle"):
    """
    Return tuple as (arr1, 1D-ndarray) 
        時間と移動標準偏差のタプル.
        *) この関数の実行には時間がかかることがある.
        
    Parameters
    ----------
    arr0: 1D-ndarray
        時刻配列
    arr1: 1D-ndarray
        各時刻のデータ配列
    points: int, optional (100)
        各時刻毎に平均をとるサンプル数.
    position: str ("middle" or some string)
        "left", "middle", "right"(other) のいずれかの文字列.
    points: int, optional (100)
        各時刻毎に標準偏差をとるサンプル数.

    """
    ker = np.ones(points) / points
    if position == 'right':
        t = arr0[points - 1:]
    elif position == "left":
        t = arr0[: -points + 1:]
    else:
        t = np.convolve(arr0, ker, mode="valid")
    sd_arr = [np.std(arr1[i : points+i]) for i in range(arr0.size-points+1)]
    return (t, np.array(sd_arr))


##### その他 #####

def nearestidx(arr, val):
    """
    Return int 
        データ配列arrからvalで指定したターゲット値に最も近い値のindex.
        
    Parameters
    ----------
    arr: 1D-ndarray
        データ配列
    val: float
        ターゲット値
    """
    return np.argmin(np.abs(arr - val))


##### クラスOscillation #####

class Oscillation(object):
    def __init__(self, arr, iniHz, finHz, sampHz=1000, gain=None):  
        """
        波状時系列データのインスタンスをつくる(デバイス : TSND151).
        
        Parameters
        ----------
        arr : arraylike
            波状時系列データの１次元配列 (ndarray, pandas配列, データフレーム, リスト)
        iniHz : float
           バンドパスフィルターの抽出波範囲の最小周波数(iniHz<finHz).
        finHz : float
            バンドパスフィルターの抽出波範囲の最大周波数(iniHz<finHz).
        sampHz: int, optional (1000)
            サンプリング周波数. デフォルトで1000Hz.
        gain : int, optional (None)
            取得された電位のゲイン(利得)。データが筋電位の時のみ指定する。
            データフレームのitem列(第1列)がeadxxxxのとき，文字列eadの後ろの
            x, x, x, xのそれぞれがch1, ch2, ch3, ch4のゲインである。
            例えば，ead324cではゲインは，ch1では3, ch2では2, ch3では4, ch4では'c'である。
            ただし，ゲインは16進数で表されているので，'c'は10進数で12でもよい。
        
        Attributes
        ----------
        <instance_name>.arr : arraylike
            処理の施されていない生データ(1D-ndarray). この属性を使うことは推奨しない.                
        <instance_name>.rawarray : arraylike
            処理の施されていない生データ(1D-ndarray). ただし筋電の場合，適切なゲインが掛けられている.
        <instance_name>.bpfarray : arraylike
            iniHz〜finHzのバンドパスフィルターの施されたデータ(1D-ndarray). 
        <instance_name>.hz : int
            サンプリング周波数. 初期値で引数のsampHz.
        <instance_name>.t : float
            波状時系列データの時刻[sec].
        <instance_name>.mt : float
            波状時系列データの時刻[msec].
            
        """
        if gain == None:
            mag = 1
        else:
            mag = {'1': 0.146, '2': 0.073, '3': 0.049, '4': 0.037, '6': 0.024, '8': 0.018, '12': 0.012, 'c': 0.012, 'C':0.012}[str(gain)]
        if type(arr) == np.ndarray:
            self.arr = arr
        elif type(arr) == pd.core.series.Series:
            self.arr = arr.to_numpy()
        elif type(arr) == pd.core.frame.DataFrame:
            self.arr = arr.to_numpy().T[0]
        elif type(arr) == list:
            self.arr = np.array(arr)
        else:
            pass
        self.rawarray = self.arr * mag
        self.__bpfrange = [iniHz, finHz]                   # バンドパスフィルターの抽出波範囲（メソッドで可変）
        self.__default_bpfrange = [iniHz, finHz]    # バンドパスフィルターのデフォルト抽出波範囲(引数で与えられた範囲)
        self.hz = sampHz
        self.mt = np.arange(len(self.rawarray))
        self.t = self.mt/self.hz
        self.bpfarray = bpfilter(self.rawarray, self.__bpfrange[0], self.__bpfrange[1], hz=self.hz)
        

    def ch_bpfrange(self, iniHz, finHz):
        """
        Return None
            バンドパスフィルターの抽出波範囲を変更し, <instance>.bpfarrayを計算し直す.
            iniHzとfinHzはバンドパスフィルターの抽出波範囲 (iniHz<finHz).
        
        Parameters
        ----------
        iniHz : int
            バンドパスフィルターの抽出波範囲の最小周波数(iniHz<finHz).
        finHz : int
            バンドパスフィルターの抽出波範囲の最大周波数(iniHz<finHz).
        """
        self.__bpfrange = [iniHz, finHz]
        self.bpfarray = bpfilter(self.rawarray, self.__bpfrange[0], self.__bpfrange[1], hz=self.hz)        


    def undo_bpfrange(self):
        """
        Return None
            変更したバンドパスフィルターの抽出波範囲を(引数で与えられた)デフォルトに戻し, <instance>.bpfarrayを計算し直す.
        
        Parameters
        ----------
        なし
        """
        self.__bpfrange = self.__default_bpfrange
        self.bpfarray = bpfilter(self.rawarray, self.__bpfrange[0], self.__bpfrange[1], hz=self.hz)        
     
           
    def get_peaks(self):
        """
        Return tuple (1D-ndarray, 1D-ndarray)
            バンドパスフィルターが施された波状データの正の山の最大値とその時刻[sec](1D-ndarray)のタプル.
            ただし，タプルの0indexが時刻，1indexが最大値.
            
        Parameters
        ----------
        なし
        """
        flag1 = np.where(self.bpfarray>0, 1, 0)
        flag2 = np.diff(flag1)
        where1 = np.where(flag2==1)[0]
        where2 = np.where(flag2==-1)[0]
        if where2[0] < where1[0]:
            where2 = np.delete(where2, 0)
        if where2[-1] < where1[-1]: 
            where1 =  np.delete(where1, -1)
        peakindices = np.array([np.argmax(self.bpfarray[i : j+1])+i for i, j in zip(where1, where2)])
        peaks = self.bpfarray[peakindices]
        return (peakindices/self.hz, peaks)

    
    def get_bottoms(self):
        """
        Return tuple (1D-ndarray, 1D-ndarray)
            バンドパスフィルターが施された波状データの正の山の最小値とその時刻[sec](1D-ndarray)のタプル.
            ただし，タプルの0indexが時刻，1indexが最小値.
            
        Parameters
        ----------
        なし
        """
        flag1=np.where(self.bpfarray<0,1,0)
        flag2=np.diff(flag1)
        where1 = np.where(flag2==1)[0]
        where2 = np.where(flag2==-1)[0]
        if where2[0] < where1[0]:
            where2 = np.delete(where2, 0)
        if where1[-1] < where2[-1]:
            where1 = np.delete(where1, -1)
        bottomindices = np.array([np.argmin(self.bpfarray[i : j+1])+i for i, j in zip(where1, where2)])
        bottoms = self.bpfarray[bottomindices]
        return (bottomindices/self.hz, bottoms)
    
    
    def get_saw(self):  #のこぎり
        """
        Return tuple (1D-ndarray, 1D-ndarray)
            バンドパスフィルターが施された波状データのと時刻[sec](1D-ndarray)と各時刻毎のピーク値とボトム値のノコギリ波(1D-ndarray)のタプル.
            (ノコギリ波: ... → 最小 → 最大 → 最小 → 最大 → ...)
            
        Parameters
        ----------
        なし
        """
        mat = np.array([np.concatenate([np.array(self.get_peaks()[0]), np.array(self.get_bottoms()[0])]),
                                    np.concatenate([np.array(self.get_peaks()[1]), np.array(self.get_bottoms()[1])])])
        sortmat = mat[:, np.argsort(mat[0])]    # mat
        return (sortmat[0], sortmat[1])
        
    
    def to_DC(self, formula="full"):
        """
        Return 1D-ndarray
            バンドパスフィルターが施された波状データの整流化.
        
        Parameters
        ----------
        formula : str, optional ("full")
            formula="full"(default)は全波整流， formula="half"は半波整流.
        """
        #arr = np.copy(self.bpfarray)
        arr = self.bpfarray
        if formula=="half":
            arr = np.where(arr<=0, 0, arr)
        else:
            arr = abs(arr)
        return (self.t, arr)
    


    def get_frqpower(self, timerange=None, filtered=True, rmDC=True):
        """
        Return tuple (1D-ndarray, 1D-ndarray)
           バンドパスフィルターが施された波状データの周波数(1D-ndarray)とFFTパワー(1D-ndarray)のタプル
        
        Parameters
        ----------
        timerange : list (None)
            時刻0と時刻1の２要素リスト.
        filtered : bool (True)
            バンドパスフィルターが施されているか否か.
        rmDC : bool (True)
            直流成分を除くか否か.
        """
        if timerange == None:
            arr = self.rawarray
        elif type(timerange) == list and len(timerange)==2:
            idx0 = nearestidx(self.t, timerange[0])
            idx1 = nearestidx(self.t, timerange[1])
            arr = self.rawarray[idx0: idx1]
        else:
            print('timerange引数は長さ2のリストでなければなりません')
            return None
        if filtered == False:
            return frq_and_power(arr, 0, self.hz/2, hz=self.hz, rmDC=rmDC)
        else:
            return frq_and_power(arr, self.__bpfrange[0], self.__bpfrange[1], hz=self.hz, rmDC=rmDC)
    
    
    def get_mp(self):
        """
        Return 1D-ndarray
            オブジェクトのFFT平均パワー.
            
        Parameters
        ----------
        なし.
        """
        arr = self.rawarray
        return mp(arr, self.__bpfrange[0], self.__bpfrange[1], hz=self.hz)
    
    
    def get_mpseries(self):
        """
        Return tuple (1D-ndarray, 1D-ndarray)
            オブジェクトの1sec間隔時刻(1D-ndarray)とそのときの平均パワー(1D-ndarray)のタプル.

        Parameters
        ----------
        なし
        """
        arr = self.rawarray
        div_num = len(arr)//self.hz
        mp_lis =[]
        mt_lis =[]
        for t in range(div_num):
            mp_lis.append(mp(arr[t*self.hz : (t + 1)*self.hz + 1], self.__bpfrange[0], self.__bpfrange[1], hz=self.hz))
            mt_lis.append((t + 1)*self.hz)
        return (np.array(mt_lis)/self.hz, np.array(mp_lis))
    
    
    def get_mpf(self):
        """
        Return 1D-ndarray
            オブジェクトのFFT平均周波数.
        
        Parameters
        ----------
        なし
        """       
        arr = self.rawarray
        return mpf(arr, self.__bpfrange[0], self.__bpfrange[1], hz=self.hz)
    
    
    def get_mpfseries(self):
        """
        Return tuple (1D-ndarray, 1D-ndarray)
            オブジェクトの1sec間隔時刻(1D-ndarray)とそのときの平均パワー周波数(1D-ndarray)のタプル.

        Parameters
        ----------
        なし
        """
        arr =self.rawarray
        div_num = len(arr)//self.hz
        mpf_lis =[]
        mt_lis =[]
        for t in range(div_num):
            mpf_lis.append(mpf(arr[t*self.hz : (t + 1)*self.hz + 1], self.__bpfrange[0], self.__bpfrange[1], hz=self.hz))
            mt_lis.append((t + 1)*self.hz)
        return (np.array(mt_lis)/self.hz, np.array(mpf_lis))
    
    
    def get_ppseries(self):
        """
        Return tuple (1D-ndarray, 1D-ndarray)
            オブジェクトの1sec間隔時刻(1D-ndarray)とそのときの最大パワー(1D-ndarray)のタプル.

        Parameters
        ----------
        なし
        """
        arr =self.rawarray
        div_num = len(arr)//self.hz
        pp_lis =[]
        pt_lis =[]
        for t in range(div_num):
            pp_lis.append(pp(arr[t*self.hz : (t + 1)*self.hz + 1], self.__bpfrange[0], self.__bpfrange[1], hz=self.hz))
            pt_lis.append((t + 1)*self.hz)
        return (np.array(pt_lis)/self.hz, np.array(pp_lis))
    
    
    def get_ppfseries(self):
        """
        Return tuple (1D-ndarray, 1D-ndarray)
            オブジェクトの1sec間隔時刻(1D-ndarray)とそのときの平均パワー周波数(1D-ndarray)のタプル.

        Parameters
        ----------
        なし
        """
        arr =self.rawarray
        div_num = len(arr)//self.hz
        ppf_lis =[]
        pt_lis =[]
        for t in range(div_num):
            ppf_lis.append(ppf(arr[t*self.hz : (t + 1)*self.hz + 1], self.__bpfrange[0], self.__bpfrange[1], hz=self.hz))
            pt_lis.append((t + 1)*self.hz)
        return (np.array(pt_lis)/self.hz, np.array(ppf_lis))

    
    def get_frqpeakpower(self, timerange=None):
        """
        Rutern tuple (1D-ndarray, 1D-ndarray)
            オブジェクトがピークパワーに達したときの周波数と(ピーク)パワー
        Parameters
        ----------
        timerange : list (None)
            時刻0と時刻2の２要素リスト.
        """
        if timerange == None:
            arr = self.rawarray
        elif type(timerange) == list and len(timerange)==2:
            idx0 = nearestidx(self.t, timerange[0])
            idx1 = nearestidx(self.t, timerange[1])
            arr = self.rawarray[idx0: idx1]
        else:
            print('timerange引数は長さ2のリストでなければなりません')
            return None
        f, p = self.get_frqpower(timerange=timerange)
        return (f[np.argmax(p)], np.max(p))            
