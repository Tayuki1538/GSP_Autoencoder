import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt

SAMPLING_RATE = 48000
UPSAMPLE_NUM = 1
c = 331.5 + 0.6 * 25
hgh = 3.98
spk0 = 2.29
spk = []

for i in range(5):
    spk.append((-1)**i*spk0 + ((2*i-1)*(-1)**i+1)/2 * hgh)

# 解析信号の生成
def AnalysisSignal(sgn):
    fft = np.fft.fft(sgn)
    fft[int((len(fft))/2):] = 0
    ifft = np.fft.ifft(fft)
    return ifft

# n次平滑化
def Smoothing(sgn, order):
    smt = []
    for i in range(len(sgn) - (order-1)):
        sum = 0
        for j in range(i, i + (order-1)):
            sum = sum + sgn[j]
        smt.append(sum/order)
    return smt

# チャープ信号の作成
def ChirpMaker(f0, ff, tC, flg):
    t = np.arange(1, tC * SAMPLING_RATE + 1, 1)
    k = (ff - f0) / tC
    phi0 = 0
    chirp = 0.5 * np.sin(phi0 + 2*np.pi*(f0*(t-1)/SAMPLING_RATE + (k/2)*((t-1)/SAMPLING_RATE)**2))

    # ハン窓の設定
    Hann = 0.5 - 0.5 * np.cos(2*np.pi*(t / (tC*SAMPLING_RATE)))

    if (flg == 1):
        chirp = chirp * Hann
        chirp = 0.5 * chirp / np.max(chirp)

    return chirp

# アップサンプリング
def UpSampling(sgn, num):
    upSgn = np.zeros(len(sgn)*num)
    for i in range(len(sgn)):
        upSgn[i*num] = sgn[i]
    fft = np.fft.fft(upSgn)
    fft[int(len(sgn)/2):-int(len(sgn)/2)] = 0
    ifft = np.fft.ifft(fft)
    return np.real(ifft)

# 包絡線の算出
def EnvelopeMaker(chirp, sound, index, wndNum, n, i):
    anlChr = AnalysisSignal(chirp)
    anlSnd = AnalysisSignal(sound[int((index-500)+0.25*i*SAMPLING_RATE):int((index-500)+0.25*(i+1)*SAMPLING_RATE)])

    crl = np.conjugate(np.fft.fft(np.concatenate([anlChr, np.zeros(len(anlSnd)-len(anlChr))]))) * np.fft.fft(anlSnd)
    env = abs(np.fft.ifft(crl))
    env = np.convolve(env[0:int(SAMPLING_RATE/4)], np.ones(11)/11, mode="same")
    env = UpSampling(env,n)[500*n:wndNum*n]

    return env

# スタートタイムの決定
def StartTime(chirp, sound):
    anlChr = AnalysisSignal(chirp)
    anlSnd = AnalysisSignal(sound)

    crl = np.conjugate(np.fft.fft(np.concatenate([anlChr, np.zeros(len(anlSnd)-len(anlChr))]))) * np.fft.fft(anlSnd)
    env = abs(np.fft.ifft(crl))
    env = np.convolve(env, np.ones(11)/11, mode="same")
    pak = PeakDetector(env, 16*UPSAMPLE_NUM, 500*UPSAMPLE_NUM, 4)

    return pak[0][0]/SAMPLING_RATE

# CA-CFAR的なピーク検出
def PeakDetector(sgn, delta, grdItv, thrs=1.6):
    # sgn 信号列
    # delta 検出の時間窓長
    # grdItv ガード幅(平均に用いない部分)

    lst = []
    mean = np.mean(sgn)
    strNum = delta + grdItv # インデックスの初期値
    fnsNum = fnsNum = len(sgn) - delta - grdItv # インデックスの最後値

    for i in range(strNum, fnsNum):
        trs = np.mean(sgn[i-grdItv:i+grdItv])*thrs
        if sgn[i]==np.max(sgn[i-delta:i+delta]) and sgn[i]>=mean and sgn[i]>=trs:
            lst.append([i,sgn[i]])
    return lst


# 2次反射波の到来時刻の範囲を返す
def T2Func(t0, t1, zMin, zMax):
    k = c*(t1-t0)
    alpha = (4*spk0**2 - k**2)/k**2
    beta = (4*spk0**2 -k**2)/4
    apek = k**2 / (4*spk0)

    # T2の定義
    def T2(z):
        return (np.emath.sqrt(alpha * z**2 - beta + (spk[2] -z)**2) - np.emath.sqrt(alpha * z**2 - beta + (spk[0] -z)**2)) / c + t0


    tLst = [T2(zMin), T2(zMax), np.real(T2(apek))]

    return np.array([np.min(tLst), np.max(tLst)]) * SAMPLING_RATE

# 3次反射波の到来時刻の範囲を返す
def T3Func(t0, t1, zMin, zMax):
    k = c*(t1-t0)
    alpha = (4*spk0**2 - k**2)/k**2
    beta = (4*spk0**2 -k**2)/4
    apek = k**2 / (4*spk0)

    # T3の定義
    def T3(z):
        return (np.emath.sqrt(alpha * z**2 - beta + (spk[3] -z)**2) - np.emath.sqrt(alpha * z**2 - beta + (spk[0] -z)**2)) / c + t0

    tLst = [T3(zMin), T3(zMax), np.real(T3(apek))]

    return np.array([np.min(tLst), np.max(tLst)]) * SAMPLING_RATE

# T2やT3に含まれるピークを抽出
def PeakSelector(pak, eta, tPre):
    i = 0
    lst = []
    while(i<len(pak)):
        if (eta[0] < pak[i][0]/SAMPLING_RATE <= eta[1] and tPre < pak[i][0]/SAMPLING_RATE):
            lst.append(pak[i][0]/SAMPLING_RATE)
        if pak[i][0]/SAMPLING_RATE >= eta[1]:
            break
        i+=1
    return lst

# 最適化を行い，ピークの位置を推定．その時の評価値もリストとして返す．
def Estimator(time):
    def func(x):
        t = x[0]
        d = x[1]
        z = x[2]
        return (np.emath.sqrt(d**2 + (spk[0] - z)**2) - c*(time[0] - t))**2 \
            + (np.emath.sqrt(d**2 + (spk[1] - z)**2) - c*(time[1] - t))**2 \
            + (np.emath.sqrt(d**2 + (spk[2] - z)**2) - c*(time[2] - t))**2 \
            + (np.emath.sqrt(d**2 + (spk[3] - z)**2) - c*(time[3] - t))**2

    # bounds_t = (0,np.inf)
    # bounds_d = (0,np.inf)
    # bounds_z = (0,np.inf)
    # bound = [bounds_t, bounds_d,bounds_z]

    #　不等式制約条件 d>=0
    def inequality_constraint1(x):
        d = x[1]
        return d

    # 不等式制約条件 z>0
    def inequality_constraint2(x):
        z = x[2]
        return z

    constraint1 = {"type":"ineq","fun":inequality_constraint1}
    constraint2 = {"type":"ineq","fun":inequality_constraint2}
    constraint = [constraint1,constraint2]

    x0 = [0.01,2,1]

    result=scipy.optimize.minimize(func, x0, method="SLSQP", constraints=constraint)

    est = [result.x[0], result.x[1], result.x[2], result.fun]
    est[0] = est[0] * SAMPLING_RATE
    return est

def DecisionMaker(pak, zMin, zMax):
    t0 = pak[0][0]/SAMPLING_RATE
    rssMin = 100
    cnt = 0
    rst = 0
    cmb = []

    # t1の絞り込み
    t1 = PeakSelector(pak, [t0, 2*zMax/c + t0], t0)
    # t2の絞り込み
    for i in range(len(t1)):
        eta2 = T2Func(t0, t1[i], zMin, zMax) / SAMPLING_RATE
        t2 = PeakSelector(pak, eta2, t1[i])
        for j in range(len(t2)):
            eta3 = T3Func(t0, t1[i], zMin, zMax) / SAMPLING_RATE
            t3 = PeakSelector(pak, eta3, t2[j])
            # 全ての組み合わせに対して位置の推定
            for k in range(len(t3)):
                est = Estimator([t0, t1[i], t2[j], t3[k]])
                cnt += 1
                # print(est)
                if (est[3] < rssMin):
                    rssMin = est[3]
                    rst = est
                    cmb = [t0, t1[i], t2[j], t3[k]]

    return [rst, cnt, cmb]


def bandpass_filter(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2 #ナイキスト周波数
    wp = fp / fn  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "band") #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x) #信号に対してフィルタをかける
    return y  

def GetGSP(chirp, snd, index_sync, is_train=True, n = 20):
    """
    ダウンサンプリングにより，特徴量を480次元に削減
    """

    GSP_train = []
    index = index_sync * SAMPLING_RATE
    if is_train:
        start = 0
    else:
        start = 20
    for i in range(start, start+n):
        env = EnvelopeMaker(chirp, snd, index, 18000, UPSAMPLE_NUM, i)
        pak = PeakDetector(env, 16*UPSAMPLE_NUM, 500*UPSAMPLE_NUM, 4)

        # ------------------------------------------------------------------------------------------------------------------
        # GSP = np.array(env[int(pak[0][0])+100 : int(pak[0][0] + 0.1*SAMPLING_RATE*UPSAMPLE_NUM)+100])
        GSP = np.array(env[:int(0.1*SAMPLING_RATE*UPSAMPLE_NUM)])
        # ------------------------------------------------------------------------------------------------------------------

        GSP_train.append(GSP)# /np.max(GSP)
    GSP_train = np.array(GSP_train)
    GSP_train = signal.resample(GSP_train, 480, axis=1)
    
    return GSP_train

def GetSyncGSP(chirp, snd, index_sync, thrs=1.6):
    l = []
    index = (index_sync + 20) * SAMPLING_RATE
    estimation_time_list = []

    for j in range(40):
        env = EnvelopeMaker(chirp, snd, index, 4000, UPSAMPLE_NUM, j)
        if thrs == 1.6:
            pak = np.array(PeakDetector(env, 16*UPSAMPLE_NUM, 500*UPSAMPLE_NUM, thrs))
        else:
            pak = np.array(PeakDetector(env, 16*2*UPSAMPLE_NUM, 500*UPSAMPLE_NUM, thrs))
        est = DecisionMaker(np.dot(pak, [[1/UPSAMPLE_NUM, 0], [0, 1]]), 0.7, 1.5)
        estimation_time_list.append(est[0][0])

    ans = np.polyfit(np.arange(0, 10, 0.25), estimation_time_list, 1)
    clock = ans[1]
    drift = ans[0]


    # 20回分のToFプロファイルを切り出してリストに格納
    index = (40 + index_sync) * SAMPLING_RATE
    GSPList1 = []
    GSPList2 = []
    for n in range(40):
        tp = EnvelopeMaker(chirp, snd, index, 8000, UPSAMPLE_NUM, n)
        # print(len(tp), (drift*(20+0.25*n) + clock)*UPSAMPLE_NUM, (drift*(20+0.25*n) + clock + 0.05*SAMPLING_RATE)*UPSAMPLE_NUM)
        if n < 20:
            GSPList1.append(tp[int((drift*(20+0.25*n) + clock)*UPSAMPLE_NUM) : int((drift*(20+0.25*n) + clock + 0.1*SAMPLING_RATE)*UPSAMPLE_NUM)])
        else:
            GSPList2.append(tp[int((drift*(20+0.25*n) + clock)*UPSAMPLE_NUM) : int((drift*(20+0.25*n) + clock + 0.1*SAMPLING_RATE)*UPSAMPLE_NUM)])
            plt.plot(tp[int((drift*(20+0.25*n) + clock)*UPSAMPLE_NUM) : int((drift*(20+0.25*n) + clock + 0.1*SAMPLING_RATE)*UPSAMPLE_NUM)])

    plt.show()

    return [GSPList1, GSPList2]

# 高速フーリエ変換
def calc_amp(data, fs):
    '''フーリエ変換して振幅スペクトルを計算する関数
    '''
    N = len(data)
    window = signal.windows.hann(N)
    F = np.fft.fft(data * window)
    freq = np.fft.fftfreq(N, d=1/fs) # 周波数スケール
    F = F / (N / 2) # フーリエ変換の結果を正規化
    F = F * (N / sum(window)) # 窓関数による振幅減少を補正する
    Amp = np.abs(F) # 振幅スペクトル
    return Amp, freq
