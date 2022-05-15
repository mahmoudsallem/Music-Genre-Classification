
from flask import Flask, request,  render_template
import keras
import joblib
import librosa
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

model = keras.models.load_model('model.h5')
np_scaled = joblib.load('scaler.h5')


@app.route("/upload", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file1 = preprocess(file)
        file2 = np_scaled.transform(file1)

        pred = model.predict(file2)
        fpred = np.argmax(pred)

    return render_template('index.html', fpred=fpred)


def preprocess(file):
    y, sr = librosa.load(file, duration=30)

    #data0 = []
    # defining the Chroma features in the dataframe
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = chroma.mean()  # chroma_stft_mean
    chroma_stft_var = chroma.var()  # chroma_stft_var
    # chroma_stft_std = chroma.std()#chroma_stft_std

    # defining the rms features in the dataframe
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()  # rms_mean
    rms_var = rms.var()  # rms_var
    # rms_std = rms.std()#rms_std

    # defining the spectral centroid features in the dataframe
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = cent.mean()  # spec_cent_mean
    spectral_centroid_var = cent.var()  # spec_cent_var
    # spec_cent_std = cent.std()#spec_cent_std

    # defining the spectral bandwidth features in the dataframe
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = spec_bw.mean()  # spec_ban_mean
    spectral_bandwidth_var = spec_bw.var()  # spec_ban_var
    # spec_ban_std = spec_bw.std()#spec_ban_std

    # defining the spectral roll-off features in the dataframe
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = rolloff.mean()  # spec_roll_mean
    rolloff_var = rolloff.var()  # spec_roll_var

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = zero_crossing_rate.mean()  # zero_crossing_rate_mean
    zero_crossing_rate_var = zero_crossing_rate.var()  # zero_crossing_rate_var

    harmonic = librosa.effects.harmonic(y)
    harmony_mean = harmonic.mean()  # harmony_mean
    harmony_var = harmonic.var()  # harmony_var

    C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('A1')))
    freqs = librosa.cqt_frequencies(C.shape[0], fmin=librosa.note_to_hz('A1'))
    perceptr = librosa.perceptual_weighting(C**2, freqs, ref=np.max)
    perceptr_mean = perceptr.mean()  # perceptr_mean
    perceptr_var = perceptr.var()  # perceptr_var

    # defining the mel frequency cepstrum coefficient features in the dataframe
    mfcc1 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)  # n_mfcc=1
    mfcc1_mean = mfcc1.mean()  # mfcc1_mean
    mfcc1_var = mfcc1.var()  # mfcc1_var

    mfcc2 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)  # n_mfcc=2
    mfcc2_mean = mfcc2.mean()  # mfcc2_mean
    mfcc2_var = mfcc2.var()  # mfcc2_var

    mfcc3 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)  # n_mfcc=3
    mfcc3_mean = mfcc3.mean()  # mfcc3_mean
    mfcc3_var = mfcc3.var()  # mfcc3_var

    mfcc4 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4)  # n_mfcc=4
    mfcc4_mean = mfcc4.mean()  # mfcc4_mean
    mfcc4_var = mfcc4.var()  # mfcc4_var

    mfcc5 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)  # n_mfcc=5
    mfcc5_mean = mfcc5.mean()  # mfcc5_mean
    mfcc5_var = mfcc5.var()  # mfcc5_var

    mfcc6 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=6)  # n_mfcc=6
    mfcc6_mean = mfcc6.mean()  # mfcc6_mean
    mfcc6_var = mfcc6.var()  # mfcc6_var

    mfcc7 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=7)  # n_mfcc=7
    mfcc7_mean = mfcc7.mean()  # mfcc7_mean
    mfcc7_var = mfcc7.var()  # mfcc7_var

    mfcc8 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8)  # n_mfcc=8
    mfcc8_mean = mfcc8.mean()  # mfcc8_mean
    mfcc8_var = mfcc8.var()  # mfcc8_var

    mfcc9 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=9)  # n_mfcc=9
    mfcc9_mean = mfcc9.mean()  # mfcc9_mean
    mfcc9_var = mfcc9.var()  # mfcc9_var

    mfcc10 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)  # n_mfcc=10
    mfcc10_mean = mfcc10.mean()  # mfcc10_mean
    mfcc10_var = mfcc10.var()  # mfcc10_var

    mfcc11 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=11)  # n_mfcc=11
    mfcc11_mean = mfcc11.mean()  # mfcc11_mean
    mfcc11_var = mfcc11.var()  # mfcc11_var

    mfcc12 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)  # n_mfcc=12
    mfcc12_mean = mfcc12.mean()  # mfcc11_mean
    mfcc12_var = mfcc12.var()  # mfcc11_var

    mfcc13 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # n_mfcc=13
    mfcc13_mean = mfcc13.mean()  # mfcc10_mean
    mfcc13_var = mfcc13.var()  # mfcc10_var

    mfcc14 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)  # n_mfcc=14
    mfcc14_mean = mfcc14.mean()  # mfcc14_mean
    mfcc14_var = mfcc14.var()  # mfcc14_var

    mfcc15 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15)  # n_mfcc=15
    mfcc15_mean = mfcc15.mean()  # mfcc15_mean
    mfcc15_var = mfcc15.var()  # mfcc15_var

    mfcc16 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16)  # n_mfcc=16
    mfcc16_mean = mfcc16.mean()  # mfcc16_mean
    mfcc16_var = mfcc16.var()  # mfcc16_var

    mfcc17 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=17)  # n_mfcc=17
    mfcc17_mean = mfcc17.mean()  # mfcc17_mean
    mfcc17_var = mfcc17.var()  # mfcc17_var

    mfcc18 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=18)  # n_mfcc=18
    mfcc18_mean = mfcc18.mean()  # mfcc18_mean
    mfcc18_var = mfcc18.var()  # mfcc18_var

    mfcc19 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=19)  # n_mfcc=19
    mfcc19_mean = mfcc19.mean()  # mfcc19_mean
    mfcc19_var = mfcc19.var()  # mfcc19_var

    mfcc20 = librosa.feature.mfcc(y=y, sr=sr)  # n_mfcc=20
    mfcc20_mean = mfcc20.mean()  # mfcc20_mean
    mfcc20_var = mfcc20.var()  # mfcc20_var

    # defining the zero crossing rate features in the dataframe
    zero_crosing_rate = librosa.feature.zero_crossing_rate(y)
    zero_cros_mean = zero_crosing_rate.mean()  # zero_cros_mean
    zero_cros_var = zero_crosing_rate.var()  # zero_cros_var

    # defining the Tempo feature in the dataframe
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    Tempo = tempo.mean()  # Tempo

    data0 = [[chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, spectral_centroid_mean, spectral_centroid_var,
              spectral_bandwidth_mean, spectral_bandwidth_var, rolloff_mean, rolloff_var,
              zero_crossing_rate_mean, zero_crossing_rate_var, harmony_mean, harmony_var, perceptr_mean, perceptr_var, tempo, mfcc1_mean, mfcc1_var, mfcc2_mean, mfcc2_var,
              mfcc3_mean, mfcc3_var, mfcc4_mean, mfcc4_var, mfcc5_mean, mfcc5_var, mfcc6_mean, mfcc6_var, mfcc7_mean, mfcc7_var, mfcc8_mean, mfcc8_var,
              mfcc9_mean, mfcc9_var, mfcc10_mean, mfcc10_var, mfcc11_mean, mfcc11_var, mfcc12_mean, mfcc12_var,
              mfcc13_mean, mfcc13_var, mfcc14_mean, mfcc14_var, mfcc15_mean, mfcc15_var, mfcc16_mean, mfcc16_var, mfcc17_mean, mfcc17_var,
              mfcc18_mean, mfcc18_var, mfcc19_mean, mfcc19_var, mfcc20_mean, mfcc20_var]]
    #
    return np.array(data0)


if __name__ == "__main__":
    app.run(debug=True)
