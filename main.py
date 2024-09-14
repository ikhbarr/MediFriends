from flask import Flask, render_template,request,jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

sym_des = pd.read_csv("datasets/filtered_symptoms - filtered_symptoms (1).csv")
precaution = pd.read_csv("datasets/filtered_precautions.csv")
workout = pd.read_csv("datasets/filtered_workout.csv")
description = pd.read_csv("datasets/filtered_description.csv")
medication = pd.read_csv('datasets/filtered_medications.csv')
diets = pd.read_csv("datasets/filtered_diets.csv")


# load model===========================================
logreg = pickle.load(open('models/logreg_model.pkl','rb'))



symptom_dict = {
    'gatal': 0, 
    'ruam kulit': 1, 
    'bersin terus menerus': 2, 
    'panas dingin': 3,
    'nyeri sendi': 4, 
    'sakit perut': 5, 
    'keasaman': 6, 
    'bisul di lidah': 7,
    'pengecilan otot': 8, 
    'muntah': 9, 
    'kelelahan': 10, 
    'penurunan berat badan': 11,
    'kegelisahan': 12, 
    'kelesuan': 13, 
    'bercak di tenggorokan': 14,
    'kadar gula tidak teratur': 15, 
    'batuk': 16, 
    'demam tinggi': 17, 
    'sesak napas': 18,
    'berkeringat': 19, 
    'gangguan pencernaan': 20, 
    'sakit kepala': 21,
    'kulit kekuningan': 22, 
    'urin gelap': 23, 
    'mual': 24, 
    'kehilangan nafsu makan': 25,
    'rasa sakit di belakang mata': 26, 
    'sakit punggung': 27, 
    'sembelit': 28,
    'sakit abdominal': 29, 
    'diare': 30, 
    'demam ringan': 31, 
    'mata menguning': 32,
    'pembengkakan kelenjar getah bening': 33, 
    'rasa tidak enak': 34,
    'penglihatan kabur dan terdistorsi': 35, 
    'dahak': 36, 
    'iritasi tenggorokan': 37,
    'mata kemerahan': 38, 
    'tekanan sinus': 39, 
    'pilek': 40, 
    'penyumbatan': 41, 
    'nyeri dada': 42,
    'detak jantung cepat': 43, 
    'nyeri saat buang air besar': 44,
    'nyeri di wilayah anal': 45, 
    'darah dalam tinja': 46, 
    'iritasi di anus': 47,
    'pusing': 48, 
    'kegemukan': 49, 
    'kelaparan berlebihan': 50, 
    'kontak di luar nikah': 51,
    'kehilangan keseimbangan': 52, 
    'kehilangan indra penciuman': 53, 
    'masuk angin': 54,
    'gatal internal': 55, 
    'penampilan toksik': 56, 
    'depresi': 57, 
    'sifat lekas marah': 58,
    'nyeri otot': 59, 
    'bintik merah di tubuh': 60, 
    'sakit perut bagian kiri': 61,
    'peningkatan nafsu makan': 62, 
    'poliuria': 63, 
    'dahak berkarat': 64,
    'kurangnya konsentrasi': 65, 
    'gangguan visual': 66, 
    'darah dalam dahak': 67
}


diseases_list = {
    4: 'GERD', 
    10: 'Penyakit tukak lambung', 
    0: 'AIDS', 
    2: 'Diabetes', 
    5: 'Hipertensi',
    7: 'Migrain', 
    8: 'Penyakit kuning', 
    6: 'Malaria', 
    1: 'Cacar air',
    14: 'Demam berdarah', 
    9: 'Penyakit tipus', 
    12: 'TBC', 
    3: 'Flu biasa',
    11: 'Radang paru-paru', 
    13: 'Wasir dimorfik (tumpukan)'
}


def helper(dis):

    desc = description.loc[description['penyakit'] == dis, 'keterangan'].values
    desc = " ".join(desc) if len(desc) > 0 else "Tidak ada deskripsi yang tersedia."

    pre = precaution.loc[precaution['Penyakit'] == dis, ['Tindakan Pencegahan_1', 'Perhatian_2', 'Perhatian_3', 'Perhatian_4']].values
    pre = pre[0] if len(pre) > 0 else ["Tidak ada tindakan pencegahan yang tersedia."]

    med = medication.loc[medication['Penyakit'] == dis, 'Pengobatan'].values
    med = med if len(med) > 0 else ["Tidak ada pengobatan yang tersedia."]

    die = diets.loc[diets['Penyakit'] == dis, 'Diet'].values
    die = die if len(die) > 0 else ["Tidak ada diet yang tersedia."]

    wrkout = workout.loc[workout['penyakit'] == dis, 'olahraga'].values
    wrkout = wrkout if len(wrkout) > 0 else ["Tidak ada saran workout yang tersedia."]

    return desc, pre, med, die, wrkout

def given_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptom_dict))
    for symptom in patient_symptoms:
        if symptom in symptom_dict:
            input_vector[symptom_dict[symptom]] = 1
    predicted_idx = logreg.predict([input_vector])[0]
    return diseases_list.get(predicted_idx, "Penyakit tidak ditemukan.")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')

        print(symptoms)
        if symptoms == "Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        else:
            # Memisahkan input dengan koma
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            # Hapus ekstra karakter
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = given_predicted_value(user_symptoms)

            # Dapatkan deskripsi penyakit dan tindakan pencegahan
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

        
            my_precautions = [p for p in precautions if p] 

            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
