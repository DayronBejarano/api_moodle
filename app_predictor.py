from flask import Flask, request, jsonify
import pandas as pd
import weka.core.jvm as jvm
import weka.core.converters as converters
import weka.classifiers
from weka.core.serialization import read as weka_read
import os
import tempfile

# Iniciar JVM
if not jvm.started:
    jvm.start()

# Ruta del modelo entrenado
RUTA_MODELO = r"C:\Users\dayro\OneDrive\OneDrive\Documentos\dayron\modelo6.model"

# Cargar modelo
def cargar_modelo(ruta_modelo):
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"Modelo no encontrado en: {ruta_modelo}")
    return weka.classifiers.Classifier(jobject=weka_read(ruta_modelo))

modelo = cargar_modelo(RUTA_MODELO)

# Flask app
app = Flask(__name__)

@app.route("/evaluar", methods=["POST"])
def evaluar_archivo():
    try:
        archivo = request.files["file"]
        incluir_id = request.args.get("incluir_id", "false").lower() == "true"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            archivo_csv = temp_file.name
            archivo.save(archivo_csv)

        # Leer CSV
        try:
            df = pd.read_csv(archivo_csv, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(archivo_csv, sep=';', encoding='latin1')

        df.columns = df.columns.str.lower().str.strip()
        ids = df['id estudiante'].tolist() if incluir_id and 'id estudiante' in df.columns else [None] * len(df)

        # Renombrar columnas
        renombres = {
            '¿ha ingresado al curso alguna vez?': 'Ingresado',
            'contenidos completados [%]': 'Contenido_completado',
            'días sin ingreso al curso desde su matrícula': 'Dias_sin_ingreso_curso',
            'días sin ingreso a la plataforma': 'dias_sin_ingreso_plataforma',
            'cuestionarios completados': 'cuestionarios_completados',
            'foro post creados': 'Foros_creados',
            'foro repuestas realizadas': 'Foros_rta',
            'foro post leídos': 'Foros_leidos',
            'número de envíos a buzones': 'Envio_buzones',
            'modo': 'Modo'
        }
        df = df.rename(columns=renombres)

        # Normalizar categóricos
        df['Ingresado'] = df['Ingresado'].astype(str).str.strip().str.lower().replace({'sí': 'Si', 'si': 'Si', 'no': 'No'})

        # Convertir números
        columnas_numericas = ['Contenido_completado', 'Dias_sin_ingreso_curso', 'dias_sin_ingreso_plataforma',
                              'cuestionarios_completados', 'Foros_creados', 'Foros_rta', 'Foros_leidos', 'Envio_buzones']
        for col in columnas_numericas:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False).astype(float)

        # Columnas del modelo
        columnas_j48 = ["Modo", "Ingresado", "Contenido_completado", "Dias_sin_ingreso_curso",
                        "dias_sin_ingreso_plataforma", "cuestionarios_completados", "Foros_creados",
                        "Foros_rta", "Foros_leidos", "Envio_buzones"]
        df[columnas_j48].to_csv(archivo_csv, index=False)

        # Evaluar en Weka
        loader = converters.Loader(classname="weka.core.converters.CSVLoader")
        dataset = loader.load_file(archivo_csv)
        dataset.class_is_last()

        resultados = []
        for i in range(dataset.num_instances):
            instancia = dataset.get_instance(i)
            prediccion_idx = modelo.classify_instance(instancia)
            dist = modelo.distribution_for_instance(instancia)
            confianza = round(dist[int(prediccion_idx)] * 100, 2)

            etiqueta_final = "Aprobó" if confianza >= 52 else "Reprobó"

            resultado = {
                "fila": i,
                "prediccion": etiqueta_final,
                "confianza": f"{confianza}%"
            }
            if incluir_id and ids[i] is not None:
                resultado["id_estudiante"] = ids[i]

            resultados.append(resultado)

        return jsonify({"resultados": resultados})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if os.path.exists(archivo_csv):
                os.remove(archivo_csv)
        except:
            pass

if __name__ == "__main__":
    app.run(debug=True, port=5000)
