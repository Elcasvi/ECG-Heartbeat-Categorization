<h1>ECG Classifier - Kaggle Challenge</h1>
    <p>
        Este proyecto es una solución para un clasificador de electrocardiogramas (ECG), basado en un reto de Kaggle.
        El objetivo del proyecto es predecir posibles condiciones cardíacas en pacientes, utilizando técnicas de aprendizaje automático.
        La prioridad es minimizar los falsos negativos para garantizar que no se pase por alto ninguna posible condición crítica.
    </p>

<h2>Pipeline del Proyecto</h2>
<ol>
<li><strong>División de datos:</strong> Los datos se dividieron en un conjunto de <em>entrenamiento</em> y un conjunto de <em>validación</em>.</li>
<li><strong>Búsqueda de hiperparámetros óptimos:</strong> Utilizando el conjunto de entrenamiento, se realizaron diferentes pruebas para seleccionar las mejores configuraciones de hiperparámetros.
    El conjunto de validación se usó para evaluar cada configuración y seleccionar los hiperparámetros más adecuados.</li>
<li><strong>Selección de características:</strong> Una vez encontrados los mejores hiperparámetros, se utilizaron todos los datos disponibles para seleccionar las características más relevantes.</li>
<li><strong>Entrenamiento para producción:</strong> Con los hiperparámetros y las características seleccionadas, el modelo final se entrenó utilizando todos los datos, optimizándolo para su uso en producción.</li>
</ol>

<h2>App Web con FastAPI y Next.js</h2>
<p>
El modelo entrenado ha sido implementado en una aplicación web utilizando <strong>FastAPI</strong> para el backend y <strong>Next.js</strong> para el frontend. La aplicación permite a los usuarios cargar un electrocardiograma (ECG) y obtener predicciones sobre posibles condiciones cardíacas en tiempo real.
<br>
La app web está alojada en <strong>Azure Web App</strong>, lo que facilita el acceso y la escalabilidad de la solución.
</p>
<p><strong>Backend (FastAPI):</strong> Se encarga de procesar las solicitudes del cliente, aplicar el modelo entrenado y devolver los resultados.</p>
<p><strong>Frontend (Next.js):</strong> Proporciona una interfaz de usuario interactiva para que los médicos o pacientes puedan subir sus ECGs y ver los resultados.</p>


<h2>Instalación</h2>
<p>
Para ejecutar este proyecto localmente, sigue los siguientes pasos:
</p>
<pre>
<code>
git clone https://github.com/tu-repositorio/ECG-classifier.git
cd ECG-classifier
pip install -r requirements.txt
</code>
</pre>


<h2>Correcciones Realizadas</h2>
<p>
    A lo largo del desarrollo de este proyecto, se implementaron varias correcciones basadas en la retroalimentación recibida por los profesores de cada módulo. Estas correcciones ayudaron a mejorar el rendimiento y la implementación del modelo de clasificación de ECG.
</p>

<h3>Correcciones del Módulo 1: Preprocesamiento de Datos</h3>
<ul>
    <li><strong>Normalización de los datos:</strong> Inicialmente, Se escalaron los datos. Tras la retroalimentación, se revertió este proceso ya que no hacia mucho sentido debido a que escalar señales no era logico y los datos no tenian una diferencia de magniotud grande.</li>
</ul>

<h3>Correcciones del Módulo 2: Selección de Características</h3>
<ul>
    <li><strong>Uso de correlación para reducir multicolinealidad:</strong> La retroalimentación indicó que no hacia sentido hacer feature selection o incluso hacer análisis de multicolinealidad debido a que las señales deben ser tratadas de forma difrente.</li>
</ul>

<h3>Correcciones del Módulo 3: Modelado y Entrenamiento</h3>
<ul>
    <li><strong>Optimización de hiperparámetros:</strong> En lugar de usar una búsqueda manual de hiperparámetros, se implementó un <em>Grid Search</em> con validación cruzada, lo que permitió encontrar configuraciones más óptimas y mejorar el rendimiento del clasificador.</li>
</ul>

<h3>Correcciones del Módulo 4: Evaluación y Validación</h3>
<ul>
    <li><strong>Métricas adicionales:</strong> Originalmente, solo se utilizaba la exactitud (accuracy) como métrica. A sugerencia del profesor, se añadieron métricas adicionales como precisión (precision), recall y F1-score para evaluar mejor el comportamiento del modelo, considerando que el dataset estaba ligeramente desbalanceado.</li>
</ul>


