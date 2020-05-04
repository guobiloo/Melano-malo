# Melano-malo

### Problema y contexto
Un melanoma maligno tiene la contrapartida de que, si es pronosticado como tal de forma tardía, será muy complicado su
tratamiento y puede producir efectos negativos en la salud de una persona. 

Sucede que la mayoría de los pacientes concurren a un dermatólogo en etapas avanzadas de un lunar, es decir, cuando los melanomas ya evidencian características malignas como son el tamaño, color (oscuros y rojizos) y sintomas de sangrado. 

Ante cualquier caso de duda es importante hacer un seguimiento del estado del melanoma, ya sea cuando un lunar no presente caracteristicas malignas antes y despues de una primera visita con un especialista, como tambien posterior al analisis. 

En el caso de detectarse lunares malignos, tambien es importante su seguimiento ya que pueden evolucionar y tornarse peor. En la etapa de observacion, el especialista de la salud no tiene posibilidad de revisar al paciente frecuentemente, sino hasta la siguiente visita.

--------------------------------

### objetivo
Este trabajo tuvo el objetivo de desarrollar un método que permita a una persona analizar el
estado de un melanoma y acudir a un médico en caso de síntomas sospechosos, o bien, que sirva de apoyo al profesional en la salud para hacer un seguimiento de la lesión.

--------------------------------

## Técnicas utilizadas
* clasificación por color y variaciones de su histograma
* clasificación por análisis bordes y varianza de Hue (del modelo de color en HSV).
 
--------------------------------
 
## Implementación del algoritmo y resultados técnicos
Con las medidas de desempeño calculadas se determinó que el porcentaje medio de aciertos del primer método fue del 75%, mientras que para el segundo fue del 81%.

Consulte mi página web personal a continuación en donde se presentan los detalles técnicos:
- [sitio web personal](https://guobiloo.github.io/Melano-Malo/)

En el siguiente link se puede acceder exclusivamente al informe que presenta los detalles técnicos de la implementación

[Informe Paper](https://github.com/guobiloo/Melano-malo/blob/master/Informe_TPfinalPDI_Gonzalez_Kalafatic.pdf)

--------------------------------

## lenguajes y librerías consideradas
* python 3.7.4
* OpenCV (librería)
* MedPy (librería)

--------------------------------

## Ejecución
Para correr el proyecto, ejecute el script main.py en un entorno de python

Dentro del script hay dos secciones: "BENIGNOS" Y "MALIGNOS". Para cada existe una serie de imágenes cargada de a pares {img,tipo}. Descomentar una de ellas y comentar el resto para realizar el analisis de un lunar en particular.

Para cada imágen de lunar, se ha dejado un comentario en aquellos melanomas incorrectamente clasificados.


-------------------------------
## Autores
* [Emiliano Kalafatic](https://github.com/abakim)
* [Joaquin Gonzalez Budiño](https://github.com/guobiloo)

## Contacto
 * Joaquin: joa_gzb@hotmail.com / joa.gzb@gmail.com
 * Emiliano: emiliano.kalafatic@gmail.com

--------------------------------

## Redes Sociales 
* [linkedin Joaquin](https://www.linkedin.com/in/joaquin-gonzalez-budino/)
* [linkedin Emiliano](https://www.linkedin.com/in/emiliano-kalafatic/)

--------------------------------

## licencia

<pre>
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Licencia Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />Esta obra está bajo una <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Licencia Creative Commons Atribución-NoComercial-CompartirIgual 4.0 Internacional</a>. 

<pre>
