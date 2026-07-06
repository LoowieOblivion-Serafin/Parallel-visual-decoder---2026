# Resumen de Cambios para la Paralelización y Concurrencia

Este documento detalla todas las modificaciones realizadas en el código fuente del proyecto para implementar la paralelización de datos en GPU (batching) y la concurrencia en disco (ThreadPoolExecutor).

---

## 1. Modificaciones en `src/sd_decoder.py`

### Propósito del cambio
Permitir que la función de reconstrucción reciba múltiples embeddings de fMRI a la vez (lotes), permitiendo a PyTorch ejecutar la inferencia de manera paralela en la GPU, manteniendo compatibilidad con ejecuciones individuales.

### Detalles de la modificación

#### Modificación en la firma e inferencia de `reconstruct_from_embedding`:
* **Antes (Secuencial):**
  * La función solo aceptaba un tensor de dimensiones de un único vector `(1, 768)` o `(768,)`.
  * Generaba un único `torch.Generator` para la semilla aleatoria.
  * Enviaba un único prompt y un único negative prompt a la API de `pipeline`.
  * Devolvía siempre una sola imagen PIL con `result.images[0]`.

* **Ahora (Soportando Lotes):**
  * Detecta si la entrada es un lote (`B > 1`) o un vector individual (`is_single_input`).
  * **Duplicación de Prompts:** Si se introduce un lote de tamaño $B$ y el prompt es un string simple, se replica automáticamente $B$ veces en una lista (`[prompt] * batch_size`) para emparejarse con el lote.
  * **Generador de Semillas por Elemento:** Genera una lista de generadores aleatorios (`torch.Generator`), uno para cada elemento del lote (utilizando `seed + i`), garantizando la reproducibilidad matemática de cada imagen de forma determinista.
  * **Retorno Inteligente:** Si se llamó con un solo embedding, devuelve la imagen sola (`result.images[0]`). Si se llamó con un lote, devuelve la lista completa de imágenes generadas.

---

## 2. Modificaciones en `src/phase2_run_sd.py`

### Propósito del cambio
Orquestar la lectura de embeddings agrupándolos en lotes para la GPU y guardar las imágenes resultantes en disco de manera asíncrona usando hilos de CPU, evitando que la GPU se detenga a esperar escrituras I/O.

### Detalles de la modificación

#### Modificaciones en los imports:
* Se añadió `from concurrent.futures import ThreadPoolExecutor` para habilitar el pool de hilos.

#### Modificaciones en `run_subject`:
* **Antes (Bucle Secuencial Sencillo):**
  * Iteraba uno a uno sobre la lista de embeddings.
  * Por cada embedding, llamaba a `reconstruct_from_embedding` de manera síncrona.
  * Ejecutaba `img.save(out_path)` bloqueando el hilo de ejecución hasta que la imagen se comprimiera y guardara en disco.

* **Ahora (Procesamiento en Lotes y Guardado Concurrente):**
  * **Filtrado previo:** Primero escanea qué imágenes ya existen en disco para omitirlas, calculando exactamente cuáles faltan antes de iniciar los recursos GPU.
  * **Agrupamiento en Lotes:** Agrupa la lista de embeddings pendientes utilizando el tamaño indicado por `batch_size` (mediante `torch.stack`).
  * **Inferencia Paralela:** Llama a `reconstruct_from_embedding` enviándole el lote completo de embeddings de golpe, reduciendo los tiempos de ejecución por imagen gracias al paralelismo de datos (SIMD) en la GPU.
  * **Concurrencia de I/O (Guardado Asíncrono):** Al salir las imágenes de la GPU, se encolan inmediatamente en un `ThreadPoolExecutor` con hilos dedicados para realizar el guardado en disco (`img.save`). Esto permite que el ciclo principal avance al siguiente lote en la GPU de inmediato, reduciendo considerablemente los tiempos muertos de procesamiento.

#### Modificaciones en `main` y Argumentos de Consola:
* Se añadió la opción `--batch-size` (por defecto `4`) al `argparse.ArgumentParser` para permitir la afinación de la memoria de video (VRAM) según el hardware.
* Se agregó la inyección del parámetro `batch_size` a la función `run_subject`.
* Se actualizó la bitácora (`logging`) para imprimir información sobre el tamaño de lote que se está utilizando al inicio de la ejecución.
