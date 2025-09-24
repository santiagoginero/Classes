# MetroArt
Proyecto final de Algoritmos y Programación 2425-I
- Nicole Ugarte Pelayo
- Amelia Vergara

El presente código se encuentra en este enlace: [https://github.com/Nicole08102005/MetroArt](https://github.com/Nicole08102005/MetroArt).

---
**NOTA:** Puede ocurrir que al correr el código se obtenga el error `403 Client Error: Forbidden for url: https://collectionapi.metmuseum.org/public/collection/v1/objects/`.
Esto ocurre por problemas de acceso a la base de datos del MoMA. Si se obtiene, por favor intente correr el código nuevamente.

---
## Descripción general
El presente es un paquete de python el cual permite visualizar datos sobre obras de arte
contenidas en el Museo Metropolitano de Arte ("MoMA", por sus siglas en inglés). A continuación se describen sus módulos.

- `interfaz`: Maneja todo lo que tiene que ver con el input/output de datos en la terminal, es decir, la interacción con el usuario
- `manejador_datos`: Se encarga de interactuar con la API del MoMA, cargando información sobre las obras de arte y almacenándola en objetos
- `obra`: Implementa la clase `Obra` la cual representa una obra de arte del museo, con sus respectivos atributos
- `main`: Es la parte principal del código que ejecuta todo el programa

Estos módulos nos permiten realizar tres tareas distintas:
1. Buscar obras por departamenteo del MoMA
2. Buscar obras por la nacionalidad del artista
3. Buscar obras por el nombre del artista, ya sea el nombre completo o parte de su nombre

A su vez, se incluye un archivo, `nacionalidades.csv` que contiene las nacionalidades de artistas disponibles a buscar.
