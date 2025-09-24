import csv
import requests
from PIL import Image
from obra import Obra
import io


class ManejadorDatos:
    """
    Clase para manejar la carga de datos de la API
    """

    def __init__(self):
        """
        Inicializa el manejador de datos
        """
        # Guardar la dirección URL de la API
        self.API = "https://collectionapi.metmuseum.org/public/collection/v1"
        # Inicializar listas para guardar los datos
        self.obras, self.departamentos_disponibles, self.nacionalidades_disponibles = [], [], []
        
    def obtener_departamentos(self):
        """
        Obtiene la lista de departamentos desde la API
        """
        try:
            respuesta = requests.get(f"{self.API}/departments")
            respuesta.raise_for_status()
            datos = respuesta.json()
            # Guardar departamentos disponibles en la lista
            for dept in datos.get('departments', []):
                self.departamentos_disponibles.append({'id': dept['departmentId'], 'nombre': dept['displayName']})
        except requests.exceptions.RequestException as e:
            print(f"Error al obtener departamentos: {e}")

    def obtener_nacionalidades(self, archivo="./nacionalidades.csv"):
        """
        Procedimiento que carga las nacionalidades disponibles de un archivo
        Argumentos:
            archivo (str): Dirección del archivo CSV que contiene las nacionalidades
        """
        # Cargar las nacionalidades disponibles desde el archivo
        self.nacionalidades_disponibles = []
        with open(archivo, "r") as f:
            lector = csv.reader(f)
            for fila in list(lector)[1:]:
                self.nacionalidades_disponibles.append(fila[0])
    
    def obtener_obras_por_departamento(self, id_departamento):
        """
        Obtiene obras de arte por departamento
        Argumentos:
            id_departamento (int): Identificador del departamento a buscar
        Return:
            bool: True si se encuentran obras; False de lo contrario
        """
        self.obras = []
        try:
            # Obtener IDs de obras en el departamento
            respuesta = requests.get(f"{self.API}/objects?departmentIds={id_departamento}")
            respuesta.raise_for_status()
            ids_objetos = respuesta.json().get('objectIDs', [])[:50]  # Limitar a 50 para demo
            # Obtener detalles de cada obra
            for id_obj in ids_objetos:
                obra = self.obtener_obra(id_obj)
                if obra:
                    self.obras.append(obra)
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error al obtener obras por departamento: {e}")
            return False
    
    def obtener_obras_por_nacionalidad(self, nacionalidad, max_obras=200):
        """
        Busca obras de arte por nacionalidad del artista directamente en la API
        Argumento:
            nacionalidad (str): Nacionalidad del artista a buscar
            max_obras (int): Número máximo de obras a devolver (por rendimiento)
        Return:
            bool: True si se encontraron obras, False si hubo error
        """
        self.obras = []
        try:
            # Primero obtenemos todos los IDs de obras disponibles
            respuesta = requests.get(f"{self.API}/objects")
            respuesta.raise_for_status()
            todos_ids = respuesta.json().get("objectIDs", [])[:max_obras]
            self.obras = []
            obras_encontradas = 0
            # Iteramos por los IDs hasta encontrar suficientes obras o terminar la lista
            for obj_id in todos_ids:
                # Obtenemos los detalles de cada obra
                obra = self.obtener_obra(obj_id)
                if obra and obra.nacionalidad.lower() == nacionalidad.lower():
                    self.obras.append(obra)
                    obras_encontradas += 1
            return len(self.obras) > 0
        except requests.exceptions.RequestException as e:
            print(f"Error al buscar obras por nacionalidad: {e}")
            return False
        
    def obtener_obras_por_artista(self, artista, max_obras=200):
        """
        Busca obras de arte por nacionalidad del artista directamente en la API
        Argumento:
            artista (str): Nombre del artista a buscar
            max_obras (int): Número máximo de obras a buscar
        Return:
            bool: True si se encontraron obras, False si no
        """
        self.obras = []
        try:
            # Primero obtenemos todos los IDs de obras disponibles
            respuesta = requests.get(f"{self.API}/objects")
            respuesta.raise_for_status()
            todos_ids = respuesta.json().get("objectIDs", [])[:max_obras]
            self.obras = []
            obras_encontradas = 0
            # Iteramos por los IDs hasta encontrar suficientes obras o terminar la lista
            for obj_id in todos_ids:
                # Obtenemos los detalles de cada obra
                obra = self.obtener_obra(obj_id)
                if obra and artista.lower() in obra.artista.lower():
                    self.obras.append(obra)
                    obras_encontradas += 1
            return len(self.obras) > 0
        except requests.exceptions.RequestException as e:
            print(f"Error al buscar obras por artista: {e}")
            return False
    
    def obtener_obra(self, id_objeto):
        """
        Obtiene detalles de una obra específica
        Argumentos:
            id_objeto (int): Identificación de la obra cuyos detalles se quieren mostrar
        """
        try:
            respuesta = requests.get(f"{self.API}/objects/{id_objeto}")
            respuesta.raise_for_status()
            datos = respuesta.json()
            # Guardar datos en una instancia de la clase Obra y retornar
            return Obra({
                'id_obra': id_objeto,
                'titulo': datos.get('title', 'Sin título'),
                'nombre_artista': datos.get('artistDisplayName', 'Desconocido'),
                'nacionalidad': datos.get('artistNationality', 'Desconocida'),
                'año_nacimiento': datos.get('artistBeginDate', ''),
                'año_muerte': datos.get('artistEndDate', ''),
                'clasificacion': datos.get('classification', 'Desconocido'),
                'fecha_obra': datos.get('objectDate', 'Desconocido'),
                'nombre_departamento': datos.get('department', 'Desconocido'),
                'url_imagen': datos.get('primaryImage', '')
            })
        except requests.exceptions.RequestException as e:
            print(f"Error al obtener detalles de la obra {id_objeto}: {e}")
            return None
    
    def mostrar_imagen_obra(self, id_obra):
        """
        Muestra la imagen de una obra específica
        Argumentos:
            id_obra (int): ID de la obra a buscar
        """
        obra = next((obra for obra in self.obras if obra.id == id_obra), None)
        if not obra or not obra.url_imagen:
            print("No se encontró la obra o no tiene imagen disponible.")
            return
        try:
            respuesta = requests.get(obra.url_imagen, stream=True)
            respuesta.raise_for_status()
            imagen = Image.open(io.BytesIO(respuesta.content))
            imagen.show()
        except requests.exceptions.RequestException as e:
            print(f"Error al obtener la imagen: {e}")
        except Exception as e:
            print(f"Error al mostrar la imagen: {e}")
