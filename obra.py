class Obra:
    """Clase que representa una obra de arte del museo"""
    
    def __init__(self, datos_obra):
        """
        Inicializa una obra de arte con todos sus atributos
        
        Args:
            datos_obra (dict): Diccionario con todos los datos de la obra
        """
        self.id = datos_obra.get("id_obra", 0)
        self.titulo = datos_obra.get("titulo", 'Sin título')
        self.artista = datos_obra.get("nombre_artista", 'Desconocido')
        self.nacionalidad = datos_obra.get('nacionalidad', 'Desconocida')
        self.año_nacimiento = datos_obra.get('año_nacimiento', '')
        self.año_muerte = datos_obra.get('año_muerte', '')
        self.clasificacion = datos_obra.get("clasificacion", 'Desconocido')
        self.fecha_creacion = datos_obra.get("fecha_obra", 'Desconocido')
        self.departamento = datos_obra.get("nombre_departamento", 'Desconocido')
        self.url_imagen = datos_obra.get("url_imagen", '')
    
    def mostrar_info_basica(self):
        """
        Imprime información básica de la obra para listados
        """
        print(f"ID: {self.id} - Título: {self.titulo} - Artista: {self.artista}")
    
    def mostrar_info_completa(self):
        """
        Imprime información completa de la obra
        """
        print(f"Título: {self.titulo}")
        print(f"Artista: {self.artista}")
        print(f"Nacionalidad: {self.nacionalidad}")
        print(f"Fechas del artista: {self.año_nacimiento}-{self.año_muerte}")
        print(f"Tipo: {self.clasificacion}")
        print(f"Año de creación: {self.fecha_creacion}")
        print(f"Departamento: {self.departamento}")
        print(f"ID de la obra: {self.id}")
