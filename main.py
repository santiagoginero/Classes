from manejador_datos import ManejadorDatos
from interfaz import Interfaz


class Main:
    """
    Clase principal que inicia la aplicación
    """
    
    def __init__(self):
        # Mostrar menú principal
        print("---")
        print("¡Hola! Bienvenido al Sistema de Catálogo MetroArt")
        
    def ejecutar_programa(self):
        # Inicializar componentes principales
        manejador = ManejadorDatos()
        interfaz = Interfaz(manejador)
        interfaz.mostrar_menu_principal()


if __name__ == "__main__":
    main = Main()
    main.ejecutar_programa()
