class Interfaz:
    """
    Clase para manejar la interfaz de usuario
    """
    
    def __init__(self, manejador_datos):
        """
        Inicializa la interfaz de usuario
        Argumentos:
            manejador_datos (ManejadorDatos): Instancia del manejador de datos
        """
        self.manejador = manejador_datos
    
    def mostrar_menu_principal(self):
        """
        Muestra el menú principal y maneja la interacción del usuario
        """
        while True:
            print("--- CATÁLOGO METROART ---")
            print("Menú de opciones:")
            print("1. Buscar obras por departamento")
            print("2. Buscar obras por nacionalidad del artista")
            print("3. Buscar obras por nombre del artista")
            print("4. Salir")
            # Obtener la opción que desea realizar el usuario
            opcion = input("Seleccione una opción: ")
            if opcion == "1":
                self.buscar_por_departamento()
            elif opcion == "2":
                self.buscar_por_nacionalidad()
            elif opcion == "3":
                self.buscar_por_artista()
            elif opcion == "4":
                print("Gracias por usar MetroArt. ¡Hasta pronto!")
                print("Sesión finalizada.")
                break
            else:
                print("La opción ingresada no es válida. Por favor, intente nuevamente.")
    
    def buscar_por_departamento(self):
        """
        Realiza la búsqueda de obras de arte por departamento
        """  
        if not self.manejador.departamentos_disponibles:
            self.manejador.obtener_departamentos()
        print("---")
        print("Departamentos disponibles:")
        for i, depto in enumerate(self.manejador.departamentos_disponibles, 1):
            print(f"{i}. {depto['nombre']}")
        while True:
            try:
                seleccion = int(input("Seleccione el número del departamento: "))
                # Verificar que el departamento seleccionado esté en el rango
                if 1 <= seleccion <= len(self.manejador.departamentos_disponibles):
                    id_depto = self.manejador.departamentos_disponibles[seleccion - 1]['id']
                    print("Cargando obras...")
                    # Mostrar obras si las hay
                    if self.manejador.obtener_obras_por_departamento(id_depto):
                        self.mostrar_obras(self.manejador.obras)
                        self.mostrar_detalles_obra()   
                    else:
                        print("No se encontraron obras en ese departamento.")
                    break
                # De lo contrario, seguir pidiendo un número de departamento
                else:
                    print("Selección no válida.")
            # Verificar que se haya ingresado un número entero
            except ValueError:
                print("Por favor ingrese un número válido.")
            
    def buscar_por_nacionalidad(self):
        """
        Realiza la búsqueda de obras por nacionalidad
        """
        self.manejador.obtener_nacionalidades()
        print("---")
        print("Nacionalidades disponibles:")
        for i, nacionalidad in enumerate(sorted(self.manejador.nacionalidades_disponibles), 1):
            print(f"{i}. {nacionalidad}")
        try:
            seleccion = int(input("Seleccione el número de la nacionalidad: "))
            nacionalidades_ordenadas = sorted(self.manejador.nacionalidades_disponibles)
            # Verificar que la nacionalidad seleccionada esté en el rango disponible
            if 1 <= seleccion <= len(nacionalidades_ordenadas):
                nacionalidad = nacionalidades_ordenadas[seleccion - 1]
                # Pedir el número de obras a buscar por nacionalidad
                while True:
                    try:
                        max_obras = int(input("Ingrese el número de obras del catálogo en las que buscar esta nacionalidad: "))
                    except ValueError:
                        print("Ingreso inválido. Por favor intente de nuevo.")
                        continue
                    if max_obras > 0:
                        break
                    else:
                        print("Por favor ingrese un número entero positivo.")
                resultado = self.manejador.obtener_obras_por_nacionalidad(nacionalidad, max_obras=max_obras)
                if resultado:
                    self.mostrar_obras(self.manejador.obras)
                    self.mostrar_detalles_obra()
                else:
                    print("No se encontraron obras para esta nacionalidad.")
            # De lo contrario, seguir pidiendo nacionalidad
            else:
                print("Selección no válida.")
        # Verificar que se haya ingresado un número entero
        except ValueError:
            print("Por favor ingrese un número válido.")
    
    def buscar_por_artista(self):
        """
        Realiza la búsqueda de obras por nombre del artista
        """
        try:
            artista = input("Ingrese todo o parte del nombre del artista: ")
            # Pedir el número de obras a buscar por nacionalidad
            while True:
                try:
                    max_obras = int(input("Ingrese el número de obras del catálogo en las que buscar este nombre: "))
                except ValueError:
                    print("Ingreso inválido. Por favor intente de nuevo.")
                    continue
                if max_obras > 0:
                    break
                else:
                    print("Por favor ingrese un número entero positivo.")
            resultado = self.manejador.obtener_obras_por_artista(artista, max_obras=max_obras)
            if resultado:
                self.mostrar_obras(self.manejador.obras)
                self.mostrar_detalles_obra()
            else:
                print("No se encontraron obras para ese artista.")
        # Verificar que se haya ingresado un número entero
        except ValueError:
            print("Por favor ingrese un número válido.")
    
    def mostrar_obras(self, obras):
        """
        Imprime la lista de obras encontradas para la búsqueda actual
        """
        print("---")
        print("Obras encontradas:")
        for obra in obras:
            obra.mostrar_info_basica()
    
    def mostrar_detalles_obra(self):
        """
        Muestra los detalles de una obra específica
        """
        if not self.manejador.obras:
            print("No hay obras cargadas en el sistema.")
            return    
        while True:
            id_obra = input("Ingrese el ID de una obra para ver detalles (o '0' para volver): ")
            if id_obra == '0':
                break
            try:
                id_obra = int(id_obra)
                obra = next((obra for obra in self.manejador.obras if obra.id == id_obra), None)
                if obra:
                    print("Detalles de la obra:")
                    obra.mostrar_info_completa()
                    # Mostrar imagen si se tiene una URL
                    if obra.url_imagen:
                        while True:
                            mostrar = input("¿Desea ver la imagen? (s/n): ").lower()
                            if mostrar == 's':
                                self.manejador.mostrar_imagen_obra(id_obra)
                                break
                            elif mostrar == 'n':
                                break
                            else:
                                print("Opción no válida. Intente de nuevo.")
                    else:
                        print("La obra seleccionada no tiene imagen disponible.")
                # Seguir pidiendo input hasta que se ingrese un ID válido
                else:
                    print("ID de obra no válido.")
            # Verificar que se haya ingresado un número entero
            except ValueError:
                print("Por favor ingrese un número válido.")
